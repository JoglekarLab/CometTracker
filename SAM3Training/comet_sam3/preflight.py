"""Fail-fast full-loss/optimizer/memory checks for both SAM3 freeze regimes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .config import load_config, required_environment


def _git_state(repo: Path) -> tuple[str, str]:
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    return commit, dirty


def _assert_import(repo: Path) -> None:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import sam3

    imported = Path(sam3.__file__).resolve()
    if repo.resolve() not in imported.parents:
        raise RuntimeError(f"imported SAM3 from {imported}, not {repo}")


def _gradient_summary(model, prefix: str) -> dict:
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(prefix) and parameter.requires_grad
    ]
    if not selected:
        raise RuntimeError(f"no trainable parameters under {prefix}")
    gradients = [parameter.grad for _, parameter in selected if parameter.grad is not None]
    if not gradients:
        raise RuntimeError(f"no gradients under {prefix}")
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    absolute = sum(float(gradient.detach().float().abs().sum()) for gradient in gradients)
    if not finite or absolute <= 0.0:
        raise RuntimeError(f"invalid or zero gradients under {prefix}")
    return {
        "parameter_tensors": len(selected),
        "gradient_tensors": len(gradients),
        "absolute_gradient_sum": absolute,
    }


def _case(config_path: str, regime: str, report_path: str) -> None:
    from .checkpointing import adapter_state_dict
    from .data.sources import collate_pair_samples
    from .data.synthetic import SyntheticConfig, generate_synthetic_pair
    from .losses import CometMultitaskLoss
    from .model import CometSAM3
    from .optim import build_optimizer, build_scheduler, optimizer_steps_per_campaign
    from .schema import PairSample

    config = load_config(config_path)
    repo, checkpoint = required_environment(config)
    _assert_import(repo)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with native BF16 is required")
    if int(config["input"]["sam_input_size"]) != 1008:
        raise ValueError("this pinned official SAM3 build requires 1008x1008 input")

    freeze = config["training"]["freezing"]
    epoch = 1 if regime == "frozen" else int(freeze["freeze_vision_through_epoch"]) + 1
    torch.manual_seed(int(config["campaign"]["seed"]) + epoch)
    torch.cuda.manual_seed_all(int(config["campaign"]["seed"]) + epoch)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = CometSAM3.from_checkpoint(
        str(checkpoint),
        sam_input_size=1008,
        track_embedding_dim=int(config["model"]["track_embedding_dim"]),
        disable_dac_query_duplication=bool(config["model"]["disable_dac_query_duplication"]),
        device="cuda",
    )
    model.configure_trainable(
        epoch,
        unfreeze_upper_vision_blocks=int(freeze["unfreeze_upper_vision_blocks"]),
        freeze_vision_through_epoch=int(freeze["freeze_vision_through_epoch"]),
    )
    model.train()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(
        optimizer, config, optimizer_steps_per_campaign(config, world_size=1)
    )
    criterion = CometMultitaskLoss(config)
    tile = int(config["input"]["source_tile_size"])
    synthetic_config = SyntheticConfig(
        tile_size=tile,
        n_comets=(2, 2),
        frozen_distractors=(0, 0),
        branch_probability=0.0,
        background_blend=float(config["input"]["background_blend"]),
    )
    batch_size = int(config["training"]["pair_batch_size_per_gpu"])
    samples = [
        generate_synthetic_pair(
            int(config["campaign"]["seed"]) + 1000 * epoch + index,
            rotation=index % 4,
            reflect=bool(index % 2),
            config=synthetic_config,
        )
        for index in range(batch_size)
    ]
    image_t, image_p, samples = collate_pair_samples(samples)
    image_t = image_t.cuda(non_blocking=True)
    image_p = image_p.cuda(non_blocking=True)

    step_started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        predictions = model(image_t, image_p)
        loss, pieces, _matches = criterion(model, predictions, samples)
    expected_queries = int(config["model"]["number_of_queries"])
    if predictions.t.presence_logits.shape[1] != expected_queries:
        raise RuntimeError("SAM3 query count does not match campaign config")
    expected_mask = int(config["model"].get("expected_mask_output_size", 288))
    if tuple(predictions.t.axis_logits.shape[-2:]) != (expected_mask, expected_mask):
        raise RuntimeError(
            f"unexpected mask output {tuple(predictions.t.axis_logits.shape[-2:])}"
        )
    tensors = [
        predictions.t.presence_logits,
        predictions.t.boxes_cxcywh,
        predictions.t.axis_logits,
        predictions.t.head_logits,
        predictions.t.track_embeddings,
        predictions.tp1.presence_logits,
        predictions.tp1.boxes_cxcywh,
        predictions.tp1.axis_logits,
        predictions.tp1.head_logits,
        predictions.tp1.track_embeddings,
    ]
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise RuntimeError("non-finite model output")
    for boxes in (predictions.t.boxes_cxcywh, predictions.tp1.boxes_cxcywh):
        if float(boxes.min()) < 0.0 or float(boxes.max()) > 1.0:
            raise RuntimeError("SAM3 boxes left normalized [0,1] range")
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite real multitask loss")
    loss.backward()

    gradient_prefixes = {
        "class_token": "class_embedding",
        "box_head": "sam3.transformer.decoder.bbox_embed.",
        "axis_head": "sam3.segmentation_head.mask_predictor.",
        "head_head": "head_predictor.",
        "track_projector": "track_projector.",
        "link_scorer": "link_scorer.",
        "fpn_neck": "sam3.backbone.vision_backbone.convs.",
    }
    gradients = {
        name: _gradient_summary(model, prefix)
        for name, prefix in gradient_prefixes.items()
    }
    visual = model.sam3.backbone.vision_backbone
    upper_count = int(freeze["unfreeze_upper_vision_blocks"])
    upper_indices = list(range(len(visual.trunk.blocks) - upper_count, len(visual.trunk.blocks)))
    if regime == "unfrozen":
        for index in upper_indices:
            gradients[f"vision_block_{index}"] = _gradient_summary(
                model, f"sam3.backbone.vision_backbone.trunk.blocks.{index}."
            )
    else:
        for index in upper_indices:
            if any(parameter.requires_grad for parameter in visual.trunk.blocks[index].parameters()):
                raise RuntimeError("upper vision block unexpectedly trainable in frozen regime")

    gradient_norm = clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    positive_step_seconds = time.perf_counter() - step_started

    # Exercise partial certified-background semantics separately.  The region
    # covers this audit tile, but exhaustive=False proves that the region path,
    # rather than the exhaustive-empty path, supplies the negative targets.
    empty = generate_synthetic_pair(
        int(config["campaign"]["seed"]) + 9000 + epoch,
        scene_kind="empty",
        rotation=0,
        reflect=False,
        config=synthetic_config,
    )
    regional = PairSample(
        sample_id=f"preflight-regional-background-{epoch}",
        source="current",
        image_t=empty.image_t,
        image_tp1=empty.image_tp1,
        exhaustive_t=False,
        exhaustive_tp1=False,
        metadata={
            "certified_background_regions_t": [[0.0, float(tile), 0.0, float(tile)]],
            "certified_background_regions_tp1": [],
        },
    ).validate()
    bg_t, bg_p, bg_samples = collate_pair_samples([regional])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        bg_predictions = model(bg_t.cuda(), bg_p.cuda())
        bg_loss, bg_pieces, _ = criterion(model, bg_predictions, bg_samples)
    if not torch.isfinite(bg_loss) or float(bg_pieces["presence"]) <= 0.0:
        raise RuntimeError("certified regional background did not produce presence loss")
    bg_loss.backward()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    adapter = adapter_state_dict(model)
    adapter_bytes = sum(value.numel() * value.element_size() for value in adapter.values())
    optimizer_state_bytes = sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    capacity = torch.cuda.get_device_properties(0).total_memory
    if reserved > 0.92 * capacity:
        raise RuntimeError("full optimizer-step preflight reserves over 92% of GPU memory")
    report = {
        "regime": regime,
        "epoch": epoch,
        "gpu": torch.cuda.get_device_name(0),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "base_checkpoint_sha256": model.base_checkpoint_sha256,
        "parameters_total": total_parameters,
        "parameters_trainable": trainable_parameters,
        "adapter_estimated_gib": adapter_bytes / 2**30,
        "optimizer_state_gib_after_first_step": optimizer_state_bytes / 2**30,
        "source_tile": tile,
        "sam_input": 1008,
        "queries": expected_queries,
        "mask_shape": list(predictions.t.axis_logits.shape[-2:]),
        "positive_multitask_loss": float(loss.detach()),
        "regional_background_loss": float(bg_loss.detach()),
        "loss_parts": {name: float(value.detach()) for name, value in pieces.items()},
        "gradients": gradients,
        "gradient_norm_before_clip": float(gradient_norm),
        "positive_forward_backward_optimizer_seconds": positive_step_seconds,
        "peak_allocated_gib": allocated / 2**30,
        "peak_reserved_gib": reserved / 2**30,
        "gpu_capacity_gib": capacity / 2**30,
        "wall_seconds_including_model_load": time.perf_counter() - started,
    }
    Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--case", choices=("frozen", "unfrozen"))
    parser.add_argument("--case-report")
    args = parser.parse_args(argv)
    if args.case:
        if not args.case_report:
            parser.error("--case requires --case-report")
        _case(args.config, args.case, args.case_report)
        return

    config = load_config(args.config)
    from .checkpointing import config_fingerprint

    repo, checkpoint = required_environment(config)
    expected = config["campaign"]["official_sam3_commit"]
    actual, dirty = _git_state(repo)
    if actual != expected:
        raise SystemExit(f"SAM3 commit mismatch: expected {expected}, found {actual}")
    if dirty:
        raise SystemExit("SAM3 checkout has tracked local modifications")
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint is missing: {checkpoint}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA with native BF16 is required")
    _assert_import(repo)

    run_dir = Path(config["paths"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    with tempfile.TemporaryDirectory(prefix="comet-sam3-preflight-") as directory:
        for regime in ("frozen", "unfrozen"):
            report_path = Path(directory) / f"{regime}.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "comet_sam3.preflight",
                    "--config",
                    str(Path(args.config).resolve()),
                    "--case",
                    regime,
                    "--case-report",
                    str(report_path),
                ],
                check=True,
                env=os.environ.copy(),
            )
            reports.append(json.loads(report_path.read_text()))
    campaign_pairs = sum(
        (int(phase["epochs"][1]) - int(phase["epochs"][0]) + 1)
        * int(phase["pairs_per_epoch"])
        for phase in config["training"]["phases"]
    )
    summary = {
        "status": "passed",
        "sam3_commit": actual,
        "sam3_repo": str(repo),
        "checkpoint": str(checkpoint),
        "config_fingerprint": config_fingerprint(config),
        "campaign_pair_samples": campaign_pairs,
        "cases": reports,
        "runtime_note": (
            "First-step timings include warm-up and are only a conservative planning signal; "
            "the 25-epoch job is resumable and records measured epoch throughput."
        ),
    }
    output = run_dir / "preflight.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
