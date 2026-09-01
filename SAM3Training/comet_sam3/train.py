"""Single-GPU, resumable final SAM3 comet training campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .checkpointing import load_checkpoint, save_checkpoint
from .config import load_config, phase_for_epoch, required_environment
from .curriculum import exact_source_counts
from .data.sources import EpochPairDataset, TrainingSources, collate_pair_samples
from .losses import CometMultitaskLoss
from .metrics import MetricAccumulator
from .model import CometSAM3
from .optim import build_optimizer, build_scheduler, optimizer_steps_per_campaign


STOP_REQUESTED = False


def _request_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"received signal {signum}; checkpointing at the next optimizer boundary", flush=True)


def _git_state(repo: Path) -> tuple[str, str]:
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    return commit, dirty


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=True) + "\n")
        handle.flush()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")


def _configure_for_epoch(model: CometSAM3, config: dict, epoch: int) -> None:
    freezing = config["training"]["freezing"]
    model.configure_trainable(
        epoch,
        unfreeze_upper_vision_blocks=int(freezing["unfreeze_upper_vision_blocks"]),
        freeze_vision_through_epoch=int(freezing["freeze_vision_through_epoch"]),
    )


def _make_loader(
    dataset: EpochPairDataset,
    config: dict,
    start_pair_index: int,
) -> DataLoader:
    if start_pair_index < 0 or start_pair_index > len(dataset):
        raise ValueError("resume pair index is outside the epoch")
    selected = (
        dataset
        if start_pair_index == 0
        else Subset(dataset, range(start_pair_index, len(dataset)))
    )
    workers = int(config["training"]["num_workers"])
    kwargs = {
        "dataset": selected,
        "batch_size": int(config["training"]["pair_batch_size_per_gpu"]),
        "shuffle": False,
        "num_workers": workers,
        "collate_fn": collate_pair_samples,
        "pin_memory": True,
        "drop_last": False,
    }
    if workers:
        kwargs.update(prefetch_factor=2, persistent_workers=False)
    return DataLoader(**kwargs)


def _move_batch(image_t: torch.Tensor, image_p: torch.Tensor):
    return (
        image_t.to("cuda", non_blocking=True),
        image_p.to("cuda", non_blocking=True),
    )


@torch.no_grad()
def _validate(
    model: CometSAM3,
    criterion: CometMultitaskLoss,
    sources: TrainingSources,
    full: bool,
) -> dict:
    model.eval()
    metrics = MetricAccumulator()
    losses: list[float] = []
    pieces: dict[str, list[float]] = defaultdict(list)
    count = 0
    for sample in sources.validation_samples(full=full):
        image_t, image_p, samples = collate_pair_samples([sample])
        image_t, image_p = _move_batch(image_t, image_p)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(image_t, image_p)
            loss, loss_parts, matches = criterion(model, predictions, samples)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite validation loss for {sample.sample_id}")
        losses.append(float(loss.detach()))
        for name, value in loss_parts.items():
            pieces[name].append(float(value.detach()))
        metrics.update(model, predictions, samples, matches)
        count += 1
    summary = metrics.summary()
    summary.update(
        {
            "validation_pairs": count,
            "validation_full": bool(full),
            "validation_loss": float(np.mean(losses)) if losses else float("nan"),
            **{
                f"validation_loss_{name}": float(np.mean(values))
                for name, values in pieces.items()
            },
        }
    )
    return summary


def _selection_key(metrics: dict) -> tuple[float, float, float, float]:
    backgrounds = max(int(metrics.get("certified_background_tiles", 0)), 1)
    false_positive_rate = float(
        metrics.get("certified_background_false_positives", math.inf)
    ) / backgrounds

    def finite_or(value, fallback):
        value = float(value)
        return value if math.isfinite(value) else fallback

    return (
        false_positive_rate,
        finite_or(metrics.get("head_p90_error_pixels", math.inf), math.inf),
        -finite_or(metrics.get("link_accuracy_at_0_5", -math.inf), -math.inf),
        finite_or(metrics.get("axis_centerline_distance_pixels", math.inf), math.inf),
    )


def _train_epoch(
    model: CometSAM3,
    criterion: CometMultitaskLoss,
    optimizer,
    scheduler,
    sources: TrainingSources,
    config: dict,
    epoch: int,
    start_pair_index: int,
    global_step: int,
    run_dir: Path,
    best_selection_key: tuple[float, ...] | None,
) -> tuple[dict, int, int, bool]:
    global STOP_REQUESTED
    model.train()
    dataset = EpochPairDataset(config, sources, epoch)
    loader = _make_loader(dataset, config, start_pair_index)
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    clip_norm = float(config["training"]["gradient_clip_norm"])
    log_every = int(config["training"]["log_every_optimizer_steps"])
    checkpoint_every = int(config["training"]["checkpoint_every_optimizer_steps"])
    optimizer.zero_grad(set_to_none=True)
    micro_batches = 0
    pairs_seen = int(start_pair_index)
    running: dict[str, float] = defaultdict(float)
    running_batches = 0
    epoch_started = time.perf_counter()
    iterator = tqdm(loader, desc=f"epoch {epoch:02d}", dynamic_ncols=True)
    for batch_number, (image_t, image_p, samples) in enumerate(iterator, start=1):
        image_t, image_p = _move_batch(image_t, image_p)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(image_t, image_p)
            loss, loss_parts, _matches = criterion(model, predictions, samples)
        if not torch.isfinite(loss):
            identifiers = [sample.sample_id for sample in samples]
            raise FloatingPointError(f"non-finite training loss for {identifiers}")
        (loss / accumulation).backward()
        micro_batches += 1
        pairs_seen += len(samples)
        running["total"] += float(loss.detach())
        for name, value in loss_parts.items():
            running[name] += float(value.detach())
        running_batches += 1

        is_last = pairs_seen >= len(dataset)
        boundary = micro_batches >= accumulation or is_last or STOP_REQUESTED
        if not boundary:
            continue
        if micro_batches < accumulation:
            correction = accumulation / max(micro_batches, 1)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        gradient_norm = clip_grad_norm_(model.parameters(), clip_norm)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("non-finite gradient norm")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        micro_batches = 0
        global_step += 1

        if global_step % log_every == 0:
            elapsed = time.perf_counter() - epoch_started
            record = {
                "event": "train_step",
                "epoch": epoch,
                "global_step": global_step,
                "pairs_seen_in_epoch": pairs_seen,
                "pairs_per_epoch": len(dataset),
                "pairs_per_second": (pairs_seen - start_pair_index) / max(elapsed, 1e-9),
                "gradient_norm": float(gradient_norm),
                "loss": running["total"] / max(running_batches, 1),
                "learning_rates": {
                    group.get("group_name", str(index)): group["lr"]
                    for index, group in enumerate(optimizer.param_groups)
                },
            }
            _append_jsonl(run_dir / "metrics.jsonl", record)
            iterator.set_postfix(loss=f"{record['loss']:.4f}", step=global_step)
        if global_step % checkpoint_every == 0 or STOP_REQUESTED:
            save_checkpoint(
                run_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                config,
                metrics={
                    "train_pairs_seen_in_epoch": pairs_seen,
                    "best_selection_key": (
                        list(best_selection_key)
                        if best_selection_key is not None
                        else None
                    ),
                },
                next_epoch=epoch,
                next_pair_index=pairs_seen,
            )
        if STOP_REQUESTED:
            break

    elapsed = time.perf_counter() - epoch_started
    summary = {
        "train_pairs": pairs_seen - start_pair_index,
        "train_seconds": elapsed,
        "train_pairs_per_second": (pairs_seen - start_pair_index) / max(elapsed, 1e-9),
        **{
            f"train_loss_{name}": value / max(running_batches, 1)
            for name, value in running.items()
        },
    }
    return summary, global_step, pairs_seen, bool(STOP_REQUESTED)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume",
        default="auto",
        help="auto, none, or an explicit comet adapter checkpoint",
    )
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args(argv)

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise SystemExit(
            "this campaign intentionally uses one GPU; task-partial batches require a "
            "separate find-unused-parameters DDP design"
        )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("a CUDA GPU with native BF16 support is required")
    for handled in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(handled, _request_stop)

    config = load_config(args.config)
    repo, checkpoint = required_environment(config)
    actual_commit, dirty = _git_state(repo)
    expected_commit = config["campaign"]["official_sam3_commit"]
    if actual_commit != expected_commit:
        raise SystemExit(f"SAM3 commit mismatch: {actual_commit} != {expected_commit}")
    if dirty:
        raise SystemExit("configured SAM3 checkout has tracked local modifications")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import sam3

    imported = Path(sam3.__file__).resolve()
    if repo.resolve() not in imported.parents:
        raise SystemExit(f"imported SAM3 from {imported}, not configured repo {repo}")

    seed = int(config["campaign"]["seed"])
    _seed_everything(seed)
    run_dir = Path(config["paths"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    sources = TrainingSources(config)
    (run_dir / "source_inventory.json").write_text(
        json.dumps(sources.inventory(), indent=2, sort_keys=True) + "\n"
    )

    model_cfg = config["model"]
    model = CometSAM3.from_checkpoint(
        str(checkpoint),
        sam_input_size=int(config["input"]["sam_input_size"]),
        track_embedding_dim=int(model_cfg["track_embedding_dim"]),
        disable_dac_query_duplication=bool(model_cfg["disable_dac_query_duplication"]),
        device="cuda",
    )
    _configure_for_epoch(model, config, 1)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(
        optimizer, config, optimizer_steps_per_campaign(config, world_size=1)
    )
    criterion = CometMultitaskLoss(config)

    last_path = run_dir / "last.pt"
    resume_path: Path | None
    if args.resume == "none":
        resume_path = None
    elif args.resume == "auto":
        resume_path = last_path if last_path.is_file() else None
    else:
        resume_path = Path(args.resume).expanduser().resolve()
    start_epoch, start_pair_index, global_step = 1, 0, 0
    best_key: tuple[float, ...] | None = None
    if resume_path is not None:
        payload = load_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler,
            config=config,
            restore_rng=True,
        )
        start_epoch = int(payload["next_epoch"])
        start_pair_index = int(payload["next_pair_index"])
        global_step = int(payload["global_step"])
        if payload.get("metrics", {}).get("best_selection_key") is not None:
            best_key = tuple(payload["metrics"]["best_selection_key"])
        print(
            f"resuming {resume_path}: epoch={start_epoch}, pair={start_pair_index}, "
            f"optimizer_step={global_step}",
            flush=True,
        )

    max_epoch = int(config["training"]["epochs"])
    if start_epoch > max_epoch:
        print("campaign is already complete", flush=True)
        return
    for epoch in range(start_epoch, max_epoch + 1):
        _configure_for_epoch(model, config, epoch)
        phase = phase_for_epoch(config, epoch)
        expected_counts = exact_source_counts(
            int(phase["pairs_per_epoch"]), phase["sources"]
        )
        print(
            f"epoch {epoch}/{max_epoch} {phase['name']}: "
            f"{phase['pairs_per_epoch']} pairs {expected_counts}",
            flush=True,
        )
        epoch_start_pair = start_pair_index if epoch == start_epoch else 0
        train_summary, global_step, pairs_seen, stopped = _train_epoch(
            model,
            criterion,
            optimizer,
            scheduler,
            sources,
            config,
            epoch,
            epoch_start_pair,
            global_step,
            run_dir,
            best_key,
        )
        if stopped:
            print(
                f"stopped safely at epoch {epoch}, pair {pairs_seen}; rerun the same "
                "sbatch command to resume automatically",
                flush=True,
            )
            return

        full = epoch in set(map(int, config["training"]["full_validation_epochs"]))
        validation = _validate(model, criterion, sources, full=full)
        epoch_record = {
            "event": "epoch_end",
            "epoch": epoch,
            "global_step": global_step,
            "phase": phase["name"],
            "source_counts": expected_counts,
            **train_summary,
            **validation,
        }
        if full:
            selection = _selection_key(validation)
            epoch_record["selection_key"] = list(selection)
            if best_key is None or selection < best_key:
                best_key = selection
                save_checkpoint(
                    run_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    config,
                    metrics={**validation, "best_selection_key": list(best_key)},
                    include_optimizer=False,
                    next_epoch=epoch + 1,
                )
                epoch_record["new_best"] = True
        _append_jsonl(run_dir / "metrics.jsonl", epoch_record)
        checkpoint_metrics = {
            **validation,
            "best_selection_key": list(best_key) if best_key is not None else None,
        }
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            global_step,
            config,
            metrics=checkpoint_metrics,
            next_epoch=epoch + 1,
            next_pair_index=0,
        )
        if epoch % int(config["training"]["save_every_epochs"]) == 0:
            save_checkpoint(
                run_dir / f"epoch_{epoch:02d}_adapter.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                config,
                metrics=checkpoint_metrics,
                include_optimizer=False,
                next_epoch=epoch + 1,
            )
        start_pair_index = 0
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            print(f"requested stop after epoch {epoch}", flush=True)
            return
    print(f"training complete; best checkpoint: {run_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
