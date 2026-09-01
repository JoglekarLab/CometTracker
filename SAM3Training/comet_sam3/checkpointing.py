"""Validated compact checkpoints layered on the immutable official SAM3 base."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch


FORMAT = "comet-sam3-adapter-v2"


def config_fingerprint(config: dict) -> str:
    """Hash semantic campaign settings while ignoring machine-specific paths."""
    keys = (
        "campaign",
        "input",
        "procedural",
        "targets",
        "model",
        "loss",
        "training",
        "validation",
        "constraints",
    )
    semantic = {key: deepcopy(config[key]) for key in keys if key in config}
    encoded = json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def adapter_state_dict(model) -> dict[str, torch.Tensor]:
    """Save exactly parameters reachable and enabled in this freeze regime."""
    state = model.state_dict()
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing = trainable - set(state)
    if missing:
        raise RuntimeError(f"trainable parameters absent from state_dict: {sorted(missing)[:10]}")
    return {name: state[name].detach().cpu() for name in sorted(trainable)}


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    config: dict,
    metrics: dict | None = None,
    include_optimizer: bool = True,
    *,
    next_epoch: int | None = None,
    next_pair_index: int = 0,
    rank: int = 0,
) -> None:
    if int(rank) != 0:
        raise RuntimeError("only rank zero may write a checkpoint")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = adapter_state_dict(model)
    base_sha = getattr(model, "base_checkpoint_sha256", None)
    if not base_sha:
        raise RuntimeError("model is missing the validated base checkpoint SHA256")
    payload = {
        "format": FORMAT,
        "base_sam3_commit": config["campaign"]["official_sam3_commit"],
        "base_checkpoint_version": config["model"]["checkpoint_version"],
        "base_checkpoint_sha256": str(base_sha),
        "config_fingerprint": config_fingerprint(config),
        "configured_epoch": int(epoch),
        "epoch": int(epoch),
        "next_epoch": int(next_epoch if next_epoch is not None else epoch + 1),
        "next_pair_index": int(next_pair_index),
        "global_step": int(global_step),
        "metrics": metrics or {},
        "adapter_keys": sorted(model_state),
        "adapter_key_sha256": hashlib.sha256(
            "\n".join(sorted(model_state)).encode()
        ).hexdigest(),
        "model": model_state,
        "rng_state": _rng_state(),
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
        payload["scheduler"] = scheduler.state_dict()
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    *,
    config: dict | None = None,
    restore_rng: bool = True,
) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError(f"unsupported comet checkpoint format: {payload.get('format')!r}")
    if config is not None:
        checks = {
            "base_sam3_commit": config["campaign"]["official_sam3_commit"],
            "base_checkpoint_version": config["model"]["checkpoint_version"],
            "config_fingerprint": config_fingerprint(config),
        }
        for key, expected in checks.items():
            if payload.get(key) != expected:
                raise ValueError(f"checkpoint {key} mismatch")
    base_sha = getattr(model, "base_checkpoint_sha256", None)
    if not base_sha or payload.get("base_checkpoint_sha256") != base_sha:
        raise ValueError("checkpoint was trained from a different SAM3 base file")

    state = payload.get("model")
    keys = payload.get("adapter_keys")
    if not isinstance(state, dict) or not isinstance(keys, list):
        raise ValueError("checkpoint is missing adapter state metadata")
    if set(keys) != set(state):
        raise ValueError("checkpoint adapter_keys do not exactly match saved tensors")
    digest = hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()
    if digest != payload.get("adapter_key_sha256"):
        raise ValueError("checkpoint adapter key digest mismatch")
    current = model.state_dict()
    for name, value in state.items():
        if name not in current:
            raise ValueError(f"checkpoint parameter is absent from model: {name}")
        if tuple(value.shape) != tuple(current[name].shape):
            raise ValueError(f"checkpoint shape mismatch for {name}")
    required_prefixes = (
        "class_embedding",
        "head_predictor.",
        "track_projector.",
        "link_scorer.",
        "sam3.transformer.",
        "sam3.segmentation_head.mask_predictor.",
    )
    absent = [prefix for prefix in required_prefixes if not any(name.startswith(prefix) for name in state)]
    if absent:
        raise ValueError(f"checkpoint omits required trained modules: {absent}")
    incompatibility = model.load_state_dict(state, strict=False)
    if incompatibility.unexpected_keys:
        raise ValueError(f"unexpected adapter parameters: {incompatibility.unexpected_keys}")
    if optimizer is not None:
        if "optimizer" not in payload:
            raise ValueError("resume checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        if "scheduler" not in payload:
            raise ValueError("resume checkpoint has no scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        _restore_rng_state(payload.get("rng_state"))
    return payload


__all__ = [
    "FORMAT",
    "adapter_state_dict",
    "config_fingerprint",
    "load_checkpoint",
    "save_checkpoint",
]
