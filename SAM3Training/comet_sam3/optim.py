"""Optimizer tiers for new heads, SAM decoder, and late-unfrozen vision."""

from __future__ import annotations

import math

import torch
from torch import nn


def _late_vision_parameters(model, count: int) -> set[int]:
    visual = model.sam3.backbone.vision_backbone
    count = min(max(int(count), 0), len(visual.trunk.blocks))
    output: set[int] = set()
    if count:
        for block in visual.trunk.blocks[-count:]:
            output.update(id(parameter) for parameter in block.parameters())
        output.update(id(parameter) for parameter in visual.trunk.ln_post.parameters())
    return output


def _no_decay_parameters(model) -> set[int]:
    output: set[int] = {id(model.class_embedding)}
    normalization = (
        nn.LayerNorm,
        nn.GroupNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.Embedding,
    )
    for module in model.modules():
        if isinstance(module, normalization):
            output.update(id(parameter) for parameter in module.parameters(recurse=False))
    for name, parameter in model.named_parameters():
        if name.endswith(".bias") or parameter.ndim <= 1:
            output.add(id(parameter))
    return output


def intended_optimizer_parameter_names(model, config: dict) -> set[str]:
    """Return reachable-now plus explicitly late-unfrozen parameter names."""
    count = int(config["training"]["freezing"]["unfreeze_upper_vision_blocks"])
    late = _late_vision_parameters(model, count)
    return {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad or id(parameter) in late
    }


def build_optimizer(model, config: dict) -> torch.optim.Optimizer:
    cfg = config["training"]["optimizer"]
    decay_value = float(cfg["weight_decay"])
    learning_rates = {
        "new_heads": float(cfg["lr_new_heads"]),
        "sam_decoder": float(cfg["lr_sam_decoder"]),
        "vision": float(cfg["lr_vision_upper"]),
    }
    intended = intended_optimizer_parameter_names(model, config)
    no_decay = _no_decay_parameters(model)
    buckets: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
    assigned: set[str] = set()
    for name, parameter in model.named_parameters():
        if name not in intended:
            continue
        if name.startswith("sam3.backbone.vision_backbone"):
            tier = "vision"
        elif name.startswith("sam3."):
            tier = "sam_decoder"
        else:
            tier = "new_heads"
        use_decay = id(parameter) not in no_decay
        buckets.setdefault((tier, use_decay), []).append(parameter)
        if name in assigned:
            raise RuntimeError(f"optimizer parameter assigned twice: {name}")
        assigned.add(name)
    if assigned != intended:
        raise RuntimeError(
            f"optimizer assignment mismatch; missing={sorted(intended - assigned)[:10]}, "
            f"extra={sorted(assigned - intended)[:10]}"
        )
    groups = []
    for (tier, use_decay), parameters in sorted(buckets.items()):
        if parameters:
            groups.append(
                {
                    "params": parameters,
                    "lr": learning_rates[tier],
                    "weight_decay": decay_value if use_decay else 0.0,
                    "group_name": f"{tier}:{'decay' if use_decay else 'no_decay'}",
                }
            )
    if not groups:
        raise RuntimeError("optimizer has no parameters")
    return torch.optim.AdamW(groups)


def optimizer_steps_per_campaign(config: dict, world_size: int = 1) -> int:
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    batch = int(config["training"]["pair_batch_size_per_gpu"])
    total = 0
    for phase in config["training"]["phases"]:
        epochs = int(phase["epochs"][1]) - int(phase["epochs"][0]) + 1
        pairs_per_rank = math.ceil(int(phase["pairs_per_epoch"]) / int(world_size))
        batches = math.ceil(pairs_per_rank / batch)
        total += epochs * math.ceil(batches / accumulation)
    return total


def build_scheduler(optimizer, config: dict, total_steps: int):
    warmup = int(config["training"]["optimizer"]["warmup_optimizer_steps"])

    def multiplier(step: int) -> float:
        if step < warmup:
            return max(step, 1) / max(warmup, 1)
        progress = min(max((step - warmup) / max(total_steps - warmup, 1), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


__all__ = [
    "build_optimizer",
    "build_scheduler",
    "intended_optimizer_parameter_names",
    "optimizer_steps_per_campaign",
]
