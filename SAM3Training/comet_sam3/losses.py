"""Partial-label-aware matching and multitask losses for CometSAM3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from .model import FramePredictions, PairPredictions
from .schema import CometInstance, PairSample
from .targets import gaussian_head, soft_uniform_axis


@dataclass
class FrameTargets:
    boxes: torch.Tensor
    box_valid: torch.Tensor
    axis: torch.Tensor
    head: torch.Tensor
    head_yx_normalized: torch.Tensor
    axis_valid: torch.Tensor
    head_valid: torch.Tensor
    presence_valid: torch.Tensor
    instance_ids: list[str]
    track_ids: list[str | None]
    exhaustive: bool
    negative_regions_y0y1x0x1: torch.Tensor

    def __len__(self) -> int:
        return len(self.instance_ids)


@dataclass
class Match:
    query_indices: torch.Tensor
    target_indices: torch.Tensor


def _axis_raster(instance: CometInstance, shape: tuple[int, int]) -> np.ndarray:
    raster = np.zeros(shape, dtype=bool)
    if instance.axis_yx is None:
        return raster
    points = np.rint(instance.axis_yx).astype(int)
    keep = (
        (points[:, 0] >= 0)
        & (points[:, 0] < shape[0])
        & (points[:, 1] >= 0)
        & (points[:, 1] < shape[1])
    )
    points = points[keep]
    if len(points):
        raster[points[:, 0], points[:, 1]] = True
    return raster


def _instance_box(instance: CometInstance, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    points: list[np.ndarray] = []
    if instance.axis_valid and instance.axis_yx is not None and len(instance.axis_yx):
        points.append(np.asarray(instance.axis_yx, np.float32))
    if instance.head_valid and instance.head_yx is not None:
        points.append(np.asarray(instance.head_yx, np.float32).reshape(1, 2))
    if not points:
        raise ValueError(f"instance {instance.instance_id} has no spatial target")
    yx = np.concatenate(points, axis=0)
    y0, x0 = yx.min(axis=0) - 1.5
    y1, x1 = yx.max(axis=0) + 1.5
    height, width = shape
    y0, y1 = np.clip((y0, y1), 0, height - 1)
    x0, x1 = np.clip((x0, x1), 0, width - 1)
    cy = ((y0 + y1) / 2.0) / height
    cx = ((x0 + x1) / 2.0) / width
    h = max(y1 - y0 + 1.0, 1.0) / height
    w = max(x1 - x0 + 1.0, 1.0) / width
    return float(cx), float(cy), float(w), float(h)


def build_frame_targets(
    instances: list[CometInstance],
    source_shape: tuple[int, int],
    output_shape: tuple[int, int],
    exhaustive: bool,
    device: torch.device,
    axis_width: float = 3.0,
    axis_edge_softness: float = 0.75,
    head_sigma: float = 1.5,
    negative_regions_y0y1x0x1: list[list[float]] | None = None,
) -> FrameTargets:
    height, width = source_shape
    boxes, box_valid, axes, heads, head_yx = [], [], [], [], []
    axis_valid, head_valid, presence_valid = [], [], []
    for instance in instances:
        has_box = bool(instance.axis_valid or instance.head_valid)
        boxes.append(
            _instance_box(instance, source_shape)
            if has_box
            else (0.5, 0.5, 1.0, 1.0)
        )
        box_valid.append(has_box)
        line = _axis_raster(instance, source_shape)
        axes.append(
            soft_uniform_axis(line, axis_width, axis_edge_softness)
            if instance.axis_valid
            else np.zeros(source_shape, np.float32)
        )
        heads.append(
            gaussian_head(source_shape, instance.head_yx, head_sigma)
            if instance.head_valid and instance.head_yx is not None
            else np.zeros(source_shape, np.float32)
        )
        if not instance.head_valid or instance.head_yx is None:
            head_yx.append((0.0, 0.0))
        else:
            head_yx.append((instance.head_yx[0] / height, instance.head_yx[1] / width))
        axis_valid.append(instance.axis_valid)
        head_valid.append(instance.head_valid)
        presence_valid.append(instance.presence_valid)

    n = len(instances)
    if n:
        axis_tensor = torch.as_tensor(np.stack(axes), device=device, dtype=torch.float32)
        head_tensor = torch.as_tensor(np.stack(heads), device=device, dtype=torch.float32)
        axis_tensor = F.interpolate(axis_tensor[:, None], output_shape, mode="bilinear", align_corners=False)[:, 0]
        head_tensor = F.interpolate(head_tensor[:, None], output_shape, mode="bilinear", align_corners=False)[:, 0]
    else:
        axis_tensor = torch.empty((0, *output_shape), device=device)
        head_tensor = torch.empty((0, *output_shape), device=device)

    regions = np.asarray(negative_regions_y0y1x0x1 or [], dtype=np.float32).reshape(-1, 4)
    if len(regions):
        regions[:, (0, 1)] /= float(height)
        regions[:, (2, 3)] /= float(width)
        regions = np.clip(regions, 0.0, 1.0)
    return FrameTargets(
        boxes=torch.as_tensor(boxes, device=device, dtype=torch.float32).reshape(n, 4),
        box_valid=torch.as_tensor(box_valid, device=device, dtype=torch.bool),
        axis=axis_tensor,
        head=head_tensor,
        head_yx_normalized=torch.as_tensor(head_yx, device=device, dtype=torch.float32).reshape(n, 2),
        axis_valid=torch.as_tensor(axis_valid, device=device, dtype=torch.bool),
        head_valid=torch.as_tensor(head_valid, device=device, dtype=torch.bool),
        presence_valid=torch.as_tensor(presence_valid, device=device, dtype=torch.bool),
        instance_ids=[obj.instance_id for obj in instances],
        track_ids=[obj.track_id for obj in instances],
        exhaustive=bool(exhaustive),
        negative_regions_y0y1x0x1=torch.as_tensor(
            regions, device=device, dtype=torch.float32
        ).reshape(-1, 4),
    )


def _softargmax_yx(logits: torch.Tensor) -> torch.Tensor:
    # Hungarian matching and coordinate losses are numerically small and use
    # operations (notably cdist) that are not implemented for CPU/BF16 and are
    # fragile on GPU/BF16.  Keep this path explicitly float32 regardless of
    # the surrounding autocast context.
    logits = logits.float()
    q, height, width = logits.shape
    probability = logits.flatten(1).softmax(dim=-1)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=logits.device, dtype=torch.float32) / float(height),
        torch.arange(width, device=logits.device, dtype=torch.float32) / float(width),
        indexing="ij",
    )
    y = probability @ yy.flatten()
    x = probability @ xx.flatten()
    return torch.stack((y, x), dim=-1)


def match_queries(
    predictions: FramePredictions,
    targets: FrameTargets,
    sample_index: int,
    weights: dict[str, float],
) -> Match:
    if len(targets) == 0:
        empty = torch.empty(0, dtype=torch.long, device=predictions.presence_logits.device)
        return Match(empty, empty)

    with torch.no_grad():
        scores = predictions.presence_logits[sample_index].float().sigmoid()
        boxes = predictions.boxes_cxcywh[sample_index].float()
        axis_logits = predictions.axis_logits[sample_index].float()
        head_logits = predictions.head_logits[sample_index].float()
        cost = -float(weights["presence"]) * scores[:, None].expand(-1, len(targets))
        box_cost = torch.cdist(boxes, targets.boxes.float(), p=1)
        cost = cost + float(weights["box_l1"]) * box_cost * targets.box_valid[None]

        if targets.head_valid.any():
            predicted_heads = _softargmax_yx(head_logits)
            head_cost = torch.cdist(predicted_heads, targets.head_yx_normalized, p=1)
            cost = cost + float(weights["head_distance"]) * head_cost * targets.head_valid[None]

        if targets.axis_valid.any():
            pred = axis_logits.sigmoid().flatten(1)
            truth = targets.axis.flatten(1)
            intersection = torch.einsum("qp,np->qn", pred, truth)
            denominator = pred.sum(-1)[:, None] + truth.sum(-1)[None]
            dice_cost = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
            cost = cost + float(weights["axis_dice"]) * dice_cost * targets.axis_valid[None]

    rows, cols = linear_sum_assignment(cost.detach().float().cpu().numpy())
    return Match(
        query_indices=torch.as_tensor(rows, dtype=torch.long, device=cost.device),
        target_indices=torch.as_tensor(cols, dtype=torch.long, device=cost.device),
    )


def _focal_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = logits.sigmoid()
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return alpha_t * (1.0 - p_t).pow(gamma) * ce


class CometMultitaskLoss(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.match_weights = config["loss"]["matching"]
        self.weights = config["loss"]["training"]
        self.alpha = float(config["loss"]["focal_alpha"])
        self.gamma = float(config["loss"]["focal_gamma"])
        target_config = config["targets"]
        self.axis_width = float(target_config["axis_width_source_pixels"])
        self.axis_edge_softness = float(target_config["axis_edge_softness_source_pixels"])
        self.head_sigma = float(target_config["head_sigma_source_pixels"])

    def _frame_loss(
        self,
        predictions: FramePredictions,
        targets: FrameTargets,
        match: Match,
        sample_index: int,
    ) -> dict[str, torch.Tensor | None]:
        result: dict[str, torch.Tensor | None] = {
            name: None
            for name in (
                "presence",
                "box",
                "axis_focal",
                "axis_dice",
                "head_focal",
                "head_coordinate",
            )
        }
        score_logits = predictions.presence_logits[sample_index]
        score_targets = torch.full_like(score_logits, -1.0)
        if targets.exhaustive:
            score_targets.zero_()
        elif len(targets.negative_regions_y0y1x0x1):
            # A hand-drawn background rectangle certifies only that region.
            # Queries centered inside it receive a negative presence target;
            # every query outside remains unknown.  Detaching the geometric
            # gate prevents the classification objective from moving boxes.
            centers = predictions.boxes_cxcywh[sample_index, :, :2].detach()
            cx, cy = centers[:, 0], centers[:, 1]
            inside = torch.zeros_like(score_targets, dtype=torch.bool)
            for y0, y1, x0, x1 in targets.negative_regions_y0y1x0x1:
                inside |= (cy >= y0) & (cy <= y1) & (cx >= x0) & (cx <= x1)
            score_targets[inside] = 0.0
        if len(match.query_indices):
            matched_presence = targets.presence_valid[match.target_indices]
            score_targets[match.query_indices[matched_presence]] = 1.0
        valid_scores = score_targets >= 0
        if valid_scores.any():
            result["presence"] = _focal_with_logits(
                score_logits[valid_scores], score_targets[valid_scores], self.alpha, self.gamma
            ).mean()

        if not len(match.query_indices):
            return result
        q, n = match.query_indices, match.target_indices
        box_keep = targets.box_valid[n]
        if box_keep.any():
            result["box"] = F.l1_loss(
                predictions.boxes_cxcywh[sample_index, q[box_keep]],
                targets.boxes[n[box_keep]],
            )

        axis_keep = targets.axis_valid[n]
        if axis_keep.any():
            axis_logits = predictions.axis_logits[sample_index, q[axis_keep]]
            axis_targets = targets.axis[n[axis_keep]]
            result["axis_focal"] = _focal_with_logits(
                axis_logits, axis_targets, self.alpha, self.gamma
            ).mean()
            probability = axis_logits.sigmoid().flatten(1)
            truth = axis_targets.flatten(1)
            dice = 1.0 - (2.0 * (probability * truth).sum(1) + 1.0) / (
                probability.sum(1) + truth.sum(1) + 1.0
            )
            result["axis_dice"] = dice.mean()

        head_keep = targets.head_valid[n]
        if head_keep.any():
            head_logits = predictions.head_logits[sample_index, q[head_keep]]
            head_targets = targets.head[n[head_keep]]
            result["head_focal"] = _focal_with_logits(
                head_logits, head_targets, self.alpha, self.gamma
            ).mean()
            coordinates = _softargmax_yx(head_logits)
            result["head_coordinate"] = F.smooth_l1_loss(
                coordinates, targets.head_yx_normalized[n[head_keep]]
            )
        return result

    def forward(
        self,
        model,
        predictions: PairPredictions,
        samples: list[PairSample],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[tuple[Match, Match]]]:
        output_shape = tuple(predictions.t.axis_logits.shape[-2:])
        device = predictions.t.axis_logits.device
        accumulated: dict[str, list[torch.Tensor]] = {}
        matches: list[tuple[Match, Match]] = []

        for index, sample in enumerate(samples):
            source_shape = tuple(sample.image_t.shape[:2])
            target_t = build_frame_targets(
                sample.instances_t, source_shape, output_shape, sample.exhaustive_t, device,
                self.axis_width, self.axis_edge_softness, self.head_sigma,
                sample.metadata.get("certified_background_regions_t", []),
            )
            target_p = build_frame_targets(
                sample.instances_tp1, source_shape, output_shape, sample.exhaustive_tp1, device,
                self.axis_width, self.axis_edge_softness, self.head_sigma,
                sample.metadata.get("certified_background_regions_tp1", []),
            )
            match_t = match_queries(predictions.t, target_t, index, self.match_weights)
            match_p = match_queries(predictions.tp1, target_p, index, self.match_weights)
            matches.append((match_t, match_p))
            frame_losses = [
                self._frame_loss(predictions.t, target_t, match_t, index),
                self._frame_loss(predictions.tp1, target_p, match_p, index),
            ]
            for frame_loss in frame_losses:
                for name, value in frame_loss.items():
                    if value is not None:
                        accumulated.setdefault(name, []).append(value)

            link_loss = self._link_loss(model, predictions, sample, index, target_t, target_p, match_t, match_p)
            if link_loss is not None:
                accumulated.setdefault("link", []).append(link_loss)

        # Normalize each task over only the frames/pairs on which that task is
        # supervised.  This keeps the configured task weights independent of
        # how many partial-label or background samples share a batch.
        zero_by_task = {
            "presence": (predictions.t.presence_logits.sum() + predictions.tp1.presence_logits.sum()) * 0.0,
            "box": (predictions.t.boxes_cxcywh.sum() + predictions.tp1.boxes_cxcywh.sum()) * 0.0,
            "axis_focal": (predictions.t.axis_logits.sum() + predictions.tp1.axis_logits.sum()) * 0.0,
            "axis_dice": (predictions.t.axis_logits.sum() + predictions.tp1.axis_logits.sum()) * 0.0,
            "head_focal": (predictions.t.head_logits.sum() + predictions.tp1.head_logits.sum()) * 0.0,
            "head_coordinate": (predictions.t.head_logits.sum() + predictions.tp1.head_logits.sum()) * 0.0,
            "link": (predictions.t.track_embeddings.sum() + predictions.tp1.track_embeddings.sum()) * 0.0,
        }
        reduced = {
            name: torch.stack(accumulated[name]).mean()
            if accumulated.get(name)
            else zero
            for name, zero in zero_by_task.items()
        }
        total = (
            self.weights["presence_focal"] * reduced["presence"]
            + self.weights["box_l1"] * reduced["box"]
            + self.weights["axis_focal"] * reduced["axis_focal"]
            + self.weights["axis_dice"] * reduced["axis_dice"]
            + self.weights["head_focal"] * reduced["head_focal"]
            + self.weights["head_coordinate"] * reduced["head_coordinate"]
            + self.weights["link_bce"] * reduced["link"]
        )
        return total, reduced, matches

    def _link_loss(
        self,
        model,
        predictions: PairPredictions,
        sample: PairSample,
        sample_index: int,
        target_t: FrameTargets,
        target_p: FrameTargets,
        match_t: Match,
        match_p: Match,
    ) -> torch.Tensor | None:
        def identifier_map(targets: FrameTargets, match: Match) -> dict[str, int]:
            output: dict[str, int] = {}
            for q, n in zip(match.query_indices, match.target_indices):
                index, query = int(n), int(q)
                output[targets.instance_ids[index]] = query
                if targets.track_ids[index] is not None:
                    output[str(targets.track_ids[index])] = query
            return output

        map_t = identifier_map(target_t, match_t)
        map_p = identifier_map(target_p, match_p)
        known_positive = set(tuple(pair) for pair in sample.links)
        if not bool(sample.metadata.get("link_supervision_valid", True)):
            return None

        pairs: list[tuple[int, int, float]] = []
        for left_id, right_id in known_positive:
            if left_id in map_t and right_id in map_p:
                pairs.append((map_t[left_id], map_p[right_id], 1.0))

        # Different fully known tracks in procedural/old data are safe negatives.
        link_exhaustive = bool(
            sample.metadata.get(
                "link_exhaustive",
                sample.source in {"procedural", "unet_masks", "unet_paste"},
            )
        )
        if link_exhaustive:
            positive_queries = {(a, b) for a, b, _ in pairs}
            for left_q in sorted(set(map_t.values())):
                for right_q in sorted(set(map_p.values())):
                    if (left_q, right_q) not in positive_queries:
                        pairs.append((left_q, right_q, 0.0))
        if not pairs:
            return None

        left = torch.stack([predictions.t.track_embeddings[sample_index, a] for a, _, _ in pairs])
        right = torch.stack([predictions.tp1.track_embeddings[sample_index, b] for _, b, _ in pairs])
        features = torch.cat((left, right, (left - right).abs()), dim=-1)
        logits = model.link_scorer(features).squeeze(-1)
        labels = logits.new_tensor([label for _, _, label in pairs])
        return F.binary_cross_entropy_with_logits(logits, labels)
