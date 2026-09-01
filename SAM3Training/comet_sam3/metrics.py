"""Validation metrics expressed in original movie pixels."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from torch.nn import functional as F

from .losses import Match
from .model import PairPredictions
from .schema import PairSample


def _softargmax_source(logits: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    logits = F.interpolate(
        logits.float()[:, None], size=shape, mode="bilinear", align_corners=False
    )[:, 0]
    probability = logits.flatten(1).softmax(-1)
    yy, xx = torch.meshgrid(
        torch.arange(shape[0], device=logits.device, dtype=torch.float32),
        torch.arange(shape[1], device=logits.device, dtype=torch.float32),
        indexing="ij",
    )
    y = probability @ yy.flatten()
    x = probability @ xx.flatten()
    return torch.stack((y, x), -1).detach().cpu().numpy()


def _axis_mask(points: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, bool)
    if points is None:
        return out
    rounded = np.rint(points).astype(int)
    keep = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < shape[0])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < shape[1])
    )
    rounded = rounded[keep]
    out[rounded[:, 0], rounded[:, 1]] = True
    return out


def symmetric_centerline_distance(predicted: np.ndarray, truth: np.ndarray) -> float:
    if not predicted.any() or not truth.any():
        return float("inf")
    d_truth = distance_transform_edt(~truth)
    d_predicted = distance_transform_edt(~predicted)
    return float(0.5 * (d_truth[predicted].mean() + d_predicted[truth].mean()))


@dataclass
class MetricAccumulator:
    head_errors: list[float] = field(default_factory=list)
    axis_distances: list[float] = field(default_factory=list)
    certified_background_false_positives: int = 0
    certified_background_tiles: int = 0
    exhaustive_negative_false_positives: int = 0
    exhaustive_negative_tiles: int = 0
    positive_link_scores: list[float] = field(default_factory=list)
    negative_link_scores: list[float] = field(default_factory=list)

    @torch.no_grad()
    def update(
        self,
        model,
        predictions: PairPredictions,
        samples: list[PairSample],
        matches: list[tuple[Match, Match]],
        presence_threshold: float = 0.5,
        axis_threshold: float = 0.5,
    ) -> None:
        for batch_index, (sample, pair_matches) in enumerate(zip(samples, matches)):
            shape = tuple(sample.image_t.shape[:2])
            for frame_predictions, instances, match, exhaustive in (
                (predictions.t, sample.instances_t, pair_matches[0], sample.exhaustive_t),
                (predictions.tp1, sample.instances_tp1, pair_matches[1], sample.exhaustive_tp1),
            ):
                if exhaustive and not instances:
                    self.exhaustive_negative_tiles += 1
                    self.exhaustive_negative_false_positives += int(
                        (frame_predictions.presence_logits[batch_index].sigmoid() >= presence_threshold).sum()
                    )
                if not len(match.query_indices):
                    continue
                query_indices = match.query_indices
                target_indices = match.target_indices
                predicted_heads = _softargmax_source(
                    frame_predictions.head_logits[batch_index, query_indices], shape
                )
                axis_probability = F.interpolate(
                    frame_predictions.axis_logits[batch_index, query_indices, None].float(),
                    size=shape,
                    mode="bilinear",
                    align_corners=False,
                )[:, 0].sigmoid().cpu().numpy()
                for local_index, target_index_tensor in enumerate(target_indices):
                    target_index = int(target_index_tensor)
                    instance = instances[target_index]
                    if instance.head_valid and instance.head_yx is not None:
                        self.head_errors.append(
                            float(np.linalg.norm(predicted_heads[local_index] - np.asarray(instance.head_yx)))
                        )
                    if instance.axis_valid:
                        predicted_axis = skeletonize(axis_probability[local_index] >= axis_threshold)
                        truth_axis = _axis_mask(instance.axis_yx, shape)
                        self.axis_distances.append(
                            symmetric_centerline_distance(predicted_axis, truth_axis)
                        )

            # Count predictions inside hand-certified partial background
            # rectangles.  Pixels outside those rectangles remain unknown.
            for frame_predictions, regions in (
                (predictions.t, sample.metadata.get("certified_background_regions_t", [])),
                (predictions.tp1, sample.metadata.get("certified_background_regions_tp1", [])),
            ):
                if not regions:
                    continue
                self.certified_background_tiles += 1
                boxes = frame_predictions.boxes_cxcywh[batch_index]
                scores = frame_predictions.presence_logits[batch_index].sigmoid()
                cx = boxes[:, 0] * shape[1]
                cy = boxes[:, 1] * shape[0]
                inside = torch.zeros_like(scores, dtype=torch.bool)
                for y0, y1, x0, x1 in regions:
                    inside |= (cy >= y0) & (cy <= y1) & (cx >= x0) & (cx <= x1)
                self.certified_background_false_positives += int(
                    (inside & (scores >= presence_threshold)).sum()
                )

            def identifier_map(instances, match):
                output: dict[str, int] = {}
                for q, n in zip(match.query_indices, match.target_indices):
                    item, query = instances[int(n)], int(q)
                    output[item.instance_id] = query
                    if item.track_id is not None:
                        output[str(item.track_id)] = query
                return output

            map_t = identifier_map(sample.instances_t, pair_matches[0])
            map_p = identifier_map(sample.instances_tp1, pair_matches[1])
            if map_t and map_p:
                queries_t = sorted(set(map_t.values()))
                queries_p = sorted(set(map_p.values()))
                index_t = {query: index for index, query in enumerate(queries_t)}
                index_p = {query: index for index, query in enumerate(queries_p)}
                emb_t = predictions.t.track_embeddings[batch_index, queries_t]
                emb_p = predictions.tp1.track_embeddings[batch_index, queries_p]
                logits = (
                    model.pairwise_link_logits(emb_t[None], emb_p[None])[0]
                    .float()
                    .sigmoid()
                    .cpu()
                    .numpy()
                )
                positive_query_pairs = {
                    (map_t[left], map_p[right])
                    for left, right in map(tuple, sample.links)
                    if left in map_t and right in map_p
                }
                for left_q, right_q in sorted(positive_query_pairs):
                    self.positive_link_scores.append(
                        float(logits[index_t[left_q], index_p[right_q]])
                    )
                if sample.metadata.get(
                    "link_exhaustive",
                    sample.source in {"procedural", "unet_masks", "unet_paste"},
                ):
                    for left_q in queries_t:
                        for right_q in queries_p:
                            if (left_q, right_q) not in positive_query_pairs:
                                self.negative_link_scores.append(
                                    float(logits[index_t[left_q], index_p[right_q]])
                                )

    def summary(self) -> dict[str, float | int]:
        heads = np.asarray(self.head_errors, np.float64)
        axes = np.asarray(self.axis_distances, np.float64)
        pos = np.asarray(self.positive_link_scores, np.float64)
        neg = np.asarray(self.negative_link_scores, np.float64)
        return {
            "head_count": int(len(heads)),
            "head_median_error_pixels": float(np.median(heads)) if len(heads) else float("nan"),
            "head_p90_error_pixels": float(np.percentile(heads, 90)) if len(heads) else float("nan"),
            "axis_count": int(len(axes)),
            "axis_centerline_distance_pixels": float(np.mean(axes)) if len(axes) else float("nan"),
            "certified_background_tiles": self.certified_background_tiles,
            "certified_background_false_positives": self.certified_background_false_positives,
            "exhaustive_negative_tiles": self.exhaustive_negative_tiles,
            "exhaustive_negative_false_positives": self.exhaustive_negative_false_positives,
            "positive_link_count": int(len(pos)),
            "positive_link_mean_score": float(pos.mean()) if len(pos) else float("nan"),
            "negative_link_count": int(len(neg)),
            "negative_link_mean_score": float(neg.mean()) if len(neg) else float("nan"),
            "link_accuracy_at_0_5": float(
                (np.count_nonzero(pos >= 0.5) + np.count_nonzero(neg < 0.5)) / (len(pos) + len(neg))
            ) if len(pos) + len(neg) else float("nan"),
        }
