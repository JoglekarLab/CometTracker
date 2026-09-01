"""Turn SAM3 query predictions into the prediction-folder layout the ilastik
and U-Net arms already use.

WHY THIS FILE HAS NO TORCH IN IT
    SAM3 emits N object queries per frame, not a dense probability map, so
    something has to decide how a set of queries becomes the ``_prob.tif``
    that ``comparemodels.py`` and ``trackcompare.py`` re-threshold. That
    decision is the interesting part and it is pure array arithmetic, so it
    lives here where it can be tested without a GPU, without SAM3, and
    without the pinned CUDA-only wrapper. ``scripts/sam3_export.py`` is the
    thin torch shell that feeds it.

THE SHARED CONTRACT, per movie, identical to the other two models
    <stem>_prob.tif    uint8, (T, Y, X), zlib. round(255 * P).
                       P = max over accepted queries of
                           presence(q) * P_axis(q, pixel)
                       A pixel's value therefore still means "confidence this
                       pixel is comet", which is what re-thresholding needs.
    <stem>_labels.tif  int32, connected components of prob >= thresh with
                       area >= min_area, renumbered 1..n per frame.
    <stem>_points.csv  frame,y,x,area,peak_prob,mean_prob

    ``points_and_labels`` below is a verbatim copy of the one in
    ``ilastik_export.py``. tests/test_export.py asserts the two agree
    element-for-element on random maps, so the three models are reduced to
    detections by the same code and a difference in the numbers is a
    difference in the models.

WHAT ONLY SAM3 HAS, written beside the shared three
    <stem>_heads.csv   frame,y,x,presence,axis_peak,query,tile_y,tile_x
                       the subpixel head of every accepted query. The other
                       two models have no head - a component centroid is not
                       a plus-end - so this is a separate file and a separate
                       napari layer rather than a column in _points.csv.
    <stem>_links.csv   frame,y0,x0,y1,x1,score
                       the learned identity link from a head in frame to its
                       match in frame+1. Drawn as a vectors layer.

FRAMES THAT CANNOT BE PREDICTED
    The frozen input contract is X_t = [I(t-2), I(t-1), I(t)], so a center
    needs frames t-2 through t+1 to exist. ``predictable_centers`` returns
    exactly the usable range. Frames outside it stay zero in _prob.tif and
    are reported, never quietly filled with a neighbour.
"""

from __future__ import annotations

import csv
import os

import numpy as np

PROB_SUFFIX = "_prob.tif"
LABEL_SUFFIX = "_labels.tif"
POINT_SUFFIX = "_points.csv"
HEAD_SUFFIX = "_heads.csv"
LINK_SUFFIX = "_links.csv"

POINT_FIELDS = ["frame", "y", "x", "area", "peak_prob", "mean_prob"]
HEAD_FIELDS = [
    "frame", "y", "x", "presence", "axis_peak", "query", "tile_y", "tile_x",
]
LINK_FIELDS = ["frame", "y0", "x0", "y1", "x1", "score"]


# ------------------------------------------------------------- tiling ------

def tile_origins(extent: int, tile: int, stride: int) -> list[int]:
    """Origins covering ``extent`` with ``tile``-wide windows.

    The last origin is snapped flush to the far edge instead of running off
    it, so every pixel is covered by at least one tile and the final overlap
    is wider than ``stride`` rather than the frame being cropped.

    A frame narrower than one tile gives ``[0]``, which is a tile that does
    not fit. The caller refuses that case rather than padding it: training's
    ``_crop_origin`` raises on the same condition, and a padded edge is a
    stretch of black the model never saw.
    """
    extent, tile, stride = int(extent), int(tile), int(stride)
    if tile <= 0 or stride <= 0:
        raise ValueError("tile and stride must be positive")
    if extent <= tile:
        return [0]
    origins = list(range(0, extent - tile + 1, stride))
    if origins[-1] != extent - tile:
        origins.append(extent - tile)
    return origins


def paste_max(canvas: np.ndarray, tile: np.ndarray, y0: int, x0: int) -> None:
    """Elementwise max of ``tile`` into ``canvas`` at ``(y0, x0)``, in place.

    Max, not overwrite: where tiles overlap, a comet the model found in one
    tile must not be erased by a tile that happened to be processed later and
    saw only part of it.
    """
    y1 = min(canvas.shape[0], y0 + tile.shape[0])
    x1 = min(canvas.shape[1], x0 + tile.shape[1])
    if y1 <= y0 or x1 <= x0:
        return
    view = canvas[y0:y1, x0:x1]
    np.maximum(view, tile[: y1 - y0, : x1 - x0], out=view)


def predictable_centers(n_frames: int) -> range:
    """Centers with frames ``t-2`` through ``t+1`` available."""
    n_frames = int(n_frames)
    if n_frames < 4:
        return range(0)
    return range(2, n_frames - 1)


# ------------------------------------------------------------- decode ------

def softargmax_yx(head_logits: np.ndarray) -> tuple[float, float]:
    """Subpixel ``(y, x)`` under a softmax over the whole map.

    numpy mirror of ``metrics._softargmax_source``, which is what validation
    scored the head against. Argmax of the heatmap would be a different
    number, so this deliberately is not that.
    """
    logits = np.asarray(head_logits, np.float64)
    if logits.ndim != 2:
        raise ValueError("head_logits must be a 2-D map")
    weights = np.exp(logits - logits.max())
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("head map has no finite mass")
    weights /= total
    yy, xx = np.indices(logits.shape, dtype=np.float64)
    return float((weights * yy).sum()), float((weights * xx).sum())


def dense_axis_map(presence: np.ndarray, axis_prob: np.ndarray) -> np.ndarray:
    """``(H, W)`` in [0, 1]: per pixel, the best ``presence * P_axis``.

    Max over queries and not a sum: two queries on the same comet are one
    comet, and summing would push a duplicated detection above a threshold a
    single confident one could not reach.
    """
    presence = np.asarray(presence, np.float32).reshape(-1)
    axis = np.asarray(axis_prob, np.float32)
    if axis.ndim != 3 or axis.shape[0] != presence.shape[0]:
        raise ValueError("axis_prob must be (Q, H, W) matching presence (Q,)")
    if presence.size == 0:
        return np.zeros(axis.shape[1:], np.float32)
    scaled = presence[:, None, None] * axis
    return np.clip(scaled.max(axis=0), 0.0, 1.0).astype(np.float32)


def dedupe_points(
    points: np.ndarray, scores: np.ndarray, min_distance: float
) -> np.ndarray:
    """Indices surviving greedy highest-score-first suppression.

    Overlapping tiles see the same comet twice. Without this the head count
    is a function of ``--stride``, which would make detections/frame - the
    one metric that needs no ground truth - meaningless.
    """
    points = np.asarray(points, np.float64).reshape(-1, 2)
    scores = np.asarray(scores, np.float64).reshape(-1)
    if points.shape[0] != scores.shape[0]:
        raise ValueError("points and scores must be the same length")
    if points.shape[0] == 0:
        return np.zeros((0,), np.int64)
    if min_distance <= 0:
        return np.arange(points.shape[0], dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    alive = np.ones(points.shape[0], bool)
    kept: list[int] = []
    limit = float(min_distance) ** 2
    for index in order:
        if not alive[index]:
            continue
        kept.append(int(index))
        delta = points - points[index]
        alive &= (delta[:, 0] ** 2 + delta[:, 1] ** 2) > limit
        alive[index] = False
    return np.asarray(sorted(kept), np.int64)


def match_links(scores: np.ndarray, threshold: float = 0.5) -> list[tuple[int, int, float]]:
    """Mutual-best pairs above ``threshold``, as ``(i, j, score)``.

    Mutual best rather than per-row best: a comet in frame t whose true match
    is already claimed should go unlinked, not be handed the runner-up. One
    missing link is readable in the viewer; a wrong one looks like real
    motion.
    """
    scores = np.asarray(scores, np.float64)
    if scores.ndim != 2:
        raise ValueError("scores must be (N_t, N_t+1)")
    if scores.size == 0:
        return []
    best_row = scores.argmax(axis=1)
    best_col = scores.argmax(axis=0)
    out = []
    for i, j in enumerate(best_row):
        if best_col[j] == i and scores[i, j] >= threshold:
            out.append((int(i), int(j), float(scores[i, j])))
    return out


# -------------------------------------------------------------- shared -----
# Verbatim from ilastik_export.points_and_labels. tests/test_export.py checks
# the two give identical output, so the three models are reduced to
# detections by the same rule.

def points_and_labels(prob, thresh=0.5, min_area=6):
    """(rows, labels) - one row per surviving component, labels renumbered."""
    from scipy import ndimage
    labels = np.zeros(prob.shape, np.int32)
    rows = []
    for t in range(len(prob)):
        lab, n = ndimage.label(prob[t] >= thresh)
        keep = 0
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            if ys.size < min_area:
                continue
            keep += 1
            labels[t][ys, xs] = keep
            v = prob[t][ys, xs]
            rows.append(dict(frame=t,
                             y=round(float(ys.mean()), 2),
                             x=round(float(xs.mean()), 2),
                             area=int(ys.size),
                             peak_prob=float(v.max()),
                             mean_prob=float(v.mean())))
    return rows, labels


# -------------------------------------------------------------- writing ----

def _write_csv(path: str, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(out_dir, stem, prob, rows, labels):
    """The three shared files, in the layout the existing tools read."""
    import tifffile

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, stem)
    tifffile.imwrite(base + PROB_SUFFIX,
                     (np.clip(prob, 0, 1) * 255).astype(np.uint8),
                     compression="zlib")
    tifffile.imwrite(base + LABEL_SUFFIX, labels, compression="zlib")
    _write_csv(base + POINT_SUFFIX, POINT_FIELDS, rows)
    return base


def write_heads(out_dir, stem, rows):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, stem + HEAD_SUFFIX)
    _write_csv(path, HEAD_FIELDS, rows)
    return path


def write_links(out_dir, stem, rows):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, stem + LINK_SUFFIX)
    _write_csv(path, LINK_FIELDS, rows)
    return path


__all__ = [
    "HEAD_FIELDS", "HEAD_SUFFIX", "LABEL_SUFFIX", "LINK_FIELDS", "LINK_SUFFIX",
    "POINT_FIELDS", "POINT_SUFFIX", "PROB_SUFFIX",
    "dedupe_points", "dense_axis_map", "match_links", "paste_max",
    "points_and_labels", "predictable_centers", "softargmax_yx",
    "tile_origins", "write_heads", "write_links", "write_outputs",
]
