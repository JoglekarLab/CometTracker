"""The frozen causal temporal input contract used by training and inference."""

from __future__ import annotations

import numpy as np


def temporal_median_background(movie: np.ndarray) -> np.ndarray:
    """One movie-level static background, matching the verified V4 labeler."""
    movie = np.asarray(movie)
    step = max(1, len(movie) // 40)
    return np.median(movie[::step].astype(np.float32), axis=0).astype(np.float32)


def robust_normalize_joint(
    values: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.7,
) -> np.ndarray:
    """Normalize every selected temporal channel with one shared intensity map."""
    data = np.asarray(values, dtype=np.float32)
    lo, hi = np.percentile(data, (low_percentile, high_percentile))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def causal_rgb_pair(
    movie: np.ndarray,
    center: int,
    background: np.ndarray | None = None,
    background_blend: float = 0.5,
    normalization_context: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the agreed causal inputs for targets at ``t`` and ``t+1``.

    The four underlying source frames are normalized jointly so repeated
    frames have exactly the same numerical values in both pseudo-RGB inputs.
    ``background_blend=0`` is raw normalized signal; ``1`` is fully positive
    temporal-median residual; ``0.5`` is the approved conservative blend.
    """
    movie = np.asarray(movie)
    if center - 2 < 0 or center + 1 >= len(movie):
        raise IndexError("center must provide frames t-2 through t+1")
    endpoints = np.arange(center - normalization_context, center + normalization_context + 1)
    indices = np.stack((endpoints - 2, endpoints - 1, endpoints), axis=-1)
    indices = np.clip(indices, 0, len(movie) - 1)
    frames = movie[indices].astype(np.float32)
    raw = robust_normalize_joint(frames)
    if background is None:
        background = temporal_median_background(movie)
    residual = robust_normalize_joint(np.maximum(frames - background[None, None], 0.0))
    blend = float(background_blend)
    if not 0.0 <= blend <= 1.0:
        raise ValueError("background_blend must be in [0, 1]")
    selected = np.clip((1.0 - blend) * raw + blend * residual, 0.0, 1.0)
    # The center and following endpoint are the supervised pseudo-RGB pair.
    image_t = np.moveaxis(selected[normalization_context], 0, -1).astype(np.float32)
    image_tp1 = np.moveaxis(selected[normalization_context + 1], 0, -1).astype(np.float32)
    return image_t, image_tp1
