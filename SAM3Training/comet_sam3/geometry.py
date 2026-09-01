"""Geometry shared by real, pasted, and procedural training sources."""

from __future__ import annotations

import numpy as np


def transform_d8_image(array: np.ndarray, rotation: int, reflect: bool) -> np.ndarray:
    """Apply one of the eight lossless square symmetries to trailing axes."""
    out = np.rot90(array, int(rotation) % 4, axes=(-3, -2))
    if reflect:
        out = np.flip(out, axis=-2)
    return np.ascontiguousarray(out)


def transform_d8_yx(
    yx: np.ndarray | tuple[float, float],
    size: int,
    rotation: int,
    reflect: bool,
) -> np.ndarray:
    """Transform one or many ``(y, x)`` coordinates in a square tile."""
    points = np.asarray(yx, dtype=np.float32)
    original_shape = points.shape
    points = points.reshape(-1, 2)
    y, x = points[:, 0].copy(), points[:, 1].copy()
    k = int(rotation) % 4
    if k == 0:
        yy, xx = y, x
    elif k == 1:
        yy, xx = size - 1 - x, y
    elif k == 2:
        yy, xx = size - 1 - y, size - 1 - x
    else:
        yy, xx = x, size - 1 - y
    if reflect:
        xx = size - 1 - xx
    return np.stack((yy, xx), axis=-1).reshape(original_shape)


def crop_yx(points: np.ndarray, y0: int, x0: int, size: int) -> np.ndarray:
    """Convert full-movie points to a tile and discard points outside it."""
    local = np.asarray(points, dtype=np.float32) - np.array([y0, x0], np.float32)
    keep = (
        (local[:, 0] >= 0)
        & (local[:, 0] < size)
        & (local[:, 1] >= 0)
        & (local[:, 1] < size)
    )
    return local[keep]

