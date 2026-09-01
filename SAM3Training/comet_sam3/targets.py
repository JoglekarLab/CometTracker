"""Rasterize exact source-space annotations into soft, uniformly wide targets."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def soft_uniform_axis(
    centerline: np.ndarray,
    width_pixels: float = 3.0,
    edge_softness: float = 0.75,
) -> np.ndarray:
    """Create a uniform-width soft tube; there is deliberately no taper."""
    line = np.asarray(centerline, dtype=bool)
    if line.ndim != 2:
        raise ValueError("centerline must be a 2-D raster")
    if not line.any():
        return np.zeros_like(line, dtype=np.float32)
    radius = max(float(width_pixels) / 2.0, 0.5)
    distance = distance_transform_edt(~line)
    softness = max(float(edge_softness), 1e-6)
    target = 1.0 / (1.0 + np.exp((distance - radius) / softness))
    target[line] = 1.0
    return target.astype(np.float32)


def gaussian_head(
    shape: tuple[int, int],
    head_yx: tuple[float, float],
    sigma_pixels: float = 1.5,
) -> np.ndarray:
    """A subpixel Gaussian head heatmap in original source-pixel units."""
    yy, xx = np.indices(shape, dtype=np.float32)
    y, x = map(float, head_yx)
    sigma2 = max(float(sigma_pixels), 1e-6) ** 2
    return np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * sigma2)).astype(
        np.float32
    )

