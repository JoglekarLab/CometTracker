"""Small, dependency-light contracts shared by every training-data source."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CometInstance:
    """Partial supervision for one comet in one source-resolution tile.

    ``axis_yx`` is the canonical one-pixel centerline in local tile coordinates.
    Any uniform widening is created later by the target rasterizer; the saved
    annotation is never widened or skeletonized a second time.
    """

    instance_id: str
    track_id: str | None = None
    head_yx: tuple[float, float] | None = None
    axis_yx: np.ndarray | None = None
    head_valid: bool = False
    axis_valid: bool = False
    presence_valid: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.head_valid and self.head_yx is None:
            raise ValueError("head_valid=True requires head_yx")
        if self.axis_valid and self.axis_yx is None:
            raise ValueError("axis_valid=True requires axis_yx")
        if self.axis_yx is not None:
            axis = np.asarray(self.axis_yx, dtype=np.float32)
            if axis.ndim != 2 or axis.shape[1] != 2:
                raise ValueError("axis_yx must have shape (N, 2)")
            self.axis_yx = axis


@dataclass
class PairSample:
    """Two consecutive causal pseudo-RGB observations and partial targets.

    Images are float32 ``(H, W, 3)`` arrays in [0, 1].  They obey:

    * ``image_t = [I(t-2), I(t-1), I(t)]``
    * ``image_tp1 = [I(t-1), I(t), I(t+1)]``

    ``links`` contains explicit positive identifier pairs.  Each endpoint may
    be either the per-frame ``instance_id`` or its persistent ``track_id``;
    the loss resolves both forms.  Unlisted pairs are *not* automatically
    negatives unless link supervision is explicitly exhaustive.
    """

    sample_id: str
    source: str
    image_t: np.ndarray
    image_tp1: np.ndarray
    instances_t: list[CometInstance] = field(default_factory=list)
    instances_tp1: list[CometInstance] = field(default_factory=list)
    links: list[tuple[str, str]] = field(default_factory=list)
    exhaustive_t: bool = False
    exhaustive_tp1: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "PairSample":
        for name, image in (("image_t", self.image_t), ("image_tp1", self.image_tp1)):
            arr = np.asarray(image)
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(f"{name} must have shape (H, W, 3), got {arr.shape}")
            if arr.dtype != np.float32:
                raise ValueError(f"{name} must be float32, got {arr.dtype}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} contains non-finite values")
            if arr.min(initial=0.0) < 0.0 or arr.max(initial=1.0) > 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.image_t.shape != self.image_tp1.shape:
            raise ValueError("paired images must have identical shapes")
        return self
