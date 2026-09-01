"""Track containers.

Two levels, matching plusTip's own vocabulary:

``Track``      a run of consecutive detections produced by frame-to-frame
               linking. plusTip calls these "track segments"; a growing comet
               that blinks out for a frame produces several of them.

``CompoundTrack``  several Tracks stitched by gap closing, with each gap
               labelled ``fgap`` (the microtubule paused, then kept growing) or
               ``bgap`` (it shrank back down the lattice). This is the object
               that corresponds to one microtubule plus-end over time, and the
               one the dynamics parameters are computed from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    """One run of consecutive linked detections."""

    track_id: int
    det_index: list[int] = field(default_factory=list)
    frames: list[int] = field(default_factory=list)
    measured: list[np.ndarray] = field(default_factory=list)
    filtered: np.ndarray | None = None
    smoothed: np.ndarray | None = None
    kalman: object | None = None

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def start_frame(self) -> int:
        return self.frames[0]

    @property
    def end_frame(self) -> int:
        return self.frames[-1]

    def positions(self, source: str = "smoothed") -> np.ndarray:
        """(n, 2) positions in (row, col).

        ``smoothed`` is the RTS output when available and is what should be used
        for anything quantitative. ``measured`` is the raw mask centroid and is
        what the linker actually saw.
        """
        if source == "measured" or self.smoothed is None and self.filtered is None:
            return np.asarray(self.measured, np.float64).reshape(-1, 2)
        if source == "smoothed" and self.smoothed is not None:
            return self.smoothed[:, :2]
        if self.filtered is not None:
            return self.filtered[:, :2]
        return np.asarray(self.measured, np.float64).reshape(-1, 2)

    def velocities(self, source: str = "smoothed") -> np.ndarray:
        """(n, 2) per-frame velocity estimates, px/frame."""
        arr = self.smoothed if source == "smoothed" and self.smoothed is not None \
            else self.filtered
        if arr is None:
            pos = self.positions("measured")
            if len(pos) < 2:
                return np.zeros((len(pos), 2))
            v = np.diff(pos, axis=0)
            return np.vstack([v[:1], v])
        return arr[:, 2:]

    @property
    def end_velocity(self) -> np.ndarray:
        v = self.velocities()
        return v[-1] if len(v) else np.zeros(2)

    @property
    def start_velocity(self) -> np.ndarray:
        v = self.velocities()
        return v[0] if len(v) else np.zeros(2)

    def net_displacement(self) -> float:
        p = self.positions()
        return float(np.hypot(*(p[-1] - p[0]))) if len(p) > 1 else 0.0

    def path_length(self) -> float:
        p = self.positions()
        return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0

    def straightness(self) -> float:
        path = self.path_length()
        return self.net_displacement() / path if path > 0 else 0.0


GAP_FORWARD = "fgap"
GAP_BACKWARD = "bgap"


@dataclass
class CompoundTrack:
    """Several Tracks joined by classified gaps -- one microtubule plus-end.

    ``gaps[k]`` describes the join between ``segments[k]`` and
    ``segments[k+1]``, so ``len(gaps) == len(segments) - 1``.
    """

    compound_id: int
    segments: list[Track] = field(default_factory=list)
    gaps: list[dict] = field(default_factory=list)
    motion_class: str | None = None
    mss_slope: float | None = None

    @property
    def start_frame(self) -> int:
        return self.segments[0].start_frame

    @property
    def end_frame(self) -> int:
        return self.segments[-1].end_frame

    def n_fgap(self) -> int:
        return sum(1 for g in self.gaps if g["kind"] == GAP_FORWARD)

    def n_bgap(self) -> int:
        return sum(1 for g in self.gaps if g["kind"] == GAP_BACKWARD)

    def growth_frames(self) -> int:
        """Frames spent in a growth segment. Pause and shrinkage time lives in
        the gaps, not here -- EB3 marks growing ends only, so a pausing or
        shrinking microtubule is INVISIBLE and its duration is inferred from the
        gap, never observed."""
        return sum(len(s) for s in self.segments)
