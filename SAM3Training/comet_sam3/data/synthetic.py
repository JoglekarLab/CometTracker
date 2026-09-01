"""Deterministic on-the-fly procedural pairs for multitask comet training.

The old procedural renderer was useful, but its public sampler returned a
five-channel cleaned stack and aggregate body/head maps.  This module keeps the
directional comet profile while implementing the final data contract directly:

* ``X_t = [I(t-2), I(t-1), I(t)]`` and
  ``X_t+1 = [I(t-1), I(t), I(t+1)]``;
* exact, persistent object identities;
* an exact head and an ordered, one-pixel tail-to-head centerline;
* explicit positive links between consecutive observations;
* exhaustive empty and frozen-comet negatives;
* rare one-to-two branches and two-to-one merges with ambiguous links masked;
* slowly changing analytic backgrounds with transient blobs and wiggling
  microtubule-like filaments; and
* one shared lossless D8 transform for both observations and all geometry.

There is deliberately no post-composition noise, intensity scaling, or blur.
The variation in a procedural comet's physical parameters is part of creating
the scene, not an augmentation of a completed training example.  Soft head and
uniform-width axis targets are rasterized later by :mod:`comet_sam3.targets`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.draw import line

from comet_sam3.geometry import transform_d8_image, transform_d8_yx
from comet_sam3.preprocessing import causal_rgb_pair
from comet_sam3.schema import CometInstance, PairSample


@dataclass(frozen=True)
class SyntheticConfig:
    """Ranges for one small procedural movie used to form a pair."""

    tile_size: int = 192
    n_frames: int = 7
    center_frame: int = 3
    n_comets: tuple[int, int] = (1, 4)
    speed_pixels_per_frame: tuple[float, float] = (0.8, 4.8)
    decay_length_bins_pixels: tuple[tuple[float, float], ...] = (
        (3.0, 5.0),
        (5.0, 18.0),
        (18.0, 24.0),
        (24.0, 28.0),
    )
    decay_length_probabilities: tuple[float, ...] = (0.05, 0.70, 0.20, 0.05)
    transverse_sigma_pixels: tuple[float, float] = (0.9, 1.8)
    signal_amplitude: tuple[float, float] = (65.0, 210.0)
    axis_visibility_threshold: float = 0.32
    branch_probability: float = 0.08
    small_branch_fraction: float = 0.50
    small_branch_angle_degrees: tuple[float, float] = (10.0, 14.0)
    small_branch_min_head_separation_pixels: float = 2.5
    branch_angle_degrees: tuple[float, float] = (18.0, 55.0)
    branch_transition_min_head_separation_pixels: float = 2.5
    merge_probability: float = 0.05
    merge_approach_angle_degrees: tuple[float, float] = (35.0, 75.0)
    merge_min_parent_separation_pixels: float = 3.0
    hotspot_probability: float = 0.25
    frozen_distractors: tuple[int, int] = (0, 3)
    background_drift_pixels_per_frame: tuple[float, float] = (0.03, 0.24)
    transient_blob_probability: float = 0.45
    transient_blob_count: tuple[int, int] = (1, 4)
    transient_blob_sigma_pixels: tuple[float, float] = (2.0, 8.0)
    transient_blob_amplitude: tuple[float, float] = (10.0, 44.0)
    transient_blob_speed_pixels_per_frame: tuple[float, float] = (0.0, 0.30)
    microtubule_crowd_probability: float = 0.35
    microtubule_count: tuple[int, int] = (8, 18)
    microtubule_length_pixels: tuple[float, float] = (28.0, 92.0)
    microtubule_sigma_pixels: tuple[float, float] = (0.65, 1.30)
    microtubule_amplitude: tuple[float, float] = (7.0, 26.0)
    microtubule_wiggle_pixels: tuple[float, float] = (0.15, 0.80)
    microtubule_drift_pixels_per_frame: tuple[float, float] = (0.0, 0.12)
    background_blend: float = 0.5

    def validate(self) -> "SyntheticConfig":
        def ordered(name: str, bounds: Sequence[float], minimum: float = 0.0) -> None:
            if len(bounds) != 2:
                raise ValueError(f"{name} must contain two values")
            low, high = map(float, bounds)
            if low < minimum or high < low:
                raise ValueError(f"invalid {name} range")

        if self.tile_size < 16:
            raise ValueError("tile_size is too small")
        if self.n_frames < 4:
            raise ValueError("n_frames must provide t-2 through t+1")
        if not 2 <= self.center_frame < self.n_frames - 1:
            raise ValueError("center_frame must provide t-2 through t+1")
        if self.n_comets[0] < 0 or self.n_comets[1] < self.n_comets[0]:
            raise ValueError("invalid n_comets range")
        if (
            self.frozen_distractors[0] < 0
            or self.frozen_distractors[1] < self.frozen_distractors[0]
        ):
            raise ValueError("invalid frozen_distractors range")
        for name in ("transient_blob_count", "microtubule_count"):
            bounds = getattr(self, name)
            if bounds[0] < 0 or bounds[1] < bounds[0]:
                raise ValueError(f"invalid {name} range")
        for name in (
            "speed_pixels_per_frame",
            "transverse_sigma_pixels",
            "signal_amplitude",
            "transient_blob_sigma_pixels",
            "transient_blob_amplitude",
            "microtubule_length_pixels",
            "microtubule_sigma_pixels",
            "microtubule_amplitude",
        ):
            ordered(name, getattr(self, name), minimum=1e-8)
        if not self.decay_length_bins_pixels:
            raise ValueError("decay_length_bins_pixels cannot be empty")
        if len(self.decay_length_bins_pixels) != len(
            self.decay_length_probabilities
        ):
            raise ValueError("decay length bins and probabilities must align")
        previous_high = None
        for index, bounds in enumerate(self.decay_length_bins_pixels):
            ordered(f"decay_length_bins_pixels[{index}]", bounds, minimum=1e-8)
            low, high = map(float, bounds)
            if previous_high is not None and low < previous_high:
                raise ValueError("decay length bins must be ordered and nonoverlapping")
            previous_high = high
        decay_probabilities = np.asarray(
            self.decay_length_probabilities, dtype=np.float64
        )
        if np.any(decay_probabilities < 0.0) or not np.isclose(
            float(decay_probabilities.sum()), 1.0, atol=1e-8
        ):
            raise ValueError("decay length probabilities must be nonnegative and sum to 1")
        for name in (
            "background_drift_pixels_per_frame",
            "transient_blob_speed_pixels_per_frame",
            "microtubule_wiggle_pixels",
            "microtubule_drift_pixels_per_frame",
        ):
            ordered(name, getattr(self, name), minimum=0.0)
        for name in (
            "small_branch_angle_degrees",
            "branch_angle_degrees",
            "merge_approach_angle_degrees",
        ):
            ordered(name, getattr(self, name), minimum=1e-8)
            if float(getattr(self, name)[1]) >= 180.0:
                raise ValueError(f"{name} must remain below 180 degrees")
        for name in (
            "small_branch_min_head_separation_pixels",
            "branch_transition_min_head_separation_pixels",
            "merge_min_parent_separation_pixels",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.axis_visibility_threshold < 1.0:
            raise ValueError("axis_visibility_threshold must be in (0, 1)")
        if not 0.0 <= self.branch_probability <= 1.0:
            raise ValueError("branch_probability must be in [0, 1]")
        if not 0.0 <= self.small_branch_fraction <= 1.0:
            raise ValueError("small_branch_fraction must be in [0, 1]")
        if not 0.0 <= self.merge_probability <= 1.0:
            raise ValueError("merge_probability must be in [0, 1]")
        if self.branch_probability + self.merge_probability > 1.0:
            raise ValueError("branch and merge probabilities must sum to at most 1")
        if not 0.0 <= self.hotspot_probability <= 1.0:
            raise ValueError("hotspot_probability must be in [0, 1]")
        for name in ("transient_blob_probability", "microtubule_crowd_probability"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 <= self.background_blend <= 1.0:
            raise ValueError("background_blend must be in [0, 1]")
        positive_speed = float(self.speed_pixels_per_frame[0])
        for name in (
            "background_drift_pixels_per_frame",
            "transient_blob_speed_pixels_per_frame",
            "microtubule_drift_pixels_per_frame",
        ):
            if float(getattr(self, name)[1]) >= positive_speed:
                raise ValueError(f"{name} must stay below the minimum comet speed")
        return self


@dataclass(frozen=True)
class _TrackSpec:
    track_id: str
    head_at_center_yx: np.ndarray
    heading_yx: np.ndarray
    speed: float
    decay_length: float
    sigma: float
    amplitude: float
    hotspot: bool
    hotspot_distance: float
    branch: bool = False
    split_frame: int | None = None
    branch_angle_radians: float = 0.0
    branch_angle_kind: str | None = None


@dataclass(frozen=True)
class _MergeSpec:
    """Two parent tracks that become one new persistent child identity."""

    parents: tuple[_TrackSpec, _TrackSpec]
    child: _TrackSpec
    merge_frame: int
    merge_point_yx: np.ndarray


@dataclass(frozen=True)
class _BlobSpec:
    center_yx: np.ndarray
    velocity_yx: np.ndarray
    sigma_yx: np.ndarray
    amplitude: float
    first_frame: int
    last_frame: int
    transient: bool


@dataclass(frozen=True)
class _MicrotubuleSpec:
    center_yx: np.ndarray
    direction_yx: np.ndarray
    length: float
    bend: float
    wiggle: float
    cycles: float
    phase: float
    phase_speed: float
    drift_yx: np.ndarray
    sigma: float
    amplitude: float


def _log_uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    low, high = map(float, bounds)
    if low <= 0.0 or high < low:
        raise ValueError(f"invalid positive range {bounds}")
    if high == low:
        return low
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _sample_decay_length(
    rng: np.random.Generator, config: SyntheticConfig
) -> float:
    """Draw one comet decay length from the configured categorical mixture."""
    probabilities = np.asarray(config.decay_length_probabilities, np.float64)
    index = int(rng.choice(len(probabilities), p=probabilities))
    low, high = map(float, config.decay_length_bins_pixels[index])
    return float(rng.uniform(low, high))


def _random_velocity(
    rng: np.random.Generator, bounds: tuple[float, float]
) -> np.ndarray:
    speed = float(rng.uniform(*bounds))
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    return np.asarray([np.sin(angle), np.cos(angle)], np.float32) * speed


def _select_lineage_event(draw: float, config: SyntheticConfig) -> str | None:
    """Map one uniform draw to disjoint exact merge and branch intervals."""
    if not 0.0 <= draw < 1.0:
        raise ValueError("lineage event draw must be in [0, 1)")
    if draw < config.merge_probability:
        return "merge"
    if draw < config.merge_probability + config.branch_probability:
        return "branch"
    return None


def _blob_envelope(spec: _BlobSpec, frame: int) -> float:
    if frame < spec.first_frame or frame > spec.last_frame:
        return 0.0
    if not spec.transient:
        return 1.0
    duration = spec.last_frame - spec.first_frame + 1
    phase = (frame - spec.first_frame + 1) / (duration + 1)
    return float(np.sin(np.pi * phase) ** 2)


def _microtubule_points(
    spec: _MicrotubuleSpec,
    frame: int,
    center_frame: int,
) -> np.ndarray:
    """Sample a slowly changing parametric filament centerline.

    A straight backbone is displaced along its normal by one broad bend plus a
    low-amplitude sinusoid.  Advancing the sinusoid phase by a small amount per
    frame produces a subpixel wiggle rather than comet-like translation.
    """
    direction = np.asarray(spec.direction_yx, np.float32)
    normal = np.asarray([-direction[1], direction[0]], np.float32)
    center = spec.center_yx + spec.drift_yx * (frame - center_frame)
    count = max(12, int(np.ceil(spec.length * 1.25)))
    coordinate = np.linspace(-0.5, 0.5, count, dtype=np.float32)
    along = coordinate * spec.length
    bend = spec.bend * (4.0 * coordinate * coordinate - 1.0)
    wiggle = spec.wiggle * np.sin(
        2.0 * np.pi * spec.cycles * coordinate
        + spec.phase
        + spec.phase_speed * (frame - center_frame)
    )
    return (
        center[None, :]
        + along[:, None] * direction[None, :]
        + (bend + wiggle)[:, None] * normal[None, :]
    )


def _draw_microtubule_field(
    image: np.ndarray,
    specs: Sequence[_MicrotubuleSpec],
    frame: int,
    center_frame: int,
) -> None:
    """Render many analytic Gaussian tubes with one distance transform.

    This is not a blur augmentation.  The centerline is the object geometry;
    the Gaussian of distance to that line is its intrinsic optical width.
    """
    if not specs:
        return
    height, width = image.shape
    centerline = np.zeros((height, width), bool)
    amplitude_at_line = np.zeros((height, width), np.float32)
    sigma_at_line = np.ones((height, width), np.float32)
    for spec in specs:
        points = _microtubule_points(spec, frame, center_frame)
        rounded = np.rint(points).astype(int)
        for start, end in zip(rounded[:-1], rounded[1:]):
            rr, cc = line(int(start[0]), int(start[1]), int(end[0]), int(end[1]))
            keep = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            rr, cc = rr[keep], cc[keep]
            if not len(rr):
                continue
            replace_pixel = spec.amplitude >= amplitude_at_line[rr, cc]
            selected_r, selected_c = rr[replace_pixel], cc[replace_pixel]
            centerline[selected_r, selected_c] = True
            amplitude_at_line[selected_r, selected_c] = spec.amplitude
            sigma_at_line[selected_r, selected_c] = spec.sigma
    if not centerline.any():
        return
    distance, nearest = distance_transform_edt(~centerline, return_indices=True)
    nearest_amplitude = amplitude_at_line[nearest[0], nearest[1]]
    nearest_sigma = sigma_at_line[nearest[0], nearest[1]]
    tube = nearest_amplitude * np.exp(
        -(distance * distance) / (2.0 * nearest_sigma * nearest_sigma)
    )
    tube[distance > 3.5 * nearest_sigma] = 0.0
    image += tube.astype(np.float32)


def _dynamic_background_movie(
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create a deterministic, weakly time-varying microscope-like background."""
    size = config.tile_size
    yy, xx = np.indices((size, size), dtype=np.float32)
    phase_a, phase_b, phase_c = rng.uniform(0.0, 2.0 * np.pi, 3)
    scale_a = float(rng.uniform(31.0, 58.0))
    scale_b = float(rng.uniform(17.0, 33.0))
    drift = _random_velocity(rng, config.background_drift_pixels_per_frame)

    broad_fields = []
    for _ in range(3):
        broad_fields.append(
            {
                "center": rng.uniform(0.0, float(size), 2).astype(np.float32),
                "sigma": rng.uniform(0.20 * size, 0.48 * size, 2).astype(np.float32),
                "amplitude": float(rng.uniform(10.0, 32.0)),
                "velocity": _random_velocity(
                    rng, (0.0, config.background_drift_pixels_per_frame[1] * 0.55)
                ),
            }
        )

    blobs: list[_BlobSpec] = []
    if rng.random() < config.transient_blob_probability:
        low, high = config.transient_blob_count
        for _ in range(int(rng.integers(low, high + 1))):
            transient = bool(rng.random() < 0.70)
            if transient:
                first = int(rng.integers(0, max(1, config.n_frames - 1)))
                duration = int(rng.integers(2, config.n_frames + 1))
                last = min(config.n_frames - 1, first + duration - 1)
            else:
                first, last = 0, config.n_frames - 1
            sigma = rng.uniform(*config.transient_blob_sigma_pixels, size=2).astype(
                np.float32
            )
            margin = min(float(size) * 0.18, 3.0 * float(sigma.max()) + 2.0)
            blobs.append(
                _BlobSpec(
                    center_yx=rng.uniform(margin, size - margin, 2).astype(np.float32),
                    velocity_yx=_random_velocity(
                        rng, config.transient_blob_speed_pixels_per_frame
                    ),
                    sigma_yx=sigma,
                    amplitude=float(rng.uniform(*config.transient_blob_amplitude)),
                    first_frame=first,
                    last_frame=last,
                    transient=transient,
                )
            )

    microtubules: list[_MicrotubuleSpec] = []
    crowded = bool(rng.random() < config.microtubule_crowd_probability)
    if crowded:
        low, high = config.microtubule_count
        count = int(rng.integers(low, high + 1))
        crowd_center = rng.uniform(0.25 * size, 0.75 * size, 2).astype(np.float32)
        crowd_spread = rng.uniform(0.10 * size, 0.24 * size, 2).astype(np.float32)
        base_angle = float(rng.uniform(0.0, np.pi))
        for index in range(count):
            # Most fibers share a local bundle orientation; every fifth fiber
            # crosses it so crowded regions do not become a single stripe.
            angle_offset = np.pi / 2.0 if index % 5 == 0 else 0.0
            angle = base_angle + angle_offset + float(rng.normal(0.0, 0.22))
            direction = np.asarray([np.sin(angle), np.cos(angle)], np.float32)
            microtubules.append(
                _MicrotubuleSpec(
                    center_yx=np.clip(
                        crowd_center + rng.normal(0.0, crowd_spread, 2),
                        0.0,
                        float(size - 1),
                    ).astype(np.float32),
                    direction_yx=direction,
                    length=float(rng.uniform(*config.microtubule_length_pixels)),
                    bend=float(rng.uniform(-3.0, 3.0)),
                    wiggle=float(rng.uniform(*config.microtubule_wiggle_pixels)),
                    cycles=float(rng.uniform(0.55, 1.35)),
                    phase=float(rng.uniform(0.0, 2.0 * np.pi)),
                    phase_speed=float(rng.uniform(-0.28, 0.28)),
                    drift_yx=_random_velocity(
                        rng, config.microtubule_drift_pixels_per_frame
                    ),
                    sigma=float(rng.uniform(*config.microtubule_sigma_pixels)),
                    amplitude=float(rng.uniform(*config.microtubule_amplitude)),
                )
            )

    movie = np.empty((config.n_frames, size, size), np.float32)
    for frame in range(config.n_frames):
        offset = drift * (frame - config.center_frame)
        shifted_y = yy - offset[0]
        shifted_x = xx - offset[1]
        image = (
            115.0
            + 15.0
            * np.sin(
                (shifted_x + 0.37 * shifted_y) / scale_a * 2.0 * np.pi + phase_a
            )
            + 11.0
            * np.cos(
                (shifted_y - 0.21 * shifted_x) / scale_b * 2.0 * np.pi + phase_b
            )
            + 6.0 * np.sin((shifted_x + shifted_y) / 13.0 + phase_c)
        ).astype(np.float32)
        for field in broad_fields:
            center = field["center"] + field["velocity"] * (
                frame - config.center_frame
            )
            sigma = field["sigma"]
            image += field["amplitude"] * np.exp(
                -((yy - center[0]) / sigma[0]) ** 2
                - ((xx - center[1]) / sigma[1]) ** 2
            ).astype(np.float32)
        for blob in blobs:
            envelope = _blob_envelope(blob, frame)
            if envelope <= 0.0:
                continue
            center = blob.center_yx + blob.velocity_yx * (
                frame - config.center_frame
            )
            image += blob.amplitude * envelope * np.exp(
                -((yy - center[0]) ** 2) / (2.0 * blob.sigma_yx[0] ** 2)
                - ((xx - center[1]) ** 2) / (2.0 * blob.sigma_yx[1] ** 2)
            ).astype(np.float32)
        _draw_microtubule_field(image, microtubules, frame, config.center_frame)
        movie[frame] = np.clip(image, 1.0, None)

    frame_change = np.mean(np.abs(np.diff(movie, axis=0)), axis=(1, 2))
    metadata = {
        "background_kind": "dynamic_analytic",
        "background_drift_yx_per_frame": [float(drift[0]), float(drift[1])],
        "background_mean_abs_frame_change": float(frame_change.mean()),
        "n_background_blobs": len(blobs),
        "n_transient_background_blobs": sum(blob.transient for blob in blobs),
        "microtubule_crowded": crowded,
        "n_background_microtubules": len(microtubules),
        "background_posthoc_noise": False,
        "background_posthoc_blur": False,
    }
    return movie, metadata


def _draw_comet(
    image: np.ndarray,
    head_yx: np.ndarray,
    heading_yx: np.ndarray,
    decay_length: float,
    sigma: float,
    amplitude: float,
) -> None:
    """Add a Gaussian-headed exponential tail, matching the old procgen model."""
    height, width = image.shape
    y, x = map(float, head_yx)
    dy, dx = map(float, heading_yx)
    reach = int(min(np.ceil(3.2 * sigma + 2.2 * decay_length), max(height, width)))
    y0, y1 = max(0, int(y) - reach), min(height, int(y) + reach + 1)
    x0, x1 = max(0, int(x) - reach), min(width, int(x) + reach + 1)
    if y1 <= y0 or x1 <= x0:
        return
    py = np.arange(y0, y1, dtype=np.float32)[:, None] - y
    px = np.arange(x0, x1, dtype=np.float32)[None, :] - x
    behind = -(py * dy + px * dx)
    perpendicular2 = (py * py + px * px) - behind * behind
    np.maximum(perpendicular2, 0.0, out=perpendicular2)
    length = max(float(decay_length), 0.6)
    profile = np.where(
        behind >= 0.0,
        np.exp(np.clip(-behind / length, -60.0, 0.0)),
        np.exp(-(behind * behind) / (2.0 * sigma * sigma)),
    )
    patch = amplitude * np.exp(-perpendicular2 / (2.0 * sigma * sigma)) * profile
    image[y0:y1, x0:x1] += patch.astype(np.float32)


def _draw_hotspot(
    image: np.ndarray,
    center_yx: np.ndarray,
    sigma: float,
    amplitude: float,
) -> None:
    """Render an intrinsic trailing intensity maximum without filtering."""
    height, width = image.shape
    y, x = map(float, center_yx)
    reach = max(2, int(np.ceil(3.2 * sigma)))
    y0, y1 = max(0, int(y) - reach), min(height, int(y) + reach + 1)
    x0, x1 = max(0, int(x) - reach), min(width, int(x) + reach + 1)
    if y1 <= y0 or x1 <= x0:
        return
    py = np.arange(y0, y1, dtype=np.float32)[:, None] - y
    px = np.arange(x0, x1, dtype=np.float32)[None, :] - x
    patch = amplitude * np.exp(-(py * py + px * px) / (2.0 * sigma * sigma))
    image[y0:y1, x0:x1] += patch.astype(np.float32)


def _axis_yx(
    head_yx: np.ndarray,
    heading_yx: np.ndarray,
    decay_length: float,
    visibility_threshold: float,
    size: int,
) -> np.ndarray:
    """Ordered one-pixel centerline from the visible tail to the exact head."""
    extent = -float(decay_length) * float(np.log(visibility_threshold))
    tail_yx = np.asarray(head_yx, np.float32) - np.asarray(heading_yx, np.float32) * extent
    rr, cc = line(
        int(round(float(tail_yx[0]))),
        int(round(float(tail_yx[1]))),
        int(round(float(head_yx[0]))),
        int(round(float(head_yx[1]))),
    )
    keep = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
    return np.stack((rr[keep], cc[keep]), axis=1).astype(np.float32)


def _rotate_heading(heading_yx: np.ndarray, radians: float) -> np.ndarray:
    dy, dx = map(float, heading_yx)
    cosine, sine = float(np.cos(radians)), float(np.sin(radians))
    return np.asarray(
        [dy * cosine - dx * sine, dy * sine + dx * cosine], dtype=np.float32
    )


def _track_states(spec: _TrackSpec, frame: int, center: int) -> list[dict[str, Any]]:
    """Return one parent state or two persistent child states for a frame."""
    parent_head = spec.head_at_center_yx + spec.heading_yx * spec.speed * (frame - center)
    if not spec.branch or spec.split_frame is None or frame < spec.split_frame:
        return [
            {
                "track_id": spec.track_id,
                "parent_id": None,
                "head_yx": parent_head,
                "heading_yx": spec.heading_yx,
            }
        ]

    # ``split_frame`` is the first frame containing children.  The branch
    # origin is therefore the parent's position in the preceding frame, and
    # each child advances one full frame before its first observation.  This
    # makes shallow branches visibly separate instead of producing two
    # identical head coordinates at the transition.
    split_head = (
        spec.head_at_center_yx
        + spec.heading_yx * spec.speed * (spec.split_frame - 1 - center)
    )
    elapsed = frame - spec.split_frame + 1
    left = _rotate_heading(spec.heading_yx, spec.branch_angle_radians)
    right = _rotate_heading(spec.heading_yx, -spec.branch_angle_radians)
    return [
        {
            "track_id": f"{spec.track_id}/a",
            "parent_id": spec.track_id,
            "head_yx": split_head + left * spec.speed * elapsed,
            "heading_yx": left,
        },
        {
            "track_id": f"{spec.track_id}/b",
            "parent_id": spec.track_id,
            "head_yx": split_head + right * spec.speed * elapsed,
            "heading_yx": right,
        },
    ]


def _make_track_specs(
    rng: np.random.Generator,
    config: SyntheticConfig,
    sample_id: str,
    force_branch: bool,
    split_transition: bool,
    *,
    minimum_count: int = 0,
    allow_branch: bool = True,
) -> list[_TrackSpec]:
    low, high = config.n_comets
    count = int(rng.integers(low, high + 1))
    if force_branch:
        count = max(count, 1)
    count = max(count, int(minimum_count))
    specs: list[_TrackSpec] = []
    for index in range(count):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        heading = np.asarray([np.sin(angle), np.cos(angle)], dtype=np.float32)
        speed = rng.uniform(*config.speed_pixels_per_frame)
        length = _sample_decay_length(rng, config)
        sigma = rng.uniform(*config.transverse_sigma_pixels)
        amplitude = _log_uniform(rng, config.signal_amplitude)
        branch = bool(
            (force_branch and index == 0)
            or (
                allow_branch
                and index == 0
                and rng.random() < config.branch_probability
            )
        )
        small_branch = bool(
            branch
            and not split_transition
            and rng.random() < config.small_branch_fraction
        )
        if branch and split_transition:
            split_frame = config.center_frame + 1
        elif branch and not small_branch:
            split_frame = int(rng.integers(2, config.n_frames - 1))
        else:
            split_frame = None
        angle_bounds = (
            config.small_branch_angle_degrees
            if small_branch
            else config.branch_angle_degrees
        )
        # Configuration values describe the total opening between children;
        # each child rotates by half that amount around the parent direction.
        branch_opening = float(rng.uniform(*angle_bounds)) if branch else 0.0
        if branch and split_transition:
            # The first child observations must be spatially resolvable.  For a
            # one-frame fork their separation is 2*v*sin(opening/2), so sample
            # a feasible regular opening and speed instead of allowing two
            # nearly coincident exhaustive detection targets.
            max_speed = float(config.speed_pixels_per_frame[1])
            required_ratio = min(
                config.branch_transition_min_head_separation_pixels
                / max(2.0 * max_speed, 1e-6),
                1.0,
            )
            minimum_opening = float(2.0 * np.degrees(np.arcsin(required_ratio)))
            opening_low = max(float(angle_bounds[0]), minimum_opening)
            opening_high = float(angle_bounds[1])
            if opening_low > opening_high:
                raise ValueError(
                    "branch angle/speed ranges cannot meet the configured "
                    "transition head separation"
                )
            branch_opening = float(rng.uniform(opening_low, opening_high))
            half_angle = max(float(np.deg2rad(branch_opening / 2.0)), 1e-6)
            minimum_speed = (
                config.branch_transition_min_head_separation_pixels
                / (2.0 * np.sin(half_angle))
            )
            speed_low = max(float(config.speed_pixels_per_frame[0]), minimum_speed)
            speed = float(rng.uniform(speed_low, max_speed)) if speed_low < max_speed else max_speed
        if small_branch:
            half_angle = max(float(np.deg2rad(branch_opening / 2.0)), 1e-6)
            required_elapsed = int(
                np.ceil(
                    config.small_branch_min_head_separation_pixels
                    / (2.0 * speed * np.sin(half_angle))
                )
            )
            # The shallow fork begins before the supervised pair so its two
            # children are resolvable at both t and t+1 and can be linked by
            # their persistent child identities.
            split_frame = config.center_frame - max(required_elapsed, 1) + 1
        axis_extent = -length * np.log(config.axis_visibility_threshold)
        temporal_reach = speed * max(
            config.center_frame, config.n_frames - config.center_frame
        )
        margin = min(
            max(axis_extent + temporal_reach + 4.0, 14.0),
            config.tile_size * 0.38,
        )
        if config.tile_size - 2.0 * margin <= 2.0:
            margin = max(3.0, config.tile_size * 0.2)
        head_center = rng.uniform(margin, config.tile_size - margin, 2).astype(
            np.float32
        )
        specs.append(
            _TrackSpec(
                track_id=f"{sample_id}/track-{index:02d}",
                head_at_center_yx=head_center,
                heading_yx=heading,
                speed=float(speed),
                decay_length=float(length),
                sigma=float(sigma),
                amplitude=float(amplitude),
                hotspot=bool(rng.random() < config.hotspot_probability),
                hotspot_distance=float(rng.uniform(2.0, 7.0)),
                branch=bool(branch),
                split_frame=split_frame,
                branch_angle_radians=float(np.deg2rad(branch_opening / 2.0)),
                branch_angle_kind=("small" if small_branch else "regular")
                if branch
                else None,
            )
        )
    return specs


def _configure_merge(
    specs: Sequence[_TrackSpec],
    rng: np.random.Generator,
    config: SyntheticConfig,
    sample_id: str,
    *,
    transition_at_pair: bool,
) -> tuple[list[_TrackSpec], _MergeSpec]:
    """Make the first two tracks converge into one new child identity."""
    if len(specs) < 2:
        raise ValueError("a merge requires at least two parent tracks")
    merge_frame = (
        config.center_frame + 1
        if transition_at_pair
        else int(rng.integers(2, config.n_frames - 1))
    )
    first, second = specs[0], specs[1]
    child_angle = float(rng.uniform(0.0, 2.0 * np.pi))
    child_heading = np.asarray(
        [np.sin(child_angle), np.cos(child_angle)], dtype=np.float32
    )
    minimum_parent_speed = min(2.5, config.speed_pixels_per_frame[1])
    parent_speeds = [
        float(rng.uniform(minimum_parent_speed, config.speed_pixels_per_frame[1]))
        for _ in range(2)
    ]
    parent_headings: tuple[np.ndarray, np.ndarray] | None = None
    opening = 0.0
    for _ in range(32):
        opening = float(
            np.deg2rad(rng.uniform(*config.merge_approach_angle_degrees))
        )
        candidates = (
            _rotate_heading(child_heading, opening / 2.0),
            _rotate_heading(child_heading, -opening / 2.0),
        )
        separation = float(
            np.linalg.norm(
                candidates[0] * parent_speeds[0]
                - candidates[1] * parent_speeds[1]
            )
        )
        if separation >= config.merge_min_parent_separation_pixels:
            parent_headings = candidates
            break
    if parent_headings is None:
        # This is reachable only under an unusually restrictive custom config.
        # Use the widest permitted approach and fastest permitted parents.
        opening = float(np.deg2rad(config.merge_approach_angle_degrees[1]))
        parent_headings = (
            _rotate_heading(child_heading, opening / 2.0),
            _rotate_heading(child_heading, -opening / 2.0),
        )
        parent_speeds = [float(config.speed_pixels_per_frame[1])] * 2
    maximum_extent = max(first.decay_length, second.decay_length) * (
        -np.log(config.axis_visibility_threshold)
    )
    margin = min(
        max(maximum_extent + 4.0 * max(first.speed, second.speed) + 6.0, 24.0),
        config.tile_size * 0.36,
    )
    merge_point = rng.uniform(margin, config.tile_size - margin, 2).astype(
        np.float32
    )

    parents: list[_TrackSpec] = []
    for parent, heading, approach_speed in zip(
        (first, second), parent_headings, parent_speeds
    ):
        head_at_center = merge_point - heading * approach_speed * (
            merge_frame - config.center_frame
        )
        parents.append(
            replace(
                parent,
                head_at_center_yx=head_at_center.astype(np.float32),
                heading_yx=heading.astype(np.float32),
                speed=float(approach_speed),
                branch=False,
                split_frame=None,
                branch_angle_radians=0.0,
                branch_angle_kind=None,
            )
        )

    child_speed = float(sum(parent_speeds) / 2.0)
    child = _TrackSpec(
        track_id=f"{sample_id}/merge-00",
        head_at_center_yx=(
            merge_point
            - child_heading * child_speed * (merge_frame - config.center_frame)
        ).astype(np.float32),
        heading_yx=child_heading,
        speed=child_speed,
        decay_length=_sample_decay_length(rng, config),
        sigma=float(np.sqrt(first.sigma * second.sigma)),
        amplitude=float(np.sqrt(first.amplitude * second.amplitude)),
        hotspot=bool(rng.random() < config.hotspot_probability),
        hotspot_distance=float(rng.uniform(2.0, 7.0)),
    )
    merged_specs = [parents[0], parents[1], *list(specs[2:])]
    return merged_specs, _MergeSpec(
        parents=(parents[0], parents[1]),
        child=child,
        merge_frame=int(merge_frame),
        merge_point_yx=merge_point,
    )


def _visible_instance(
    state: Mapping[str, Any],
    spec: _TrackSpec,
    frame: int,
    config: SyntheticConfig,
) -> CometInstance | None:
    head = np.asarray(state["head_yx"], dtype=np.float32)
    if not (
        0.0 <= head[0] < config.tile_size and 0.0 <= head[1] < config.tile_size
    ):
        return None
    axis = _axis_yx(
        head,
        np.asarray(state["heading_yx"], dtype=np.float32),
        spec.decay_length,
        config.axis_visibility_threshold,
        config.tile_size,
    )
    if len(axis) == 0:
        return None
    track_id = str(state["track_id"])
    return CometInstance(
        instance_id=f"{track_id}:f{frame:03d}",
        track_id=track_id,
        head_yx=(float(head[0]), float(head[1])),
        axis_yx=axis,
        head_valid=True,
        axis_valid=True,
        presence_valid=True,
        metadata={
            "frame": int(frame),
            "parent_id": state.get("parent_id"),
            "parent_ids": state.get("parent_ids"),
            "event": state.get("event"),
            "speed_pixels_per_frame": spec.speed,
            "decay_length_pixels": spec.decay_length,
            "branch_angle_kind": spec.branch_angle_kind,
            "branch_opening_degrees": float(
                np.degrees(2.0 * spec.branch_angle_radians)
            )
            if spec.branch
            else None,
            "axis_kind": "exact_uniform_tail_to_head_centerline",
        },
    )


def _render_track_state(
    movie: np.ndarray,
    frame_instances: list[list[CometInstance]],
    frame: int,
    state: Mapping[str, Any],
    spec: _TrackSpec,
    config: SyntheticConfig,
) -> None:
    head = np.asarray(state["head_yx"], np.float32)
    heading = np.asarray(state["heading_yx"], np.float32)
    _draw_comet(
        movie[frame],
        head,
        heading,
        spec.decay_length,
        spec.sigma,
        spec.amplitude,
    )
    if spec.hotspot:
        phase = frame / max(config.n_frames - 1, 1) * np.pi
        distance = spec.hotspot_distance * float(np.sin(phase) ** 2)
        _draw_hotspot(
            movie[frame],
            head - heading * distance,
            max(0.65, spec.sigma * 0.75),
            spec.amplitude * 1.25,
        )
    instance = _visible_instance(state, spec, frame, config)
    if instance is not None:
        frame_instances[frame].append(instance)


def _render_scene(
    seed: int,
    scene_kind: str,
    config: SyntheticConfig,
    sample_id: str,
    force_branch: bool = False,
    split_transition: bool = False,
    force_merge: bool = False,
    merge_transition: bool = True,
) -> tuple[np.ndarray, list[list[CometInstance]], dict[str, Any]]:
    if force_branch and force_merge:
        raise ValueError("one scene cannot force both a branch and a merge")
    # Background and object RNG streams are isolated.  Background difficulty
    # can therefore change without silently changing any head, axis, ID, link,
    # or frozen-comet geometry for a fixed seed.
    background_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 101]))
    object_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 211]))
    frozen_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 307]))
    movie, background_metadata = _dynamic_background_movie(config, background_rng)
    frame_instances: list[list[CometInstance]] = [
        [] for _ in range(config.n_frames)
    ]

    if scene_kind not in {"positive", "empty", "frozen"}:
        raise ValueError(f"unknown synthetic scene_kind: {scene_kind}")

    # Frozen directional objects are deliberately not annotations: their shape
    # is comet-like, but their zero motion makes them hard negatives.
    if scene_kind == "frozen":
        frozen_count = max(1, int(frozen_rng.integers(1, 4)))
    elif scene_kind == "positive":
        lo, hi = config.frozen_distractors
        frozen_count = int(frozen_rng.integers(lo, hi + 1))
    else:
        frozen_count = 0
    for _ in range(frozen_count):
        angle = frozen_rng.uniform(0.0, 2.0 * np.pi)
        heading = np.asarray([np.sin(angle), np.cos(angle)], np.float32)
        head = frozen_rng.uniform(18.0, config.tile_size - 18.0, 2)
        length = _sample_decay_length(frozen_rng, config)
        sigma = frozen_rng.uniform(*config.transverse_sigma_pixels)
        amplitude = _log_uniform(frozen_rng, config.signal_amplitude)
        for frame in range(config.n_frames):
            _draw_comet(movie[frame], head, heading, length, sigma, amplitude)

    specs: list[_TrackSpec] = []
    merge_spec: _MergeSpec | None = None
    if scene_kind == "positive":
        if force_merge:
            merge_event, branch_event = True, False
        elif force_branch:
            merge_event, branch_event = False, True
        else:
            lineage_event = _select_lineage_event(float(object_rng.random()), config)
            merge_event = lineage_event == "merge"
            branch_event = lineage_event == "branch"
        specs = _make_track_specs(
            object_rng,
            config,
            sample_id,
            branch_event,
            split_transition,
            minimum_count=2 if merge_event else 0,
            allow_branch=False,
        )
        if merge_event:
            specs, merge_spec = _configure_merge(
                specs,
                object_rng,
                config,
                sample_id,
                transition_at_pair=merge_transition,
            )

        ordinary_specs = specs[2:] if merge_spec is not None else specs
        for spec in ordinary_specs:
            for frame in range(config.n_frames):
                states = _track_states(spec, frame, config.center_frame)
                for state in states:
                    _render_track_state(
                        movie, frame_instances, frame, state, spec, config
                    )

        if merge_spec is not None:
            for frame in range(config.n_frames):
                if frame < merge_spec.merge_frame:
                    for parent in merge_spec.parents:
                        state = {
                            "track_id": parent.track_id,
                            "parent_id": None,
                            "head_yx": parent.head_at_center_yx
                            + parent.heading_yx
                            * parent.speed
                            * (frame - config.center_frame),
                            "heading_yx": parent.heading_yx,
                            "event": "pre_merge_parent",
                        }
                        _render_track_state(
                            movie, frame_instances, frame, state, parent, config
                        )
                else:
                    child = merge_spec.child
                    state = {
                        "track_id": child.track_id,
                        "parent_ids": [
                            parent.track_id for parent in merge_spec.parents
                        ],
                        "head_yx": child.head_at_center_yx
                        + child.heading_yx
                        * child.speed
                        * (frame - config.center_frame),
                        "heading_yx": child.heading_yx,
                        "event": "post_merge_child",
                    }
                    _render_track_state(
                        movie, frame_instances, frame, state, child, config
                    )

    transition_ids = [
        spec.track_id
        for spec in specs
        if spec.branch and spec.split_frame == config.center_frame + 1
    ]
    merge_transition_ids = (
        [parent.track_id for parent in merge_spec.parents]
        if merge_spec is not None
        and merge_spec.merge_frame == config.center_frame + 1
        else []
    )
    metadata = {
        **background_metadata,
        "scene_kind": scene_kind,
        "seed": int(seed),
        "n_tracks": len(specs),
        "n_frozen_distractors": int(frozen_count),
        "decay_length_mixture": [
            {
                "range_pixels": [float(bounds[0]), float(bounds[1])],
                "probability": float(probability),
            }
            for bounds, probability in zip(
                config.decay_length_bins_pixels,
                config.decay_length_probabilities,
            )
        ],
        "branch_transition": bool(transition_ids),
        "branch_transition_parent_ids": transition_ids,
        "branch_openings_degrees": [
            float(np.degrees(2.0 * spec.branch_angle_radians))
            for spec in specs
            if spec.branch
        ],
        "small_branch_count": sum(
            spec.branch and spec.branch_angle_kind == "small" for spec in specs
        ),
        "merge_event": merge_spec is not None,
        "merge_transition": bool(merge_transition_ids),
        "merge_transition_parent_ids": merge_transition_ids,
        "merge_child_id": merge_spec.child.track_id
        if merge_spec is not None
        else None,
        "lineage_event": (
            "merge" if merge_spec is not None else "branch" if any(
                spec.branch for spec in specs
            ) else None
        ),
        "posthoc_noise": False,
        "posthoc_intensity_change": False,
        "posthoc_blur": False,
        "pauses": False,
    }
    return movie, frame_instances, metadata


def _transform_instance(
    instance: CometInstance,
    size: int,
    rotation: int,
    reflect: bool,
) -> CometInstance:
    head = None
    if instance.head_yx is not None:
        transformed = transform_d8_yx(instance.head_yx, size, rotation, reflect)
        head = (float(transformed[0]), float(transformed[1]))
    axis = None
    if instance.axis_yx is not None:
        axis = transform_d8_yx(instance.axis_yx, size, rotation, reflect).astype(
            np.float32
        )
    return replace(instance, head_yx=head, axis_yx=axis)


def generate_synthetic_pair(
    seed: int | np.random.Generator | None = None,
    *,
    rng: np.random.Generator | None = None,
    tile_size: int | None = None,
    scene_kind: str = "positive",
    rotation: int | None = None,
    reflect: bool | None = None,
    force_branch: bool = False,
    split_transition: bool = False,
    force_merge: bool = False,
    merge_transition: bool = True,
    sample_id: str | None = None,
    split: str = "train",
    config: SyntheticConfig | None = None,
) -> PairSample:
    """Generate one deterministic :class:`PairSample` without touching disk.

    Passing an integer reproduces that exact sample.  Passing ``rng`` either as
    the first argument or by keyword draws one scene seed from that generator,
    which is the convenient form for an on-the-fly curriculum worker.
    """
    if rng is not None:
        if seed is not None:
            raise ValueError("provide seed or rng, not both")
        seed = rng
    if seed is None:
        raise ValueError("seed or rng is required")
    if isinstance(seed, np.random.Generator):
        scene_seed = int(seed.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    else:
        scene_seed = int(seed)
    config = config or SyntheticConfig()
    if tile_size is not None:
        config = replace(config, tile_size=int(tile_size))
    config = config.validate()
    sample_id = sample_id or f"procedural-{scene_seed:010d}"
    transform_rng = np.random.default_rng(np.random.SeedSequence([scene_seed, 991]))
    if rotation is None:
        rotation = int(transform_rng.integers(0, 4))
    if reflect is None:
        reflect = bool(transform_rng.integers(0, 2))
    rotation = int(rotation) % 4
    reflect = bool(reflect)

    movie, frame_instances, metadata = _render_scene(
        scene_seed,
        scene_kind,
        config,
        sample_id,
        force_branch=force_branch,
        split_transition=split_transition,
        force_merge=force_merge,
        merge_transition=merge_transition,
    )
    image_t, image_tp1 = causal_rgb_pair(
        movie,
        config.center_frame,
        background_blend=config.background_blend,
    )
    image_t = transform_d8_image(image_t, rotation, reflect).astype(np.float32)
    image_tp1 = transform_d8_image(image_tp1, rotation, reflect).astype(np.float32)

    raw_t = frame_instances[config.center_frame]
    raw_tp1 = frame_instances[config.center_frame + 1]
    instances_t = [
        _transform_instance(item, config.tile_size, rotation, reflect) for item in raw_t
    ]
    instances_tp1 = [
        _transform_instance(item, config.tile_size, rotation, reflect)
        for item in raw_tp1
    ]
    tracks_t = {item.track_id for item in instances_t if item.track_id is not None}
    tracks_tp1 = {
        item.track_id for item in instances_tp1 if item.track_id is not None
    }
    links = [(track_id, track_id) for track_id in sorted(tracks_t & tracks_tp1)]

    # At a one-to-two branch or two-to-one merge, association is structurally
    # ambiguous rather than negative.  Detection/head/axis supervision remains
    # exhaustive while the pair's link loss is masked.
    link_supervision_valid = not bool(
        metadata["branch_transition"] or metadata["merge_transition"]
    )
    if not link_supervision_valid:
        links = []
    metadata.update(
        {
            "split": str(split),
            "center_frame": int(config.center_frame),
            "frames_t": [
                config.center_frame - 2,
                config.center_frame - 1,
                config.center_frame,
            ],
            "frames_tp1": [
                config.center_frame - 1,
                config.center_frame,
                config.center_frame + 1,
            ],
            "rotation": rotation,
            "reflect": reflect,
            "link_supervision_valid": link_supervision_valid,
            "link_exhaustive": link_supervision_valid,
            "ambiguous_lineage_event": (
                "branch"
                if metadata["branch_transition"]
                else "merge"
                if metadata["merge_transition"]
                else None
            ),
            "axis_tapered": False,
        }
    )
    return PairSample(
        sample_id=sample_id,
        source="procedural",
        image_t=np.ascontiguousarray(image_t),
        image_tp1=np.ascontiguousarray(image_tp1),
        instances_t=instances_t,
        instances_tp1=instances_tp1,
        links=links,
        exhaustive_t=True,
        exhaustive_tp1=True,
        metadata=metadata,
    ).validate()


def synthetic_records(
    count: int,
    *,
    seed: int = 20260830,
    split: str = "train",
    include_explicit_negatives: bool = True,
) -> list[dict[str, Any]]:
    """Return JSON-serializable recipes, not generated image data.

    Every tenth recipe is empty and the next is a frozen-comet hard negative.
    This makes both negative types explicit in manifests while the remaining
    recipes generate ordinary positive scenes on demand.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    records: list[dict[str, Any]] = []
    for index in range(int(count)):
        recipe_seed = int(np.random.SeedSequence([int(seed), index]).generate_state(1)[0])
        if include_explicit_negatives and index % 10 == 0:
            scene_kind = "empty"
        elif include_explicit_negatives and index % 10 == 1:
            scene_kind = "frozen"
        else:
            scene_kind = "positive"
        augmentation_rng = np.random.default_rng(
            np.random.SeedSequence([recipe_seed, 991])
        )
        rotation = int(augmentation_rng.integers(0, 4))
        reflect = bool(augmentation_rng.integers(0, 2))
        sample_id = f"procedural-{split}-{index:07d}-{recipe_seed:010d}"
        records.append(
            {
                "sample_id": sample_id,
                "source": "procedural",
                "split": str(split),
                "source_movie": None,
                "seed": recipe_seed,
                "scene_kind": scene_kind,
                "rotation": rotation,
                "reflect": reflect,
                "force_branch": False,
                "split_transition": False,
                "force_merge": False,
                "merge_transition": True,
                "augmentation": {
                    "kind": "d8",
                    "rotation_quarter_turns": rotation,
                    "reflect": reflect,
                    "appearance": "none",
                },
            }
        )
    return records


def build_synthetic_pair_sample(
    record: Mapping[str, Any],
    project_root: str | None = None,
    config: SyntheticConfig | None = None,
) -> PairSample:
    """Materialize one manifest recipe; ``project_root`` is intentionally unused."""
    del project_root
    return generate_synthetic_pair(
        int(record["seed"]),
        scene_kind=str(record.get("scene_kind", "positive")),
        rotation=int(record.get("rotation", 0)),
        reflect=bool(record.get("reflect", False)),
        force_branch=bool(record.get("force_branch", False)),
        split_transition=bool(record.get("split_transition", False)),
        force_merge=bool(record.get("force_merge", False)),
        merge_transition=bool(record.get("merge_transition", True)),
        sample_id=str(record.get("sample_id") or f"procedural-{record['seed']}"),
        split=str(record.get("split", "train")),
        config=config,
    )


class SyntheticPairSource(Sequence[PairSample]):
    """Indexable deterministic source whose images are generated only on access."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]] | None = None,
        *,
        count: int = 0,
        seed: int = 20260830,
        split: str = "train",
        config: SyntheticConfig | None = None,
    ) -> None:
        if records is not None and count:
            raise ValueError("provide records or count, not both")
        self.records = list(records) if records is not None else synthetic_records(
            count, seed=seed, split=split
        )
        self.config = config or SyntheticConfig()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> PairSample:
        return build_synthetic_pair_sample(self.records[index], config=self.config)
