"""Axis-only samples made from the hand-painted masks used by the old U-Net.

The source masks are incomplete instance annotations, not semantic-background
maps.  A nonzero label is one persistent comet identity; zero means *unknown*.
This module consequently provides positive presence, an axis at ``t`` and
``t+1``, and an explicit positive association.  It never invents head labels
and never turns an unpainted pixel into a negative.

Real tracks are copied as positive temporal-median residuals and added at unit
intensity to raw, certified real-background clips.  The only augmentation is a
lossless D8 spatial symmetry.  There is deliberately no intensity multiplier,
added noise, blur, or bleaching.
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import binary_dilation, convolve, gaussian_filter, label as ndi_label
from skimage.morphology import skeletonize

from ..geometry import transform_d8_image
from ..preprocessing import causal_rgb_pair, temporal_median_background
from ..schema import CometInstance, PairSample


MASK_SUFFIX = "_comet_masks.tif"
BACKGROUND_SUFFIX = "_background.tif"
UNCERTAIN_LABEL = 65535
DEFAULT_TILE_SIZE = 192
DEFAULT_MARGIN = 10
D8_TRANSFORMS: tuple[tuple[int, bool], ...] = tuple(
    (rotation, reflect) for rotation in range(4) for reflect in (False, True)
)


def _project_root(path: str | Path) -> Path:
    """Return the directory that owns ``Data/``.

    Callers commonly start either at the Claude-CometTrack root, its ``Data``
    directory, or ``SAM3Training``.  Supporting all three keeps manifest paths
    project-relative and therefore portable to the cluster.
    """

    root = Path(path).expanduser().resolve()
    if root.name == "Data":
        return root.parent
    if (root / "Data").is_dir():
        return root
    if root.name == "SAM3Training" and (root.parent / "Data").is_dir():
        return root.parent
    raise FileNotFoundError(f"could not find Data/ from {root}")


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _movie_for_sidecar(sidecar: Path, suffix: str) -> Path | None:
    stem = sidecar.name[: -len(suffix)]
    for extension in (".nd2", ".tif", ".tiff"):
        candidate = sidecar.with_name(stem + extension)
        if candidate.is_file():
            return candidate
    return None


def _split_for_movie(
    relative_movie: str,
    movie_splits: Mapping[str, str] | None,
    default: str,
) -> str:
    if not movie_splits:
        return default
    path = Path(relative_movie)
    for key in (relative_movie, path.as_posix(), path.name, path.stem):
        if key in movie_splits:
            return str(movie_splits[key])
    # A partial split map must not make an unassigned movie silently appear in
    # whichever split happens to be requested.  Requiring an explicit entry is
    # the conservative movie-level leakage guard.
    return "unassigned"


def discover_unet_pairs(project_root: str | Path) -> list[dict]:
    """Discover primary ``Data`` movie/mask pairs, never ``EvalData`` copies."""

    root = _project_root(project_root)
    data = root / "Data"
    records: list[dict] = []
    # The one-level cell-line layout is intentional.  A recursive search from
    # the project root previously double-counted byte-identical EvalData masks.
    for mask_path in sorted(data.glob(f"*/*{MASK_SUFFIX}")):
        movie_path = _movie_for_sidecar(mask_path, MASK_SUFFIX)
        if movie_path is None:
            continue
        records.append(
            {
                "movie_path": _relative(movie_path, root),
                "mask_path": _relative(mask_path, root),
                "source_movie": _relative(movie_path, root),
                "movie_stem": movie_path.stem,
                "condition": movie_path.parent.name,
            }
        )
    return records


def discover_certified_backgrounds(
    project_root: str | Path,
    *,
    split: str = "train",
    movie_splits: Mapping[str, str] | None = None,
) -> list[dict]:
    """Return real movies with nonempty per-frame certified-background masks."""

    root = _project_root(project_root)
    donors: list[dict] = []
    for background_path in sorted((root / "Data").glob(f"*/*{BACKGROUND_SUFFIX}")):
        movie_path = _movie_for_sidecar(background_path, BACKGROUND_SUFFIX)
        if movie_path is None:
            continue
        relative_movie = _relative(movie_path, root)
        donor_split = _split_for_movie(relative_movie, movie_splits, split)
        if donor_split != split:
            continue
        background = _read_tiff(str(background_path))
        if background.ndim != 3 or not np.any(background > 0):
            continue
        donors.append(
            {
                "movie_path": relative_movie,
                "background_path": _relative(background_path, root),
                "source_movie": relative_movie,
                "movie_stem": movie_path.stem,
                "condition": movie_path.parent.name,
                "split": donor_split,
            }
        )
    return donors


def split_label_runs(
    masks: np.ndarray,
    *,
    min_frames: int = 1,
    uncertain_label: int = UNCERTAIN_LABEL,
) -> list[dict]:
    """Split persistent label IDs wherever their painted frames are not adjacent."""

    labels = np.asarray(masks)
    if labels.ndim != 3:
        raise ValueError(f"masks must have shape (T,Y,X), got {labels.shape}")
    runs: list[dict] = []
    valid_labels = np.unique(labels)
    valid_labels = valid_labels[(valid_labels != 0) & (valid_labels != uncertain_label)]
    for label_value in valid_labels:
        frames = np.flatnonzero(np.any(labels == label_value, axis=(1, 2)))
        pieces = np.split(frames, np.flatnonzero(np.diff(frames) != 1) + 1)
        for run_index, piece in enumerate(pieces):
            if len(piece) < int(min_frames):
                continue
            runs.append(
                {
                    "label_id": int(label_value),
                    "run_index": int(run_index),
                    "frames": [int(frame) for frame in piece],
                }
            )
    return runs


def _neighbor_count(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0
    return convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)


def derive_skeleton_axis(mask: np.ndarray) -> np.ndarray | None:
    """Return an ordered one-pixel axis, or ``None`` for an unsafe mask.

    The audited corpus is 99.2% one connected component with exactly two
    skeleton endpoints.  The remaining cases are rejected rather than repaired
    silently.  An 8-neighbour shortest path orders the retained centerline.
    """

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        return None
    _components, count = ndi_label(binary, structure=np.ones((3, 3), np.uint8))
    if int(count) != 1:
        return None
    skeleton = skeletonize(binary)
    neighbors = _neighbor_count(skeleton)
    endpoints = np.argwhere(skeleton & (neighbors == 1))
    if len(endpoints) != 2:
        return None

    start = tuple(int(value) for value in endpoints[0])
    goal = tuple(int(value) for value in endpoints[1])
    pixels = {tuple(int(value) for value in point) for point in np.argwhere(skeleton)}
    queue: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue and goal not in parent:
        y, x = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nxt = (y + dy, x + dx)
                if nxt in pixels and nxt not in parent:
                    parent[nxt] = (y, x)
                    queue.append(nxt)
    if goal not in parent:
        return None
    ordered = []
    point: tuple[int, int] | None = goal
    while point is not None:
        ordered.append(point)
        point = parent[point]
    ordered.reverse()
    return np.asarray(ordered, dtype=np.float32)


def _source_bbox(
    masks: np.ndarray,
    label_id: int,
    frames: Sequence[int],
    margin: int,
) -> tuple[int, int, int, int]:
    selected = np.any(np.asarray(masks)[list(frames)] == int(label_id), axis=0)
    yy, xx = np.nonzero(selected)
    if not len(yy):
        raise ValueError(f"label {label_id} is absent from requested frames")
    height, width = selected.shape
    return (
        max(0, int(yy.min()) - margin),
        min(height, int(yy.max()) + 1 + margin),
        max(0, int(xx.min()) - margin),
        min(width, int(xx.max()) + 1 + margin),
    )


def load_unet_records(
    project_root: str | Path,
    *,
    split: str = "train",
    movie_splits: Mapping[str, str] | None = None,
    tile_size: int = DEFAULT_TILE_SIZE,
    margin: int = DEFAULT_MARGIN,
) -> list[dict]:
    """Create immutable source records for every safe causal axis pair.

    These records intentionally do not pick a donor or D8 transform.  Use
    :func:`augment_unet_record` to make a fully materializable manifest record.
    """

    root = _project_root(project_root)
    output: list[dict] = []
    for pair in discover_unet_pairs(root):
        pair_split = _split_for_movie(pair["movie_path"], movie_splits, split)
        if pair_split != split:
            continue
        masks = _read_tiff(str(root / pair["mask_path"]))
        for run in split_label_runs(masks, min_frames=4):
            run_frames = run["frames"]
            # i>=2 supplies t-2 and t-1; i+1 supplies the linked t+1 target.
            for index in range(2, len(run_frames) - 1):
                center = int(run_frames[index])
                frames = [center - 2, center - 1, center, center + 1]
                axis_t = derive_skeleton_axis(masks[center] == run["label_id"])
                axis_tp1 = derive_skeleton_axis(masks[center + 1] == run["label_id"])
                if axis_t is None or axis_tp1 is None:
                    continue
                bbox = _source_bbox(masks, run["label_id"], frames, margin)
                if max(bbox[1] - bbox[0], bbox[3] - bbox[2]) > int(tile_size):
                    continue
                track_id = (
                    f"{pair['movie_stem']}:label-{run['label_id']}:run-{run['run_index']}"
                )
                sample_id = f"unet:{track_id}:t-{center}"
                output.append(
                    {
                        "sample_id": sample_id,
                        "source": "unet_masks",
                        "split": pair_split,
                        "source_movie": pair["movie_path"],
                        "source_mask": pair["mask_path"],
                        "condition": pair["condition"],
                        "source_label": int(run["label_id"]),
                        "source_run_index": int(run["run_index"]),
                        "source_run_start": int(run_frames[0]),
                        "source_run_end": int(run_frames[-1]),
                        "source_run_length": len(run_frames),
                        "source_frames": frames,
                        "target_frames": [center, center + 1],
                        "track_id": track_id,
                        "source_bbox_yxyx": list(bbox),
                        "tile_size": int(tile_size),
                        "margin": int(margin),
                        "head_supervision": False,
                        "axis_supervision": True,
                        "presence_supervision": True,
                        "association_supervision": "positive",
                        "exhaustive_t": False,
                        "exhaustive_tp1": False,
                        "nonexhaustive_reason": "old masks are nonexhaustive",
                        "augmented": False,
                    }
                )
    return output


def _stable_rng(record: Mapping, seed: int, rotation: int, reflect: bool) -> np.random.Generator:
    text = f"{record['sample_id']}|{seed}|{rotation % 4}|{int(reflect)}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _rectangle_sums(binary: np.ndarray, height: int, width: int) -> np.ndarray:
    integral = np.pad(
        np.asarray(binary, dtype=np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0))
    )
    return (
        integral[height:, width:]
        - integral[:-height, width:]
        - integral[height:, :-width]
        + integral[:-height, :-width]
    )


def _find_certified_placement(
    background_mask: np.ndarray,
    *,
    box_shape: tuple[int, int],
    tile_size: int,
    rng: np.random.Generator,
) -> dict | None:
    """Place the complete source cutout in pixels certified empty for all 4 frames."""

    certified = np.asarray(background_mask) > 0
    if certified.ndim != 3 or len(certified) < 4:
        return None
    _, full_h, full_w = certified.shape
    box_h, box_w = map(int, box_shape)
    if box_h > tile_size or box_w > tile_size or tile_size > min(full_h, full_w):
        return None
    for donor_start in rng.permutation(len(certified) - 3):
        span = np.all(certified[donor_start : donor_start + 4], axis=0)
        sums = _rectangle_sums(span, box_h, box_w)
        candidates = np.argwhere(sums == box_h * box_w)
        if not len(candidates):
            continue
        global_y, global_x = map(int, candidates[int(rng.integers(len(candidates)))])
        crop_y_min = max(0, global_y + box_h - tile_size)
        crop_y_max = min(global_y, full_h - tile_size)
        crop_x_min = max(0, global_x + box_w - tile_size)
        crop_x_max = min(global_x, full_w - tile_size)
        if crop_y_min > crop_y_max or crop_x_min > crop_x_max:
            continue
        crop_y = int(rng.integers(crop_y_min, crop_y_max + 1))
        crop_x = int(rng.integers(crop_x_min, crop_x_max + 1))
        tile_certified = span[
            crop_y : crop_y + tile_size, crop_x : crop_x + tile_size
        ]
        return {
            "donor_start": int(donor_start),
            "donor_frames": [int(donor_start + offset) for offset in range(4)],
            "donor_crop_yx": [crop_y, crop_x],
            "paste_yx": [global_y - crop_y, global_x - crop_x],
            "certified_fraction": float(tile_certified.mean()),
            "tile_fully_certified": bool(tile_certified.all()),
            "paste_rectangle_certified": True,
        }
    return None


def augment_unet_record(
    record: Mapping,
    project_root: str | Path,
    *,
    rotation: int,
    reflect: bool,
    seed: int = 0,
    movie_splits: Mapping[str, str] | None = None,
    allow_same_movie_donor: bool = False,
) -> dict:
    """Attach a certified donor, translation, and one D8 transform to a record."""

    root = _project_root(project_root)
    if record.get("source") != "unet_masks":
        raise ValueError("record is not a unet_masks source record")
    rng = _stable_rng(record, seed, rotation, reflect)
    y0, y1, x0, x1 = map(int, record["source_bbox_yxyx"])
    height, width = y1 - y0, x1 - x0
    if int(rotation) % 2:
        height, width = width, height

    donors = discover_certified_backgrounds(
        root,
        split=str(record.get("split", "train")),
        movie_splits=movie_splits,
    )
    if not allow_same_movie_donor:
        donors = [donor for donor in donors if donor["movie_path"] != record["source_movie"]]
    if not donors:
        raise ValueError("no nonempty certified real-background donors are available")

    for donor_index in rng.permutation(len(donors)):
        donor = donors[int(donor_index)]
        background_mask = _read_tiff(str(root / donor["background_path"]))
        placement = _find_certified_placement(
            background_mask,
            box_shape=(height, width),
            tile_size=int(record.get("tile_size", DEFAULT_TILE_SIZE)),
            rng=rng,
        )
        if placement is None:
            continue
        augmented = dict(record)
        augmented.update(
            {
                "sample_id": (
                    f"{record['sample_id']}:d8-{int(rotation) % 4}{int(bool(reflect))}"
                    f":seed-{int(seed)}"
                ),
                "donor_movie": donor["movie_path"],
                "donor_background": donor["background_path"],
                "rotation": int(rotation) % 4,
                "reflect": bool(reflect),
                "augmentation_seed": int(seed),
                "transformed_box_shape": [height, width],
                "augmented": True,
                "unit_intensity": True,
                "added_noise": False,
                "added_blur": False,
                "added_bleaching": False,
                "intensity_multiplier": 1.0,
                **placement,
            }
        )
        exhaustive = bool(placement["tile_fully_certified"])
        augmented["exhaustive_t"] = exhaustive
        augmented["exhaustive_tp1"] = exhaustive
        augmented["nonexhaustive_reason"] = (
            None
            if exhaustive
            else "pasted footprint is certified but pixels elsewhere in the tile are unknown"
        )
        return augmented
    raise ValueError("no certified placement can contain this transformed track cutout")


def build_unet_manifest(
    project_root: str | Path,
    *,
    split: str = "train",
    movie_splits: Mapping[str, str] | None = None,
    transforms: Iterable[tuple[int, bool]] = D8_TRANSFORMS,
    seed: int = 0,
    tile_size: int = DEFAULT_TILE_SIZE,
    margin: int = DEFAULT_MARGIN,
    max_source_records: int | None = None,
) -> list[dict]:
    """Materialize JSON-safe records for all requested D8 variants."""

    sources = load_unet_records(
        project_root,
        split=split,
        movie_splits=movie_splits,
        tile_size=tile_size,
        margin=margin,
    )
    if max_source_records is not None:
        sources = sources[: max(0, int(max_source_records))]
    output = []
    for source_index, source in enumerate(sources):
        for transform_index, (rotation, reflect) in enumerate(transforms):
            output.append(
                augment_unet_record(
                    source,
                    project_root,
                    rotation=rotation,
                    reflect=reflect,
                    seed=int(seed) + source_index * 101 + transform_index,
                    movie_splits=movie_splits,
                )
            )
    return output


def _soft_alpha(mask: np.ndarray, grow: int = 3) -> np.ndarray:
    """Soft cutout edge used by the old copy/paste pipeline.

    This smooths only the extraction window, not the image; it is not an image
    blur augmentation.
    """

    binary = np.asarray(mask, dtype=bool)
    dilated = binary_dilation(binary, iterations=max(int(grow), 1))
    alpha = gaussian_filter(dilated.astype(np.float32), max(float(grow) / 2.5, 0.5))
    peak = float(alpha.max(initial=0.0))
    if peak <= 0:
        return binary.astype(np.float32)
    return np.clip(alpha / (0.7 * peak), 0.0, 1.0).astype(np.float32)


@lru_cache(maxsize=12)
def _read_tiff(path: str) -> np.ndarray:
    import tifffile

    return np.asarray(tifffile.imread(path))


def _reduce_nd2(array: np.ndarray, axes: Sequence[str]) -> np.ndarray:
    output = np.asarray(array)
    labels = list(axes)
    for name, operation in (("C", "first"), ("Z", "max")):
        if name in labels:
            index = labels.index(name)
            output = (
                output.max(axis=index)
                if operation == "max"
                else np.take(output, 0, axis=index)
            )
            labels.pop(index)
    for name in list(labels):
        if name not in ("T", "Y", "X"):
            index = labels.index(name)
            output = np.take(output, 0, axis=index)
            labels.pop(index)
    desired = [name for name in ("T", "Y", "X") if name in labels]
    if desired and desired != labels:
        output = np.transpose(output, [labels.index(name) for name in desired])
    return output


@lru_cache(maxsize=4)
def _load_movie(path: str) -> np.ndarray:
    movie_path = Path(path)
    if movie_path.suffix.lower() == ".nd2":
        import nd2

        with nd2.ND2File(movie_path) as handle:
            movie = _reduce_nd2(np.asarray(handle.asarray()), list(handle.sizes.keys()))
    else:
        movie = _read_tiff(str(movie_path))
    movie = np.asarray(movie)
    if movie.ndim == 2:
        movie = movie[None]
    if movie.ndim != 3:
        raise ValueError(f"movie must reduce to (T,Y,X), got {movie.shape} for {movie_path}")
    return np.ascontiguousarray(movie)


@lru_cache(maxsize=8)
def _load_background(path: str) -> np.ndarray:
    """Cache the costly temporal median once per worker and source movie."""
    return temporal_median_background(_load_movie(path))


def build_unet_pair_sample(
    record: Mapping,
    project_root: str | Path,
    *,
    background_blend: float = 0.5,
) -> PairSample:
    """Materialize one augmented old-mask record as a validated ``PairSample``."""

    if not record.get("augmented"):
        raise ValueError("record needs augment_unet_record() before materialization")
    root = _project_root(project_root)
    source_movie = _load_movie(str(root / str(record["source_movie"])))
    source_masks = _read_tiff(str(root / str(record["source_mask"])))
    donor_movie = _load_movie(str(root / str(record["donor_movie"])))
    donor_certification = _read_tiff(str(root / str(record["donor_background"]))) > 0

    source_frames = [int(frame) for frame in record["source_frames"]]
    if source_frames != list(range(source_frames[0], source_frames[0] + 4)):
        raise ValueError("source_frames must be four consecutive frames t-2 through t+1")
    donor_frames = [int(frame) for frame in record["donor_frames"]]
    if donor_frames != list(range(donor_frames[0], donor_frames[0] + 4)):
        raise ValueError("donor_frames must be four consecutive frames")
    y0, y1, x0, x1 = map(int, record["source_bbox_yxyx"])
    label_id = int(record["source_label"])
    source_background = _load_background(str((root / str(record["source_movie"])).resolve()))
    signals = []
    local_masks = []
    for frame in source_frames:
        mask = source_masks[frame, y0:y1, x0:x1] == label_id
        if not mask.any():
            raise ValueError(f"source label {label_id} absent at frame {frame}")
        residual = np.maximum(
            source_movie[frame, y0:y1, x0:x1].astype(np.float32)
            - source_background[y0:y1, x0:x1],
            0.0,
        )
        signals.append(residual * _soft_alpha(mask))
        local_masks.append(mask)
    signal_stack = np.stack(signals, axis=0)[..., None]
    mask_stack = np.stack(local_masks, axis=0)[..., None]
    transformed_signal = transform_d8_image(
        signal_stack, int(record["rotation"]), bool(record["reflect"])
    )[..., 0]
    transformed_masks = transform_d8_image(
        mask_stack, int(record["rotation"]), bool(record["reflect"])
    )[..., 0].astype(bool)

    tile_size = int(record.get("tile_size", DEFAULT_TILE_SIZE))
    donor_y, donor_x = map(int, record["donor_crop_yx"])
    paste_y, paste_x = map(int, record["paste_yx"])
    donor_clip = donor_movie[
        donor_frames[0] : donor_frames[-1] + 1,
        donor_y : donor_y + tile_size,
        donor_x : donor_x + tile_size,
    ].astype(np.float32, copy=True)
    if donor_clip.shape != (4, tile_size, tile_size):
        raise ValueError(f"donor clip has wrong shape {donor_clip.shape}")
    box_h, box_w = transformed_signal.shape[1:]
    donor_clip[:, paste_y : paste_y + box_h, paste_x : paste_x + box_w] += transformed_signal

    # Recheck the certified-footprint claim at materialization time.  This
    # catches stale or manually edited manifests rather than silently changing
    # their negative-label semantics.
    certified_span = np.all(
        donor_certification[donor_frames[0] : donor_frames[-1] + 1], axis=0
    )
    footprint = certified_span[
        donor_y + paste_y : donor_y + paste_y + box_h,
        donor_x + paste_x : donor_x + paste_x + box_w,
    ]
    if footprint.shape != (box_h, box_w) or not footprint.all():
        raise ValueError("pasted track footprint is not certified background")

    donor_background = _load_background(str((root / str(record["donor_movie"])).resolve()))[
        donor_y : donor_y + tile_size, donor_x : donor_x + tile_size
    ]
    image_t, image_tp1 = causal_rgb_pair(
        donor_clip,
        center=2,
        background=donor_background,
        background_blend=float(background_blend),
    )

    axis_t = derive_skeleton_axis(transformed_masks[2])
    axis_tp1 = derive_skeleton_axis(transformed_masks[3])
    if axis_t is None or axis_tp1 is None:
        raise ValueError("D8-transformed target unexpectedly failed axis validation")
    translation = np.array([paste_y, paste_x], dtype=np.float32)
    axis_t = axis_t + translation
    axis_tp1 = axis_tp1 + translation
    sample_id = str(record["sample_id"])
    track_id = str(record["track_id"])
    instance_t_id = f"{sample_id}:instance-t"
    instance_tp1_id = f"{sample_id}:instance-tp1"
    common_metadata = {
        "source_movie": str(record["source_movie"]),
        "source_mask": str(record["source_mask"]),
        "source_label": label_id,
        "head_supervision": "unknown",
        "presence_label": 1,
    }
    instance_t = CometInstance(
        instance_id=instance_t_id,
        track_id=track_id,
        head_yx=None,
        axis_yx=axis_t,
        head_valid=False,
        axis_valid=True,
        presence_valid=True,
        metadata={**common_metadata, "source_frame": source_frames[2]},
    )
    instance_tp1 = CometInstance(
        instance_id=instance_tp1_id,
        track_id=track_id,
        head_yx=None,
        axis_yx=axis_tp1,
        head_valid=False,
        axis_valid=True,
        presence_valid=True,
        metadata={**common_metadata, "source_frame": source_frames[3]},
    )
    metadata = dict(record)
    metadata.update(
        {
            "background_blend": float(background_blend),
            "input_contract": "causal-R=t-2,G=t-1,B=t",
            "partial_supervision": "axis+positive-presence+positive-link; head unknown",
        }
    )
    return PairSample(
        sample_id=sample_id,
        source="unet_masks",
        image_t=image_t,
        image_tp1=image_tp1,
        instances_t=[instance_t],
        instances_tp1=[instance_tp1],
        links=[(instance_t_id, instance_tp1_id)],
        exhaustive_t=bool(record.get("exhaustive_t", False)),
        exhaustive_tp1=bool(record.get("exhaustive_tp1", False)),
        metadata=metadata,
    ).validate()


__all__ = [
    "BACKGROUND_SUFFIX",
    "D8_TRANSFORMS",
    "DEFAULT_MARGIN",
    "DEFAULT_TILE_SIZE",
    "MASK_SUFFIX",
    "UNCERTAIN_LABEL",
    "augment_unet_record",
    "build_unet_manifest",
    "build_unet_pair_sample",
    "derive_skeleton_axis",
    "discover_certified_backgrounds",
    "discover_unet_pairs",
    "load_unet_records",
    "split_label_runs",
]
