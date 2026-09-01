"""Current V4 head/axis annotations as portable, partially labelled pairs.

The V4 annotation store uses full-movie coordinates.  This module keeps those
coordinates in the JSONL descriptor and converts them to a translated source
tile only when the sample is materialized.  Unlabelled pixels are always
unknown: generic rejections, uncertain reviews, and drafts never become
negative examples.  Only explicitly drawn background rectangles are negative
supervision.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from comet_sam3.geometry import transform_d8_image, transform_d8_yx
from comet_sam3.preprocessing import causal_rgb_pair, temporal_median_background
from comet_sam3.schema import CometInstance, PairSample


CURRENT_RECORD_VERSION = 1
DEFAULT_SOURCE_TILE_SIZE = 192
DEFAULT_VALIDATION_MOVIES = frozenset(
    {
        "20260710_pAJV103_0.25DOX-ON_010",
        "20260716_N271_0.25DOX_ON_002",
    }
)
ACCEPTED_VERDICTS = frozenset({"both", "head_only", "axis_only"})
IGNORED_VERDICTS = frozenset({"rejected", "uncertain", "draft"})


def _stable_seed(text: str, base_seed: int = 0) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}|{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF


def _relative_path(path: str | Path, project_root: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError as error:
        raise ValueError(f"path is outside project root: {resolved}") from error


def _read_queue(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["candidate_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"duplicate candidate_id in {path}")
    return by_id


def _group_frame_points(points: Iterable[dict[str, Any]]) -> dict[int, list[list[float]]]:
    grouped: dict[int, list[list[float]]] = defaultdict(list)
    for point in points:
        grouped[int(point["frame"])].append([float(point["y"]), float(point["x"])])
    return dict(grouped)


def _split_for_movie(movie: str, validation_movies: frozenset[str]) -> str:
    return "val" if movie in validation_movies else "train"


def _is_pajv101(movie: str) -> bool:
    return "pAJV101" in movie


def _base_record(
    *,
    sample_id: str,
    kind: str,
    movie: str,
    movie_path: str,
    labels_path: str,
    queue_path: str,
    center_frame: int,
    tile_size: int,
    anchor_yx: Sequence[float],
    validation_movies: frozenset[str],
    seed: int,
    background_blend: float,
) -> dict[str, Any]:
    return {
        "record_version": CURRENT_RECORD_VERSION,
        "sample_id": sample_id,
        "source": "current",
        "kind": kind,
        "split": _split_for_movie(movie, validation_movies),
        "source_movie": movie,
        "movie_path": movie_path,
        "center_frame": int(center_frame),
        "source_tile_size": int(tile_size),
        "crop_policy": {
            "name": "translated_square_about_annotation",
            "anchor_yx": [float(anchor_yx[0]), float(anchor_yx[1])],
            "translation_yx": [0, 0],
            "targets_must_remain_inside": True,
        },
        "augmentation": {
            "name": "lossless_d8_and_integer_crop_translation",
            "seed": int(seed),
            "rotation_quadrants_ccw": 0,
            "reflect_horizontal_after_rotation": False,
            "translation_yx": [0, 0],
            "appearance_augmentation": "none",
        },
        "preprocessing": {
            "causal_channels_t": [-2, -1, 0],
            "causal_channels_tp1": [-1, 0, 1],
            "joint_robust_percentiles": [1.0, 99.7],
            "temporal_median_background": True,
            "positive_residual_clip": True,
            "background_blend": float(background_blend),
        },
        "provenance": {
            "labels_path": labels_path,
            "queue_path": queue_path,
            "coordinate_system": "absolute_movie_yx",
        },
    }


def _cap_pajv101_backgrounds(
    records: list[dict[str, Any]], max_fraction: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 <= float(max_fraction) < 1.0:
        raise ValueError("pAJV101 background fraction must be in [0, 1)")
    pajv = [
        record
        for record in records
        if record["kind"] == "background_negative"
        and _is_pajv101(record["source_movie"])
        and record["split"] == "train"
    ]
    other = [
        record
        for record in records
        if record["kind"] == "background_negative"
        and not _is_pajv101(record["source_movie"])
        and record["split"] == "train"
    ]
    allowed = (
        math.floor(float(max_fraction) * len(other) / (1.0 - float(max_fraction)))
        if max_fraction > 0.0
        else 0
    )
    keep_ids = {
        record["sample_id"]
        for record in sorted(pajv, key=lambda item: item["sample_id"])[:allowed]
    }
    kept = [
        record
        for record in records
        if record not in pajv or record["sample_id"] in keep_ids
    ]
    retained = min(len(pajv), allowed)
    denominator = len(other) + retained
    metadata = {
        "pajv101_background_only": True,
        "pajv101_max_background_fraction": float(max_fraction),
        "pajv101_background_candidates": len(pajv),
        "pajv101_background_retained": retained,
        "pajv101_background_dropped": len(pajv) - retained,
        "pajv101_background_fraction_after_cap": (
            retained / denominator if denominator else 0.0
        ),
    }
    for record in kept:
        if _is_pajv101(record["source_movie"]):
            record.setdefault("constraints", {}).update(metadata)
    return kept, metadata


def load_current_records(
    labels: str | Path,
    queue_csv: str | Path,
    project_root: str | Path,
    *,
    source_tile_size: int = DEFAULT_SOURCE_TILE_SIZE,
    validation_movies: Iterable[str] = DEFAULT_VALIDATION_MOVIES,
    base_seed: int = 20260830,
    include_background_rectangles: bool = True,
    pajv101_max_background_fraction: float = 0.15,
    background_blend: float = 0.5,
) -> list[dict[str, Any]]:
    """Create portable descriptors for accepted pairs and certified negatives.

    ``labels`` may name ``labels.json`` or its containing V4 session directory.
    All paths written into records are relative to ``project_root``.  Positive
    pAJV101 records are forbidden; pAJV101 contributes background rectangles
    only, capped deterministically to the requested fraction of the training
    background pool.
    """
    root = Path(project_root).expanduser().resolve()
    labels_path = Path(labels).expanduser().resolve()
    if labels_path.is_dir():
        labels_path = labels_path / "labels.json"
    queue_path = Path(queue_csv).expanduser().resolve()
    if int(source_tile_size) <= 0:
        raise ValueError("source_tile_size must be positive")
    if not 0.0 <= float(background_blend) <= 1.0:
        raise ValueError("background_blend must be in [0, 1]")
    validation = frozenset(map(str, validation_movies))
    if validation != DEFAULT_VALIDATION_MOVIES:
        raise ValueError(
            "current annotations must use exactly the frozen validation movies: "
            f"{sorted(DEFAULT_VALIDATION_MOVIES)}"
        )
    with labels_path.open() as handle:
        store = json.load(handle)
    queue = _read_queue(queue_path)
    labels_rel = _relative_path(labels_path, root)
    queue_rel = _relative_path(queue_path, root)
    records: list[dict[str, Any]] = []

    for candidate_id, review in sorted(store.get("reviews", {}).items()):
        if candidate_id not in queue:
            raise KeyError(f"review {candidate_id} has no queue row")
        row = queue[candidate_id]
        movie = str(row["movie"])
        movie_path = _relative_path(row["movie_path"], root)
        proposal_frame = int(float(row["frame"]))
        if int(review.get("proposal_frame", proposal_frame)) != proposal_frame:
            raise ValueError(f"proposal frame mismatch for {candidate_id}")
        verdict = str(review.get("verdict", "draft"))
        head_valid = bool(review.get("head_accepted", False))
        axis_valid = bool(review.get("axis_accepted", False))

        if verdict in ACCEPTED_VERDICTS:
            expected_head = verdict in {"both", "head_only"}
            expected_axis = verdict in {"both", "axis_only"}
            if (head_valid, axis_valid) != (expected_head, expected_axis):
                raise ValueError(f"acceptance flags disagree with verdict for {candidate_id}")
            if _is_pajv101(movie):
                raise ValueError(f"pAJV101 positive annotation is forbidden: {candidate_id}")
            heads = _group_frame_points(review.get("head_points", []))
            axes = _group_frame_points(review.get("axis_pixels", []))
            for frame, valid, grouped, label_name in (
                (proposal_frame, head_valid, heads, "head"),
                (proposal_frame + 1, head_valid, heads, "head"),
                (proposal_frame, axis_valid, axes, "axis"),
                (proposal_frame + 1, axis_valid, axes, "axis"),
            ):
                if valid and frame not in grouped:
                    raise ValueError(
                        f"accepted {label_name} missing at frame {frame} for {candidate_id}"
                    )
            if head_valid and any(len(heads[frame]) != 1 for frame in (proposal_frame, proposal_frame + 1)):
                raise ValueError(f"accepted pair must have one head per frame: {candidate_id}")
            record = _base_record(
                sample_id=f"current:{candidate_id}",
                kind="accepted_pair",
                movie=movie,
                movie_path=movie_path,
                labels_path=labels_rel,
                queue_path=queue_rel,
                center_frame=proposal_frame,
                tile_size=int(source_tile_size),
                anchor_yx=(float(row["y"]), float(row["x"])),
                validation_movies=validation,
                seed=_stable_seed(f"current:{candidate_id}", base_seed),
                background_blend=float(background_blend),
            )
            record["supervision"] = {
                "t": {
                    "frame": proposal_frame,
                    "head_valid": head_valid,
                    "head_yx": heads.get(proposal_frame, [None])[0] if head_valid else None,
                    "axis_valid": axis_valid,
                    "axis_yx": axes.get(proposal_frame, []) if axis_valid else None,
                    "presence_valid": True,
                    "presence": True,
                },
                "tp1": {
                    "frame": proposal_frame + 1,
                    "head_valid": head_valid,
                    "head_yx": heads.get(proposal_frame + 1, [None])[0] if head_valid else None,
                    "axis_valid": axis_valid,
                    "axis_yx": axes.get(proposal_frame + 1, []) if axis_valid else None,
                    "presence_valid": True,
                    "presence": True,
                },
                "positive_link": True,
                "negative_links_exhaustive": False,
            }
            record["provenance"].update(
                {
                    "candidate_id": candidate_id,
                    "verdict": verdict,
                    "queue_category": row.get("category", ""),
                    "quality_warnings": list(review.get("quality_warnings", [])),
                }
            )
            records.append(record)
        elif verdict not in IGNORED_VERDICTS:
            raise ValueError(f"unknown V4 verdict {verdict!r} for {candidate_id}")

        if include_background_rectangles:
            for region_index, region in enumerate(review.get("background_regions", [])):
                frame = int(region["frame"])
                center_y = (float(region["y0"]) + float(region["y1"])) / 2.0
                center_x = (float(region["x0"]) + float(region["x1"])) / 2.0
                sample_id = f"current-bg:{candidate_id}:{region_index}"
                record = _base_record(
                    sample_id=sample_id,
                    kind="background_negative",
                    movie=movie,
                    movie_path=movie_path,
                    labels_path=labels_rel,
                    queue_path=queue_rel,
                    center_frame=frame,
                    tile_size=int(source_tile_size),
                    anchor_yx=(center_y, center_x),
                    validation_movies=validation,
                    seed=_stable_seed(sample_id, base_seed),
                    background_blend=float(background_blend),
                )
                record["supervision"] = {
                    "t": {
                        "frame": frame,
                        "certified_background_rectangles_y0y1x0x1": [
                            [
                                float(region["y0"]),
                                float(region["y1"]),
                                float(region["x0"]),
                                float(region["x1"]),
                            ]
                        ],
                    },
                    "tp1": {"frame": frame + 1},
                    "positive_link": False,
                    "negative_links_exhaustive": False,
                }
                record["provenance"].update(
                    {
                        "candidate_id": candidate_id,
                        "source_review_verdict": verdict,
                        "background_region_index": region_index,
                        "background_is_explicit": True,
                        "unlabelled_tile_pixels_are_unknown": True,
                    }
                )
                records.append(record)

    records, cap = _cap_pajv101_backgrounds(records, pajv101_max_background_fraction)
    for record in records:
        record.setdefault("dataset_policy", {}).update(cap)
    return sorted(records, key=lambda item: (item["split"], item["sample_id"]))


def augment_current_record(
    record: dict[str, Any],
    *,
    seed: int | None = None,
    rotation: int | None = None,
    reflect: bool | None = None,
    translation_yx: Sequence[int] | None = None,
    max_translation: int | None = None,
) -> dict[str, Any]:
    """Return a descriptor with one deterministic D8/crop translation draw."""
    out = copy.deepcopy(record)
    if seed is None:
        seed = int(out["augmentation"]["seed"])
    rng = np.random.default_rng(int(seed))
    tile_size = int(out["source_tile_size"])
    limit = tile_size // 4 if max_translation is None else int(max_translation)
    if limit < 0:
        raise ValueError("max_translation must be nonnegative")
    if rotation is None:
        rotation = int(rng.integers(0, 4))
    if reflect is None:
        reflect = bool(rng.integers(0, 2))
    if translation_yx is None:
        translation_yx = [
            int(rng.integers(-limit, limit + 1)),
            int(rng.integers(-limit, limit + 1)),
        ]
    if len(translation_yx) != 2:
        raise ValueError("translation_yx must contain dy, dx")
    out["augmentation"].update(
        {
            "seed": int(seed),
            "rotation_quadrants_ccw": int(rotation) % 4,
            "reflect_horizontal_after_rotation": bool(reflect),
            "translation_yx": [int(translation_yx[0]), int(translation_yx[1])],
        }
    )
    out["crop_policy"]["translation_yx"] = list(out["augmentation"]["translation_yx"])
    return out


@lru_cache(maxsize=12)
def _load_movie(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".nd2":
        import nd2

        with nd2.ND2File(path) as handle:
            axes = [str(axis).upper() for axis in handle.sizes]
            array = np.asarray(handle.asarray())
        for name, method in (("C", "first"), ("S", "first"), ("Z", "max")):
            if name in axes:
                index = axes.index(name)
                array = array.max(axis=index) if method == "max" else np.take(array, 0, axis=index)
                axes.pop(index)
        for name in list(axes):
            if name not in {"T", "Y", "X"}:
                index = axes.index(name)
                array = np.take(array, 0, axis=index)
                axes.pop(index)
        wanted = [name for name in ("T", "Y", "X") if name in axes]
        if wanted and wanted != axes:
            array = np.transpose(array, [axes.index(name) for name in wanted])
    else:
        import tifffile

        array = np.asarray(tifffile.imread(path))
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        raise ValueError(f"expected (T,Y,X), got {array.shape} from {path}")
    return np.ascontiguousarray(array)


@lru_cache(maxsize=12)
def _load_background(path: Path) -> np.ndarray:
    """Cache the expensive movie-level median once per worker and movie."""
    return temporal_median_background(_load_movie(path))


def _crop_origin(record: dict[str, Any], movie_shape: Sequence[int]) -> tuple[int, int]:
    tile_size = int(record["source_tile_size"])
    height, width = int(movie_shape[-2]), int(movie_shape[-1])
    if tile_size > height or tile_size > width:
        raise ValueError(f"tile {tile_size} does not fit movie shape {(height, width)}")
    anchor = np.asarray(record["crop_policy"]["anchor_yx"], dtype=np.float64)
    shift = np.asarray(record["augmentation"].get("translation_yx", [0, 0]), dtype=np.float64)
    center = anchor + shift
    y0 = int(np.clip(round(float(center[0])) - tile_size // 2, 0, height - tile_size))
    x0 = int(np.clip(round(float(center[1])) - tile_size // 2, 0, width - tile_size))
    return y0, x0


def _local_target(
    global_yx: Sequence[float] | Sequence[Sequence[float]],
    y0: int,
    x0: int,
    size: int,
    rotation: int,
    reflect: bool,
    *,
    name: str,
) -> np.ndarray:
    points = np.asarray(global_yx, dtype=np.float32).reshape(-1, 2)
    local = points - np.asarray([y0, x0], dtype=np.float32)
    inside = (
        (local[:, 0] >= 0)
        & (local[:, 0] < size)
        & (local[:, 1] >= 0)
        & (local[:, 1] < size)
    )
    if not bool(np.all(inside)):
        raise ValueError(f"translated crop loses accepted {name} target")
    return np.asarray(transform_d8_yx(local, size, rotation, reflect), np.float32)


def _local_rectangle(
    region: Sequence[float],
    y0: int,
    x0: int,
    size: int,
    rotation: int,
    reflect: bool,
) -> list[float]:
    gy0, gy1, gx0, gx1 = map(float, region)
    corners = np.asarray(
        [[gy0, gx0], [gy0, gx1], [gy1, gx1], [gy1, gx0]], np.float32
    )
    local = corners - np.asarray([y0, x0], np.float32)
    local[:, 0] = np.clip(local[:, 0], 0, size - 1)
    local[:, 1] = np.clip(local[:, 1], 0, size - 1)
    transformed = transform_d8_yx(local, size, rotation, reflect)
    return [
        float(transformed[:, 0].min()),
        float(transformed[:, 0].max()),
        float(transformed[:, 1].min()),
        float(transformed[:, 1].max()),
    ]


def build_current_pair_sample(
    record: dict[str, Any], project_root: str | Path
) -> PairSample:
    """Materialize one current-annotation descriptor into a ``PairSample``."""
    if record.get("source") != "current":
        raise ValueError("record is not a current-annotation sample")
    root = Path(project_root).expanduser().resolve()
    movie_path = (root / record["movie_path"]).resolve()
    movie = _load_movie(movie_path)
    center = int(record["center_frame"])
    if center < 2 or center + 1 >= len(movie):
        raise IndexError(f"sample {record['sample_id']} lacks t-2 through t+1")
    size = int(record["source_tile_size"])
    y0, x0 = _crop_origin(record, movie.shape)
    movie_tile = movie[:, y0 : y0 + size, x0 : x0 + size]
    background = _load_background(movie_path)[y0 : y0 + size, x0 : x0 + size]
    image_t, image_tp1 = causal_rgb_pair(
        movie_tile,
        center,
        background=background,
        background_blend=float(record["preprocessing"]["background_blend"]),
    )
    rotation = int(record["augmentation"].get("rotation_quadrants_ccw", 0)) % 4
    reflect = bool(record["augmentation"].get("reflect_horizontal_after_rotation", False))
    image_t = transform_d8_image(image_t, rotation, reflect).astype(np.float32)
    image_tp1 = transform_d8_image(image_tp1, rotation, reflect).astype(np.float32)
    instances_t: list[CometInstance] = []
    instances_tp1: list[CometInstance] = []
    links: list[tuple[str, str]] = []
    metadata: dict[str, Any] = {
        "split": record["split"],
        "source_movie": record["source_movie"],
        "center_frame": center,
        "crop_y0x0": [y0, x0],
        "source_tile_size": size,
        "augmentation": copy.deepcopy(record["augmentation"]),
        "provenance": copy.deepcopy(record["provenance"]),
        "certified_background_regions_t": [],
        "certified_background_regions_tp1": [],
    }

    if record["kind"] == "accepted_pair":
        track_id = f"current-track:{record['provenance']['candidate_id']}"
        built: list[CometInstance] = []
        for endpoint in ("t", "tp1"):
            target = record["supervision"][endpoint]
            head_yx = None
            if target["head_valid"]:
                transformed = _local_target(
                    target["head_yx"], y0, x0, size, rotation, reflect, name="head"
                )
                head_yx = tuple(map(float, transformed[0]))
            axis_yx = None
            if target["axis_valid"]:
                axis_yx = _local_target(
                    target["axis_yx"], y0, x0, size, rotation, reflect, name="axis"
                )
            instance = CometInstance(
                instance_id=f"{record['sample_id']}:{endpoint}",
                track_id=track_id,
                head_yx=head_yx,
                axis_yx=axis_yx,
                head_valid=bool(target["head_valid"]),
                axis_valid=bool(target["axis_valid"]),
                presence_valid=bool(target["presence_valid"]),
                metadata={"source_frame": int(target["frame"]), "present": True},
            )
            built.append(instance)
        instances_t, instances_tp1 = [built[0]], [built[1]]
        if record["supervision"].get("positive_link", False):
            links = [(built[0].instance_id, built[1].instance_id)]
    elif record["kind"] == "background_negative":
        for region in record["supervision"]["t"].get(
            "certified_background_rectangles_y0y1x0x1", []
        ):
            metadata["certified_background_regions_t"].append(
                _local_rectangle(region, y0, x0, size, rotation, reflect)
            )
        metadata["background_is_local_only"] = True
        metadata["unlabelled_tile_pixels_are_unknown"] = True
    else:
        raise ValueError(f"unsupported current record kind: {record['kind']}")

    return PairSample(
        sample_id=record["sample_id"],
        source="current",
        image_t=image_t,
        image_tp1=image_tp1,
        instances_t=instances_t,
        instances_tp1=instances_tp1,
        links=links,
        exhaustive_t=False,
        exhaustive_tp1=False,
        metadata=metadata,
    ).validate()
