from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from comet_sam3.data.unet_masks import (
    DEFAULT_TILE_SIZE,
    augment_unet_record,
    build_unet_pair_sample,
    derive_skeleton_axis,
    discover_unet_pairs,
    load_unet_records,
    split_label_runs,
)


def _write_movie_pair(
    root: Path,
    condition: str,
    stem: str,
    movie: np.ndarray,
    masks: np.ndarray,
) -> tuple[Path, Path]:
    folder = root / "Data" / condition
    folder.mkdir(parents=True, exist_ok=True)
    movie_path = folder / f"{stem}.tif"
    mask_path = folder / f"{stem}_comet_masks.tif"
    tifffile.imwrite(movie_path, movie)
    tifffile.imwrite(mask_path, masks)
    return movie_path, mask_path


def _source_arrays(frames: int = 9, height: int = 64, width: int = 64):
    yy, xx = np.indices((height, width))
    base = (100 + yy * 0.15 + xx * 0.08).astype(np.float32)
    movie = np.repeat(base[None], frames, axis=0)
    masks = np.zeros((frames, height, width), np.uint16)
    for frame in range(frames):
        y = 25 + frame // 4
        x0 = 8 + frame * 2
        masks[frame, y, x0 : x0 + 13] = 1
        movie[frame, y, x0 : x0 + 13] += np.linspace(45, 120, 13)
    return np.clip(movie, 0, 65535).astype(np.uint16), masks


def _write_donor(root: Path, *, fully_certified: bool) -> tuple[Path, Path]:
    folder = root / "Data" / "donors"
    folder.mkdir(parents=True, exist_ok=True)
    yy, xx = np.indices((256, 256))
    static = (80 + 0.03 * yy + 0.02 * xx).astype(np.float32)
    movie = np.stack([static + frame * 0.2 for frame in range(9)]).astype(np.uint16)
    certification = np.ones_like(movie, np.uint8)
    if not fully_certified:
        certification[:] = 0
        certification[:, 82:174, 84:176] = 1
    movie_path = folder / "certified_donor.tif"
    background_path = folder / "certified_donor_background.tif"
    tifffile.imwrite(movie_path, movie)
    tifffile.imwrite(background_path, certification)
    return movie_path, background_path


def test_discovery_is_confined_to_primary_data(tmp_path: Path):
    movie = np.zeros((4, 8, 8), np.uint16)
    masks = np.zeros_like(movie)
    masks[:, 3, 2:6] = 1
    for index in range(14):
        _write_movie_pair(tmp_path, f"condition-{index % 4}", f"movie-{index:02d}", movie, masks)

    duplicate = tmp_path / "EvalData"
    duplicate.mkdir()
    tifffile.imwrite(duplicate / "movie-00.tif", movie)
    tifffile.imwrite(duplicate / "movie-00_comet_masks.tif", masks)

    pairs = discover_unet_pairs(tmp_path)
    assert len(pairs) == 14
    assert all(record["movie_path"].startswith("Data/") for record in pairs)
    assert not any("EvalData" in json.dumps(record) for record in pairs)


def test_real_project_has_the_fourteen_audited_primary_pairs():
    project = Path(__file__).resolve().parents[2]
    if not (project / "Data").is_dir():
        pytest.skip("real Claude-CometTrack Data directory is unavailable")
    pairs = discover_unet_pairs(project)
    assert len(pairs) == 14
    assert {record["condition"] for record in pairs} == {
        "2EB3",
        "EB3-GW16",
        "EB3-N271",
        "EB3WT",
    }


def test_label_ids_are_split_at_every_temporal_gap():
    masks = np.zeros((10, 12, 12), np.uint16)
    for frame in (0, 1, 2, 3, 5, 6, 7, 8):
        masks[frame, 5, 2:9] = 4
    masks[9, 7, 4:10] = 65535

    runs = split_label_runs(masks, min_frames=1)
    assert runs == [
        {"label_id": 4, "run_index": 0, "frames": [0, 1, 2, 3]},
        {"label_id": 4, "run_index": 1, "frames": [5, 6, 7, 8]},
    ]


def test_axis_requires_one_component_and_exactly_two_endpoints():
    clean = np.zeros((20, 20), bool)
    clean[10, 3:17] = True
    axis = derive_skeleton_axis(clean)
    assert axis is not None
    assert len(axis) == 14
    assert {tuple(axis[0]), tuple(axis[-1])} == {(10.0, 3.0), (10.0, 16.0)}

    disconnected = clean.copy()
    disconnected[2, 2:5] = True
    assert derive_skeleton_axis(disconnected) is None

    loop = np.zeros((20, 20), bool)
    loop[5, 5:15] = True
    loop[14, 5:15] = True
    loop[5:15, 5] = True
    loop[5:15, 14] = True
    assert derive_skeleton_axis(loop) is None


def test_records_do_not_bridge_gaps_and_never_claim_heads(tmp_path: Path):
    movie = np.full((10, 32, 32), 100, np.uint16)
    masks = np.zeros_like(movie)
    for frame in (0, 1, 2, 3, 5, 6, 7, 8):
        masks[frame, 12, 5 + frame : 13 + frame] = 2
        movie[frame, 12, 5 + frame : 13 + frame] = 180
    _write_movie_pair(tmp_path, "cell", "gapped", movie, masks)

    records = load_unet_records(tmp_path, tile_size=192)
    assert [record["source_frames"] for record in records] == [
        [0, 1, 2, 3],
        [5, 6, 7, 8],
    ]
    assert records[0]["track_id"] != records[1]["track_id"]
    assert all(record["head_supervision"] is False for record in records)
    assert all(record["axis_supervision"] is True for record in records)
    assert all(record["association_supervision"] == "positive" for record in records)


def test_partial_movie_split_map_never_admits_unassigned_movies(tmp_path: Path):
    movie, masks = _source_arrays()
    movie_path, _mask_path = _write_movie_pair(
        tmp_path, "source", "painted_track", movie, masks
    )
    relative = movie_path.relative_to(tmp_path).as_posix()

    assert load_unet_records(
        tmp_path,
        split="train",
        movie_splits={relative: "validation"},
    ) == []
    assert load_unet_records(
        tmp_path,
        split="validation",
        movie_splits={relative: "validation"},
    )


@pytest.mark.parametrize("rotation,reflect", [(0, False), (1, True), (2, False), (3, True)])
def test_end_to_end_pair_uses_d8_axis_only_and_positive_link(
    tmp_path: Path, rotation: int, reflect: bool
):
    movie, masks = _source_arrays()
    _write_movie_pair(tmp_path, "source", "painted_track", movie, masks)
    _write_donor(tmp_path, fully_certified=True)

    base_record = load_unet_records(tmp_path)[0]
    record = augment_unet_record(
        base_record,
        tmp_path,
        rotation=rotation,
        reflect=reflect,
        seed=13,
    )
    sample = build_unet_pair_sample(record, tmp_path)

    assert sample.image_t.shape == (DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE, 3)
    assert sample.image_tp1.shape == sample.image_t.shape
    assert sample.image_t.dtype == np.float32
    assert np.array_equal(sample.image_t[..., 1], sample.image_tp1[..., 0])
    assert np.array_equal(sample.image_t[..., 2], sample.image_tp1[..., 1])
    assert np.any(sample.image_t > 0)

    assert len(sample.instances_t) == len(sample.instances_tp1) == 1
    instance_t, instance_tp1 = sample.instances_t[0], sample.instances_tp1[0]
    assert instance_t.head_yx is None and not instance_t.head_valid
    assert instance_tp1.head_yx is None and not instance_tp1.head_valid
    assert instance_t.axis_valid and instance_tp1.axis_valid
    assert instance_t.presence_valid and instance_tp1.presence_valid
    assert instance_t.track_id == instance_tp1.track_id == record["track_id"]
    assert sample.links == [(instance_t.instance_id, instance_tp1.instance_id)]
    assert sample.exhaustive_t and sample.exhaustive_tp1

    assert record["unit_intensity"] is True
    assert record["intensity_multiplier"] == 1.0
    assert record["added_noise"] is False
    assert record["added_blur"] is False
    assert record["added_bleaching"] is False


def test_partial_certification_keeps_sample_nonexhaustive(tmp_path: Path):
    movie, masks = _source_arrays()
    _write_movie_pair(tmp_path, "source", "painted_track", movie, masks)
    _write_donor(tmp_path, fully_certified=False)

    base_record = load_unet_records(tmp_path)[0]
    record = augment_unet_record(
        base_record,
        tmp_path,
        rotation=0,
        reflect=False,
        seed=7,
    )
    assert record["paste_rectangle_certified"] is True
    assert record["certified_fraction"] < 1.0
    assert record["exhaustive_t"] is False
    assert record["exhaustive_tp1"] is False
    assert record["nonexhaustive_reason"]

    sample = build_unet_pair_sample(record, tmp_path)
    assert not sample.exhaustive_t
    assert not sample.exhaustive_tp1
    assert sample.instances_t[0].presence_valid
