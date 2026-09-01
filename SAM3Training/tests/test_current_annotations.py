from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from comet_sam3.data.current_annotations import (
    DEFAULT_VALIDATION_MOVIES,
    augment_current_record,
    build_current_pair_sample,
    load_current_records,
)
from comet_sam3.geometry import transform_d8_yx
from scripts.build_manifest import emit_manifests


VAL_A = "20260710_pAJV103_0.25DOX-ON_010"
VAL_B = "20260716_N271_0.25DOX_ON_002"
TRAIN = "20260710_EB3WT_0.25DOX-ON_001"
PAJV = "20260715_pAJV101_0.25DOX_ON_017"


def _movie(path: Path, offset: int = 0) -> None:
    time = np.arange(10, dtype=np.uint16)[:, None, None] * 50
    yy, xx = np.mgrid[:16, :16]
    array = time + yy[None] * 3 + xx[None] + offset
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, array.astype(np.uint16))


def _review(
    movie: str,
    frame: int,
    verdict: str,
    *,
    heads: bool = False,
    axes: bool = False,
    backgrounds: list[dict] | None = None,
) -> dict:
    return {
        "movie": movie,
        "proposal_frame": frame,
        "verdict": verdict,
        "head_accepted": heads,
        "axis_accepted": axes,
        "head_points": (
            [
                {"frame": frame, "y": 8.0, "x": 8.0},
                {"frame": frame + 1, "y": 7.0, "x": 9.0},
            ]
            if heads
            else []
        ),
        "axis_pixels": (
            [
                *({"frame": frame, "y": y, "x": x} for y, x in [(8, 8), (9, 7), (10, 6)]),
                *(
                    {"frame": frame + 1, "y": y, "x": x}
                    for y, x in [(7, 9), (8, 8), (9, 7)]
                ),
            ]
            if axes
            else []
        ),
        "background_regions": backgrounds or [],
        "quality_warnings": [],
    }


def _fixture(tmp_path: Path, *, many_backgrounds: bool = False) -> tuple[Path, Path, Path]:
    root = tmp_path / "project"
    movie_paths = {}
    for index, movie in enumerate((TRAIN, VAL_A, VAL_B, PAJV)):
        path = root / "Data" / f"{movie}.tif"
        _movie(path, offset=index * 10)
        movie_paths[movie] = path

    queue_path = root / "HeadLabeling/session_001/queue.csv"
    queue_path.parent.mkdir(parents=True)
    rows = [
        ("both", TRAIN),
        ("head", VAL_A),
        ("axis", VAL_B),
        ("reject", TRAIN),
        ("uncertain", TRAIN),
        ("draft", TRAIN),
        ("pajv", PAJV),
    ]
    with queue_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate_id", "movie", "movie_path", "frame", "y", "x", "category"],
        )
        writer.writeheader()
        for candidate_id, movie in rows:
            writer.writerow(
                {
                    "candidate_id": candidate_id,
                    "movie": movie,
                    "movie_path": movie_paths[movie],
                    "frame": 4,
                    "y": 8,
                    "x": 8,
                    "category": "temporal_rgb",
                }
            )

    non_pajv_backgrounds = [
        {"frame": 4, "y0": 2.0, "y1": 5.0, "x0": 2.0, "x1": 6.0}
        for _ in range(17 if many_backgrounds else 1)
    ]
    pajv_backgrounds = [
        {"frame": 4, "y0": 3.0, "y1": 6.0, "x0": 3.0, "x1": 7.0}
        for _ in range(10 if many_backgrounds else 1)
    ]
    labels = {
        "schema_version": 1,
        "reviews": {
            "both": _review(TRAIN, 4, "both", heads=True, axes=True),
            "head": _review(VAL_A, 4, "head_only", heads=True),
            "axis": _review(VAL_B, 4, "axis_only", axes=True),
            "reject": _review(
                TRAIN, 4, "rejected", backgrounds=non_pajv_backgrounds
            ),
            "uncertain": _review(TRAIN, 4, "uncertain"),
            "draft": _review(TRAIN, 4, "draft"),
            "pajv": _review(PAJV, 4, "rejected", backgrounds=pajv_backgrounds),
        },
    }
    labels_path = root / "TrajectoryAxisLabeling/v4_test_session/labels.json"
    labels_path.parent.mkdir(parents=True)
    labels_path.write_text(json.dumps(labels))
    return root, labels_path, queue_path


def test_records_preserve_partial_labels_and_ignore_generic_rejects(tmp_path: Path):
    root, labels, queue = _fixture(tmp_path)
    records = load_current_records(labels, queue, root, source_tile_size=8)
    accepted = {r["provenance"]["candidate_id"]: r for r in records if r["kind"] == "accepted_pair"}
    assert set(accepted) == {"both", "head", "axis"}
    assert accepted["both"]["supervision"]["t"]["head_valid"] is True
    assert accepted["both"]["supervision"]["t"]["axis_valid"] is True
    assert accepted["head"]["supervision"]["t"]["head_valid"] is True
    assert accepted["head"]["supervision"]["t"]["axis_valid"] is False
    assert accepted["axis"]["supervision"]["t"]["head_valid"] is False
    assert accepted["axis"]["supervision"]["t"]["axis_valid"] is True
    assert accepted["head"]["split"] == "val"
    assert accepted["axis"]["split"] == "val"
    assert accepted["both"]["split"] == "train"
    assert not Path(accepted["both"]["movie_path"]).is_absolute()
    assert "reject" not in accepted and "uncertain" not in accepted and "draft" not in accepted
    negatives = [r for r in records if r["kind"] == "background_negative"]
    # With one non-pAJV background, retaining one pAJV rectangle would exceed
    # the 15% cap, so the pAJV rectangle is deterministically omitted.
    assert len(negatives) == 1
    assert all(r["provenance"]["background_is_explicit"] for r in negatives)
    assert all(r["provenance"]["unlabelled_tile_pixels_are_unknown"] for r in negatives)
    assert negatives[0]["dataset_policy"]["pajv101_background_dropped"] == 1


def test_materializer_applies_same_d8_to_pair_and_absolute_targets(tmp_path: Path):
    root, labels, queue = _fixture(tmp_path)
    record = next(
        r
        for r in load_current_records(labels, queue, root, source_tile_size=8)
        if r["sample_id"] == "current:both"
    )
    record = augment_current_record(
        record, seed=7, rotation=1, reflect=True, translation_yx=(1, -1)
    )
    sample = build_current_pair_sample(record, root)
    assert sample.image_t.shape == (8, 8, 3)
    assert sample.image_t.dtype == np.float32
    np.testing.assert_allclose(sample.image_t[..., 1], sample.image_tp1[..., 0])
    np.testing.assert_allclose(sample.image_t[..., 2], sample.image_tp1[..., 1])
    assert len(sample.instances_t) == len(sample.instances_tp1) == 1
    assert sample.instances_t[0].head_valid and sample.instances_t[0].axis_valid
    assert sample.links == [
        (sample.instances_t[0].instance_id, sample.instances_tp1[0].instance_id)
    ]
    y0, x0 = sample.metadata["crop_y0x0"]
    expected = transform_d8_yx(
        np.asarray([8.0 - y0, 8.0 - x0]), 8, rotation=1, reflect=True
    )
    np.testing.assert_allclose(sample.instances_t[0].head_yx, expected)
    assert sample.exhaustive_t is False and sample.exhaustive_tp1 is False


def test_pajv101_is_background_only_and_capped(tmp_path: Path):
    root, labels, queue = _fixture(tmp_path, many_backgrounds=True)
    records = load_current_records(labels, queue, root, source_tile_size=8)
    backgrounds = [r for r in records if r["kind"] == "background_negative"]
    pajv = [r for r in backgrounds if "pAJV101" in r["source_movie"]]
    # 17 non-pAJV backgrounds permit floor(.15 * 17 / .85) == 3 pAJV backgrounds.
    assert len(backgrounds) == 20
    assert len(pajv) == 3
    assert len(pajv) / len(backgrounds) <= 0.15
    assert all(r["constraints"]["pajv101_background_only"] for r in pajv)


def test_manifest_emission_has_exact_movie_split_and_leakage_audit(tmp_path: Path):
    root, labels, queue = _fixture(tmp_path)
    records = load_current_records(labels, queue, root, source_tile_size=8)
    output = root / "SAM3Training/training_data/manifests"
    audit = emit_manifests(records, output, root)
    assert audit["status"] == "passed"
    assert set(audit["validation_movies_observed"]) == set(DEFAULT_VALIDATION_MOVIES)
    train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    val = [json.loads(line) for line in (output / "val.jsonl").read_text().splitlines()]
    assert all(not Path(record["movie_path"]).is_absolute() for record in [*train, *val])
    assert {record["source_movie"] for record in val} == set(DEFAULT_VALIDATION_MOVIES)
    assert not ({record["source_movie"] for record in train} & set(DEFAULT_VALIDATION_MOVIES))
    assert json.loads((output / "leakage_audit.json").read_text())["status"] == "passed"


def test_validation_movies_are_frozen(tmp_path: Path):
    root, labels, queue = _fixture(tmp_path)
    with pytest.raises(ValueError, match="exactly the frozen validation movies"):
        load_current_records(
            labels, queue, root, source_tile_size=8, validation_movies={VAL_A}
        )
