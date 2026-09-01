"""The SAM3 -> prediction-folder decode, without a GPU, SAM3, or a checkpoint.

Everything scripts/sam3_export.py decides about turning 200 object queries
into a probability map lives in comet_sam3/export.py, so it is all reachable
here. The torch shell around it is not tested and says so in its docstring.

The load-bearing one is test_points_and_labels_matches_ilastik_export: if that
fails, the three models are no longer being reduced to detections by the same
rule and none of the comparison numbers mean anything.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from comet_sam3 import export as ex

ROOT = Path(__file__).resolve().parents[2]
ILASTIK_EXPORT = ROOT / "ModelComparison" / "compare_ilastik_UNETV1" / "ilastik_export.py"


def _load_ilastik_export():
    if not ILASTIK_EXPORT.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_ilastik_export", ILASTIK_EXPORT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------- tiling ------

def test_tile_origins_covers_every_pixel_and_ends_flush():
    origins = ex.tile_origins(512, 192, 128)
    assert origins[0] == 0
    assert origins[-1] == 512 - 192
    covered = np.zeros(512, bool)
    for start in origins:
        covered[start:start + 192] = True
    assert covered.all()


def test_tile_origins_exact_fit_does_not_duplicate_the_last_tile():
    assert ex.tile_origins(384, 192, 192) == [0, 192]


def test_tile_origins_frame_smaller_than_a_tile():
    assert ex.tile_origins(100, 192, 128) == [0]


@pytest.mark.parametrize("tile,stride", [(0, 8), (8, 0), (-1, 4)])
def test_tile_origins_rejects_nonpositive(tile, stride):
    with pytest.raises(ValueError):
        ex.tile_origins(64, tile, stride)


def test_paste_max_keeps_the_larger_value_in_the_overlap():
    canvas = np.zeros((6, 6), np.float32)
    ex.paste_max(canvas, np.full((4, 4), 0.9, np.float32), 0, 0)
    ex.paste_max(canvas, np.full((4, 4), 0.3, np.float32), 2, 2)
    assert canvas[3, 3] == pytest.approx(0.9)
    assert canvas[5, 5] == pytest.approx(0.3)


def test_paste_max_crops_overhang_rather_than_raising():
    canvas = np.zeros((5, 5), np.float32)
    ex.paste_max(canvas, np.ones((4, 4), np.float32), 3, 3)
    assert canvas[4, 4] == pytest.approx(1.0)
    assert canvas[0, 0] == pytest.approx(0.0)


def test_predictable_centers_needs_t_minus_2_through_t_plus_1():
    assert list(ex.predictable_centers(10)) == [2, 3, 4, 5, 6, 7, 8]
    assert list(ex.predictable_centers(4)) == [2]
    assert list(ex.predictable_centers(3)) == []
    assert list(ex.predictable_centers(0)) == []


# ------------------------------------------------------------- decode ------

def test_softargmax_is_subpixel_between_two_equal_peaks():
    logits = np.full((9, 9), -20.0)
    logits[4, 3] = 10.0
    logits[4, 5] = 10.0
    y, x = ex.softargmax_yx(logits)
    assert y == pytest.approx(4.0, abs=1e-3)
    assert x == pytest.approx(4.0, abs=1e-3)


def test_softargmax_is_pulled_off_the_argmax_by_a_weaker_neighbour():
    logits = np.full((9, 9), -20.0)
    logits[4, 4] = 10.0
    logits[4, 6] = 9.0
    y, x = ex.softargmax_yx(logits)
    assert y == pytest.approx(4.0, abs=1e-3)
    assert 4.0 < x < 5.0


def test_softargmax_survives_large_logits():
    logits = np.zeros((5, 5))
    logits[1, 1] = 900.0
    y, x = ex.softargmax_yx(logits)
    assert (y, x) == pytest.approx((1.0, 1.0), abs=1e-6)


def test_softargmax_rejects_a_non_map():
    with pytest.raises(ValueError):
        ex.softargmax_yx(np.zeros(9))


def test_dense_axis_map_takes_the_max_not_the_sum():
    axis = np.zeros((2, 4, 4), np.float32)
    axis[0, 1, 1] = 0.6
    axis[1, 1, 1] = 0.5
    out = ex.dense_axis_map(np.array([0.8, 0.8], np.float32), axis)
    assert out[1, 1] == pytest.approx(0.48, abs=1e-6)


def test_dense_axis_map_scales_by_presence_and_stays_in_range():
    axis = np.full((1, 3, 3), 1.0, np.float32)
    assert ex.dense_axis_map(np.array([0.25]), axis).max() == pytest.approx(0.25)
    assert ex.dense_axis_map(np.array([1.5]), axis).max() == pytest.approx(1.0)


def test_dense_axis_map_with_no_accepted_query_is_all_zero():
    out = ex.dense_axis_map(np.zeros((0,), np.float32), np.zeros((0, 7, 5), np.float32))
    assert out.shape == (7, 5)
    assert not out.any()


def test_dense_axis_map_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        ex.dense_axis_map(np.zeros(3), np.zeros((2, 4, 4), np.float32))


def test_dedupe_keeps_the_best_of_a_cluster_and_all_separated_points():
    points = np.array([[10.0, 10.0], [10.5, 10.5], [30.0, 30.0]])
    scores = np.array([0.4, 0.9, 0.5])
    kept = ex.dedupe_points(points, scores, 3.0)
    assert kept.tolist() == [1, 2]


def test_dedupe_is_a_noop_at_zero_distance():
    points = np.zeros((4, 2))
    kept = ex.dedupe_points(points, np.arange(4.0), 0.0)
    assert kept.tolist() == [0, 1, 2, 3]


def test_dedupe_handles_an_empty_frame():
    kept = ex.dedupe_points(np.zeros((0, 2)), np.zeros((0,)), 3.0)
    assert kept.shape == (0,)


def test_dedupe_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        ex.dedupe_points(np.zeros((3, 2)), np.zeros(2), 1.0)


def test_match_links_is_mutual_best_not_row_best():
    # query 0 and query 1 both prefer column 0; only the mutual pair survives,
    # and query 1 goes unlinked rather than being handed the runner-up.
    scores = np.array([[0.9, 0.6], [0.8, 0.55]])
    assert ex.match_links(scores, 0.5) == [(0, 0, 0.9)]


def test_match_links_applies_the_threshold():
    scores = np.array([[0.4, 0.1], [0.1, 0.35]])
    assert ex.match_links(scores, 0.5) == []
    assert [pair[:2] for pair in ex.match_links(scores, 0.3)] == [(0, 0), (1, 1)]


def test_match_links_on_an_empty_side():
    assert ex.match_links(np.zeros((0, 4)), 0.5) == []
    assert ex.match_links(np.zeros((4, 0)), 0.5) == []


# ------------------------------------------------------------- shared ------

def test_points_and_labels_matches_ilastik_export():
    """The one that keeps the comparison honest.

    If these two ever disagree, SAM3's detections/frame is measured by a
    different rule than ilastik's and the U-Net's, and every number in the
    comparison is between incommensurable things.
    """
    module = _load_ilastik_export()
    if module is None:
        pytest.skip(f"ilastik_export.py not found at {ILASTIK_EXPORT}")
    rng = np.random.default_rng(20260831)
    prob = rng.random((4, 24, 26)).astype(np.float32) ** 3
    prob[1, 5:12, 5:9] = 0.95
    prob[2, 14:17, 3:20] = 0.7
    for thresh, min_area in ((0.5, 6), (0.9, 1), (0.2, 30)):
        mine_rows, mine_labels = ex.points_and_labels(prob, thresh, min_area)
        theirs_rows, theirs_labels = module.points_and_labels(prob, thresh, min_area)
        assert mine_rows == theirs_rows
        assert np.array_equal(mine_labels, theirs_labels)


def test_shared_field_names_are_byte_identical_to_ilastik_export():
    module = _load_ilastik_export()
    if module is None:
        pytest.skip(f"ilastik_export.py not found at {ILASTIK_EXPORT}")
    assert ex.POINT_FIELDS == module.POINT_FIELDS
    assert ex.PROB_SUFFIX == module.PROB_SUFFIX
    assert ex.LABEL_SUFFIX == module.LABEL_SUFFIX
    assert ex.POINT_SUFFIX == module.POINT_SUFFIX


# ------------------------------------------------------------ writing ------

def test_write_outputs_round_trips_within_uint8_quantisation(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    rng = np.random.default_rng(7)
    prob = rng.random((3, 12, 14)).astype(np.float32)
    rows, labels = ex.points_and_labels(prob, 0.6, 4)
    base = ex.write_outputs(str(tmp_path), "movie", prob, rows, labels)

    stored = tifffile.imread(base + ex.PROB_SUFFIX)
    assert stored.dtype == np.uint8
    assert stored.shape == prob.shape
    assert np.abs(stored.astype(np.float32) / 255.0 - prob).max() <= 1.0 / 255.0
    assert tifffile.imread(base + ex.LABEL_SUFFIX).dtype == np.int32


def test_points_csv_header_is_byte_for_byte_what_the_other_tools_parse(tmp_path):
    pytest.importorskip("tifffile")
    prob = np.zeros((2, 8, 8), np.float32)
    prob[0, 2:5, 2:5] = 0.9
    rows, labels = ex.points_and_labels(prob, 0.5, 4)
    base = ex.write_outputs(str(tmp_path), "movie", prob, rows, labels)
    with open(base + ex.POINT_SUFFIX) as handle:
        assert handle.readline().strip() == "frame,y,x,area,peak_prob,mean_prob"


def test_component_count_equals_csv_row_count_per_frame(tmp_path):
    pytest.importorskip("tifffile")
    rng = np.random.default_rng(11)
    prob = (rng.random((5, 20, 20)) > 0.7).astype(np.float32)
    rows, labels = ex.points_and_labels(prob, 0.5, 2)
    for frame in range(len(prob)):
        in_csv = sum(1 for row in rows if row["frame"] == frame)
        assert in_csv == int(labels[frame].max())


def test_head_and_link_files_have_their_own_headers(tmp_path):
    head_rows = [dict(frame=2, y=1.0, x=2.0, presence=0.9, axis_peak=0.8,
                      query=17, tile_y=0, tile_x=128)]
    link_rows = [dict(frame=2, y0=1.0, x0=2.0, y1=1.5, x1=3.0, score=0.77)]
    head_path = ex.write_heads(str(tmp_path), "movie", head_rows)
    link_path = ex.write_links(str(tmp_path), "movie", link_rows)
    with open(head_path) as handle:
        assert next(csv.reader(handle)) == ex.HEAD_FIELDS
    with open(link_path) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["score"] == "0.77"
    assert head_path.endswith("_heads.csv")
    assert link_path.endswith("_links.csv")


def test_empty_head_and_link_files_still_have_a_header(tmp_path):
    path = ex.write_heads(str(tmp_path), "movie", [])
    with open(path) as handle:
        assert handle.read().strip() == ",".join(ex.HEAD_FIELDS)
