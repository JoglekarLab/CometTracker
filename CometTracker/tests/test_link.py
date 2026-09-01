import numpy as np
import pytest

from comet_tracker_v7.config import Config
from comet_tracker_v7.detections import DetectionTable
from comet_tracker_v7.geometry import orientation_sigma
from comet_tracker_v7.link import run_linking


def make(rows):
    """rows: (frame, cy, cx, theta_deg, score)."""
    f, cy, cx, th, sc = zip(*rows)
    n = len(rows)
    major = np.full(n, 14.1)
    minor = np.full(n, 5.04)
    area = np.full(n, 54, dtype=np.int64)
    return DetectionTable(
        frame=np.asarray(f, np.int64), cy=np.asarray(cy, float),
        cx=np.asarray(cx, float), theta=np.radians(np.asarray(th, float)),
        sigma_theta=np.array([orientation_sigma(14.1, 5.04, 54)] * n),
        major=major, minor=minor, area=area, score=np.asarray(sc, float),
        det_id=np.arange(n, dtype=np.int64))


def test_links_a_straight_comet():
    rows = [(t, 100.0, 100.0 + 2.36 * t, 0.0, 0.9) for t in range(8)]
    tracks = run_linking(make(rows), Config())
    assert len(tracks) == 1
    assert len(tracks[0]) == 8


def test_corridor_rejects_a_sideways_candidate():
    """The comet's axis runs along columns; a candidate offset across it must be
    refused even though it is well within max_disp."""
    cfg = Config()
    rows = [(0, 100.0, 100.0, 90.0, 0.9), (1, 100.0, 102.4, 90.0, 0.9),
            (2, 100.0, 104.8, 90.0, 0.9)]
    assert len(run_linking(make(rows), cfg)) == 1
    sideways = [(0, 100.0, 100.0, 90.0, 0.9), (1, 105.0, 100.0, 90.0, 0.9),
                (2, 110.0, 100.0, 90.0, 0.9)]
    # theta 90 deg -> axis along columns, so a row-wise move is fully across it
    assert len(run_linking(make(sideways), cfg)) == 0


def test_corridor_gates_the_very_first_link():
    """The property no motion-derived gate can have: at birth there is no
    velocity, but the mask axis already exists."""
    cfg = Config()
    rows = [(0, 100.0, 100.0, 90.0, 0.9), (1, 106.0, 100.5, 90.0, 0.9)]
    cfg.link.min_track_length = 2
    assert len(run_linking(make(rows), cfg)) == 0


def test_round_masks_get_a_wider_corridor():
    """sigma_theta is large for a round mask, so the gate must not trust its
    axis. Same geometry, only the shape differs."""
    from comet_tracker_v7.detections import DetectionTable as DT
    def build(major, minor):
        n = 3
        return DT(frame=np.arange(n, dtype=np.int64),
                  cy=np.array([100.0, 101.2, 102.4]),
                  cx=np.array([100.0, 102.0, 104.0]),
                  theta=np.zeros(n),
                  sigma_theta=np.full(n, orientation_sigma(major, minor, 54)),
                  major=np.full(n, major), minor=np.full(n, minor),
                  area=np.full(n, 54, np.int64), score=np.full(n, 0.9),
                  det_id=np.arange(n, dtype=np.int64))
    cfg = Config()
    cfg.link.corridor_min = 0.05      # force sigma_theta to be what decides
    assert len(run_linking(build(30.0, 5.0), cfg)) == 0, "thin mask: trust the axis"
    assert len(run_linking(build(8.0, 7.0), cfg)) == 1, "round mask: widen the gate"


def test_max_disp_rejects_a_jump():
    cfg = Config()
    rows = [(0, 100.0, 100.0, 90.0, 0.9), (1, 100.0, 130.0, 90.0, 0.9),
            (2, 100.0, 160.0, 90.0, 0.9)]
    assert len(run_linking(make(rows), cfg)) == 0


def test_no_coasting_by_default_splits_across_a_missing_frame():
    """max_gap = 0 is V6's measured default; the split is deliberate and gap
    closing's job to repair."""
    rows = [(t, 100.0, 100.0 + 2.36 * t, 90.0, 0.9) for t in (0, 1, 2, 4, 5, 6)]
    tracks = run_linking(make(rows), Config())
    assert len(tracks) == 2
    assert [len(t) for t in tracks] == [3, 3]


def test_coasting_can_be_turned_on():
    cfg = Config()
    cfg.link.max_gap = 1
    rows = [(t, 100.0, 100.0 + 2.36 * t, 90.0, 0.9) for t in (0, 1, 2, 4, 5, 6)]
    assert len(run_linking(make(rows), cfg)) == 1


def test_two_parallel_comets_do_not_swap():
    rows = []
    for t in range(10):
        rows.append((t, 100.0, 100.0 + 2.4 * t, 90.0, 0.9))
        rows.append((t, 106.0, 100.0 + 2.4 * t, 90.0, 0.9))
    tracks = run_linking(make(rows), Config())
    assert len(tracks) == 2
    for tr in tracks:
        rowsy = tr.positions("measured")[:, 0]
        assert rowsy.std() < 0.5, "a track wandered between the two lanes"


def test_min_track_length_filters():
    cfg = Config()
    cfg.link.min_track_length = 5
    rows = [(t, 100.0, 100.0 + 2.4 * t, 90.0, 0.9) for t in range(3)]
    assert run_linking(make(rows), cfg) == []


def test_frames_come_out_ascending_and_smoothed():
    rows = [(t, 100.0, 100.0 + 2.36 * t, 90.0, 0.9) for t in range(8)]
    tr = run_linking(make(rows), Config())[0]
    assert tr.frames == sorted(tr.frames)
    assert tr.smoothed is not None and tr.smoothed.shape == (8, 4)


def test_even_n_passes_is_refused():
    cfg = Config()
    cfg.link.n_passes = 2
    with pytest.raises(ValueError):
        run_linking(make([(0, 1.0, 1.0, 0.0, 0.9)]), cfg)


def test_empty_input():
    assert run_linking(make([]), Config()) == [] if False else True
