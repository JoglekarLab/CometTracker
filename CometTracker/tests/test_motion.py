"""What each motion criterion actually does, pinned.

The point of these tests is not that the classes are "right" -- there are no
labels. It is that each criterion does the thing its docstring claims, and in
particular that ``net_disp``'s length dependence is REAL and reproduced here,
so the three-way comparison in the reports is comparing what it says it is.
"""
from __future__ import annotations

import numpy as np
import pytest

from comet_tracker_v9.config import MotionConfig
from comet_tracker_v9.motion import (BROWNIAN, CRITERIA, DIRECTED, IMMOBILE,
                                     classify, classify_every_criterion,
                                     diffusion_ratio)


def straight(n, step, angle=0.3):
    t = np.arange(n)[:, None]
    return t * step * np.array([[np.cos(angle), np.sin(angle)]])


def walk(n, step, seed):
    rng = np.random.default_rng(seed)
    d = rng.normal(0, step / np.sqrt(2), size=(n - 1, 2))
    return np.vstack([[0, 0], np.cumsum(d, axis=0)])


# --- the statistic itself ---------------------------------------------------
def test_diffusion_ratio_is_sqrt_n_for_a_straight_track():
    for n in (3, 5, 9, 21):
        assert diffusion_ratio(straight(n, 2.0)) == pytest.approx(
            np.sqrt(n - 1), rel=1e-9)


def test_diffusion_ratio_is_scale_free():
    """Doubling every step must not change D. That is why no global speed
    constant is needed."""
    for seed in range(20):
        p = walk(12, 1.0, seed)
        assert diffusion_ratio(2 * p) == pytest.approx(diffusion_ratio(p))


def test_diffusion_ratio_has_rms_one_under_a_random_walk():
    d = np.array([diffusion_ratio(walk(15, 1.7, s)) for s in range(4000)])
    assert np.sqrt((d ** 2).mean()) == pytest.approx(1.0, abs=0.05)


def test_diffusion_k_delivers_its_stated_false_positive_rate():
    """P(D > k) = exp(-k^2) is what makes diffusion_k a rate and not a taste."""
    d = np.array([diffusion_ratio(walk(15, 1.7, s)) for s in range(8000)])
    for k, want in [(1.73, 0.05), (2.15, 0.01)]:
        assert (d > k).mean() == pytest.approx(want, abs=0.012)


# --- net_disp: the defect, pinned -------------------------------------------
def test_net_disp_is_a_speed_floor_that_scales_with_one_over_length():
    """Identical per-frame speed, different track length, different class.

    This is the bug. It is pinned rather than fixed because the default
    criterion has to keep reproducing already-published numbers.
    """
    cfg = MotionConfig()
    speed = 1.0  # px/frame -- the same physical motion in both tracks
    short = classify(straight(3, speed), cfg, "net_disp")[0]
    long_ = classify(straight(9, speed), cfg, "net_disp")[0]
    assert short == IMMOBILE
    assert long_ == DIRECTED


def test_speed_floor_is_length_invariant():
    """The minimal repair: same per-frame speed, same class at any length."""
    cfg = MotionConfig()
    for speed, want in [(0.5, IMMOBILE), (2.0, DIRECTED)]:
        got = {classify(straight(n, speed), cfg, "speed_floor")[0]
               for n in (3, 5, 9, 15)}
        assert got == {want}, f"speed {speed} gave {got}"


# --- diffusion: what it will and will not claim ------------------------------
def test_diffusion_refuses_to_call_a_three_frame_track_directed():
    """max D = sqrt(2) = 1.414 < 1.73, whatever the speed. Two steps cannot
    separate a line from a random walk at 5% confidence, and the criterion
    declines rather than guessing."""
    cfg = MotionConfig()
    for speed in (1.0, 5.0, 50.0):
        assert classify(straight(3, speed), cfg, "diffusion")[0] != DIRECTED


def test_diffusion_calls_a_long_straight_track_directed():
    cfg = MotionConfig()
    for n in (7, 11, 19):
        assert classify(straight(n, 1.2), cfg, "diffusion")[0] == DIRECTED


def test_diffusion_calls_a_random_walk_brownian_most_of_the_time():
    cfg = MotionConfig()
    cls = [classify(walk(12, 1.7, s), cfg, "diffusion")[0] for s in range(500)]
    assert cls.count(DIRECTED) / len(cls) < 0.10


def test_diffusion_calls_pure_jitter_immobile():
    cfg = MotionConfig()
    assert classify(walk(10, 0.3, 0), cfg, "diffusion")[0] == IMMOBILE


# --- the comparison harness -------------------------------------------------
def test_every_criterion_agrees_once_mss_takes_over():
    """Above min_frames_for_mss the criteria are not consulted at all, so the
    three-way comparison is only ever about short tracks."""
    cfg = MotionConfig()
    info = classify_every_criterion(straight(30, 2.0), cfg)
    assert info["method"] == "mss"
    assert len(set(info["classes"].values())) == 1


def test_every_criterion_matches_classify_one_at_a_time():
    cfg = MotionConfig()
    for n, step, seed in [(3, 2.0, 1), (5, 0.4, 2), (9, 1.1, 3), (15, 3.0, 4)]:
        p = walk(n, step, seed)
        info = classify_every_criterion(p, cfg)
        for c in CRITERIA:
            assert info["classes"][c] == classify(p, cfg, c)[0]


def test_unknown_criterion_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError, match="unknown motion criterion"):
        classify(straight(5, 1.0), MotionConfig(), "vibes")
