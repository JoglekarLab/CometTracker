"""v9: EMG plus-end positions, the mask-only refit, the turned background box,
and the EMG track speeds."""
import numpy as np
import pytest

from comet_tracker_v9 import photometry as PH
from comet_tracker_v9 import postprocess as PP


def _comet(s, mu, lam=5.0, sd=1.5, A=100.0, tail=-1):
    """EMG with its plus end at mu and the tail running toward `tail`."""
    return PH._emg(tail * s, A, lam, tail * mu, sd, 0.0)


def test_core_recovers_plus_end_and_max_sits_behind_it():
    s = np.arange(-16, 16.01, 0.5)
    rng = np.random.default_rng(0)
    y = _comet(s, mu=2.0) + rng.normal(0, 2, s.size)
    r = PH._emg_core(s, y)
    assert r is not None and r["r2"] > 0.95
    assert abs(r["mu_s"] - 2.0) < 0.5
    assert r["max_s"] < r["mu_s"]            # top of the curve is behind the plus end
    assert r["mu_err"] < 1.0


def test_short_mask_gets_no_fit():
    s = np.arange(-12, 12.01, 0.5)
    ep = PH._emg_positions(s, _comet(s, 0.5), s, _comet(s, 0.5), half=1.5, plus_dir=1)
    assert ep["window"] == -1 and ep["ok"] == 0 and np.isnan(ep["mu_s"])


def test_bright_neighbour_outside_mask_triggers_mask_only_refit():
    s = np.arange(-28, 28.01, 0.5)
    truth = 3.0
    y = _comet(s, truth) + 300.0 * np.exp(-((s - 13.0) ** 2) / (2 * 1.5 ** 2))
    half = 6.0
    near = np.abs(s) <= half + 10
    ep = PH._emg_positions(s[near], y[near], s, y, half, plus_dir=1)
    assert ep["window"] == 2 and ep["ok"] == 1
    assert abs(ep["mu_s"] - truth) < 1.5


def test_wrong_way_round_is_not_mistaken_for_the_plus_end():
    s = np.arange(-16, 16.01, 0.5)
    y = _comet(s, 2.0)                        # plus end faces +s
    right = PH._emg_positions(s, y, s, y, 6.0, plus_dir=1)
    wrong = PH._emg_positions(s, y, s, y, 6.0, plus_dir=-1)
    assert right["ok"] == 1 and abs(right["mu_s"] - 2.0) < 0.5
    # forced the wrong way, the fit either fails or cannot land on the true end
    assert wrong["ok"] == 0 or abs(wrong["mu_s"] - 2.0) > 1.0


def test_refit_floor_is_fixed_at_zero():
    s = np.arange(-6, 6.01, 0.5)
    r = PH._emg_core(s, _comet(s, 2.0), floor=0.0)
    assert r is not None and abs(r["mu_s"] - 2.0) < 0.5


def test_box_background_ignores_far_structure():
    H = W = 61
    patch = np.full((H, W), 100.0)
    patch[:, :8] = 900.0                      # a bundle far to one side
    own = np.zeros((H, W), bool); own[28:33, 22:39] = True   # horizontal comet
    patch[own] = 400.0
    free = ~own
    med, sd, n = PH._bg_box(patch, own, free, 30.0, 30.0, np.pi / 2, 8.0)
    assert n >= PH._BOX_MIN_PX and med == pytest.approx(100.0)


def test_robust_speed_ignores_one_jump():
    f = np.arange(8, dtype=float)
    pos = np.c_[np.zeros(8), 3.0 * f]
    pos[2, 1] -= 25.0                         # one frame jumps far back
    v = PP._robust_speed_px_per_frame(f, pos)
    assert v == pytest.approx(3.0, abs=0.2)


def test_emg_speed_needs_three_frames_and_skips_bad_ones():
    n = 6
    good = {"ok": np.array([1, 1, 0, 1, 1, 1], float),
            "window": np.array([1, 2, -1, 1, 1, 1], float),
            "mu_y": np.zeros(n), "mu_x": 3.0 * np.arange(n),
            "max_y": np.zeros(n), "max_x": 3.0 * np.arange(n) - 2.0}
    good["mu_x"][2] = -100.0                 # the skipped frame's junk is never read
    bad = {k: (np.ones(n) if k == "ok" else np.ones(n) if k == "window" else
               np.full(n, 1e3) * (np.arange(n) % 2)) for k in good}
    # the comets' axes point along +x (theta = pi/2), travel is +x, so the
    # "_p" fits are the ones facing the travel; "_n" holds junk that must be ignored
    phot = {f"emg_{k}_p": v for k, v in good.items()}
    phot.update({f"emg_{k}_n": v for k, v in bad.items()})
    theta = np.full(n, np.pi / 2)
    old = PP._set_scale(0.1, 2.0)
    try:
        r = PP._emg_speeds(phot, np.arange(n), np.arange(n), theta, (0.0, 3.0))
        assert r["n_emg_frames"] == 5 and r["n_emg_mask_frames"] == 1
        assert r["n_emg_short_frames"] == 1
        assert r["v_emg_mu_um_min"] == pytest.approx(3.0 * 0.1 / 2.0 * 60)
        r_rev = PP._emg_speeds(phot, np.arange(n), np.arange(n), theta, (0.0, -3.0))
        assert r_rev["v_emg_mu_um_min"] != pytest.approx(r["v_emg_mu_um_min"])  # took "_n"
        phot["emg_ok_p"][:] = [1, 1, 0, 0, 0, 0]
        r = PP._emg_speeds(phot, np.arange(n), np.arange(n), theta, (0.0, 3.0))
        assert np.isnan(r["v_emg_mu_um_min"]) and r["n_emg_frames"] == 2
    finally:
        PP._set_scale(*old)


def test_measure_end_to_end_on_a_moving_synthetic_comet():
    T, H, W = 5, 80, 120
    yy, xx = np.mgrid[0:H, 0:W]
    movie = np.full((T, H, W), 200.0)
    labels = np.zeros((T, H, W), np.int32)
    rng = np.random.default_rng(1)
    truth = []
    for t in range(T):
        mu = 40.0 + 3.0 * t                   # plus end moves +x by 3 px/frame
        # tail toward -x: EMG in mirrored coordinate
        with np.errstate(all="ignore"):
            a = np.nan_to_num(PH._emg(-(xx.astype(float)), 400.0, 4.0, -mu, 1.2, 0.0),
                              nan=0.0, posinf=0.0)   # erfcx overflows far out
        a[np.abs(xx - mu) > 40] = 0.0
        img = a * np.exp(-((yy - 40.0) ** 2) / (2 * 1.3 ** 2))
        movie[t] += img + rng.normal(0, 4, (H, W))
        labels[t][img > 60] = 1
        truth.append(mu)
    out = PH.measure(movie, labels)
    n = len(out["peak"])
    assert n == T and all(len(v) == n for v in out.values())
    # the comet's axis is along x; its plus end faces +x. Which of _p / _n
    # that is depends on the sign regionprops gave the axis.
    from skimage.measure import regionprops
    sgn = np.array([np.sin(regionprops(labels[t])[0].orientation) for t in range(T)])
    mu_x = np.where(sgn > 0, out["emg_mu_x_p"], out["emg_mu_x_n"])
    ok = np.where(sgn > 0, out["emg_ok_p"], out["emg_ok_n"]) > 0
    assert ok.sum() == T
    err = np.abs(mu_x[ok] - np.asarray(truth)[ok])
    assert np.max(err) < 1.0
    assert np.all(np.isfinite(out["bg"])) and np.all(np.abs(out["bg"] - 200.0) < 10)
