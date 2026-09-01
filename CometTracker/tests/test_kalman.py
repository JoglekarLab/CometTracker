import numpy as np
import pytest

from comet_tracker_v7.kalman import KalmanState, process_noise, rts_smooth, transition


def test_transition_moves_position_by_velocity():
    F = transition(3)
    x = np.array([10.0, 20.0, 1.0, -2.0])
    assert (F @ x)[:2].tolist() == [13.0, 14.0]


def test_process_noise_is_positive_semidefinite():
    for dt in (1, 2, 5):
        Q = process_noise(1.0, dt)
        assert np.allclose(Q, Q.T)
        assert (np.linalg.eigvalsh(Q) >= -1e-12).all()


def test_covariance_stays_positive_definite_over_a_long_track():
    """Joseph form. The plain (I-KH)P update drifts non-PD over ~90 frames."""
    kf = KalmanState(0.0, 0.0, q=1.0, r=0.4)
    rng = np.random.default_rng(0)
    for k in range(1, 91):
        kf.step(np.array([2.0 * k, 1.0 * k]) + rng.normal(0, 0.6, 2))
    assert (np.linalg.eigvalsh(kf.P) > 0).all()


def test_filter_recovers_a_constant_velocity():
    """The INSTANTANEOUS filtered velocity is noisy by design -- q sets how much
    the filter lets velocity wander, so a single final-step estimate is not the
    thing to assert on. Two claims that are meaningful: the smoothed velocity
    averaged over the track is close, and tightening q tightens the estimate."""
    rng = np.random.default_rng(1)
    v = np.array([2.36, -1.1])

    def run(q):
        kf = KalmanState(0.0, 0.0, q=q, r=0.4)
        r = np.random.default_rng(1)
        for k in range(1, 40):
            kf.step(v * k + r.normal(0, 0.6, 2))
        return kf

    kf = run(0.1)
    mean_v = rts_smooth(kf)[:, 2:].mean(axis=0)
    assert np.allclose(mean_v, v, atol=0.15), f"mean smoothed velocity {mean_v}"

    loose = np.abs(run(1.0).velocity - v).sum()
    tight = np.abs(run(0.001).velocity - v).sum()
    assert tight < loose, "a smaller q must give a steadier velocity estimate"


def test_rts_gives_frame_zero_a_velocity_the_filter_could_not_have():
    """Forward filtering initialises velocity to zero with no evidence; only the
    backward pass can fill it in."""
    kf = KalmanState(0.0, 0.0, q=0.5, r=0.4)
    v = np.array([2.0, 1.0])
    for k in range(1, 15):
        kf.step(v * k)
    assert np.hypot(*kf.xf[0][2:]) < 1e-9
    sm = rts_smooth(kf)
    assert np.allclose(sm[0, 2:], v, atol=0.3)


def test_rts_does_not_move_the_last_state():
    kf = KalmanState(0.0, 0.0, q=0.5, r=0.4)
    for k in range(1, 10):
        kf.step(np.array([k * 2.0, 0.0]))
    sm = rts_smooth(kf)
    assert np.allclose(sm[-1], kf.xf[-1])


def test_predict_does_not_mutate():
    kf = KalmanState(3.0, 4.0, q=1.0, r=0.4)
    before_x, before_P = kf.x.copy(), kf.P.copy()
    kf.predict(2)
    assert np.array_equal(kf.x, before_x) and np.array_equal(kf.P, before_P)


def test_rts_reduces_position_error_against_the_raw_measurements():
    rng = np.random.default_rng(3)
    v = np.array([2.36, 0.9])
    truth = np.array([v * k for k in range(30)])
    meas = truth + rng.normal(0, 0.61, truth.shape)
    kf = KalmanState(meas[0][0], meas[0][1], q=0.3, r=0.4)
    for m in meas[1:]:
        kf.step(m)
    sm = rts_smooth(kf)[:, :2]
    raw_err = np.sqrt(np.mean((meas - truth) ** 2))
    sm_err = np.sqrt(np.mean((sm - truth) ** 2))
    assert sm_err < raw_err
