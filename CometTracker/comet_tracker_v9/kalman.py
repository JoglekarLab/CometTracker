"""Constant-velocity Kalman filter on the mask centroid, plus an RTS smoother.

WHAT IS TRACKED, AND WHAT IS NOT

State is ``[y, x, vy, vx]`` -- position and velocity of the mask CENTROID.

Not the head: SAM3's head branch gives 1.37 heads per comet (up to 8), so there
is no reliable plus-end to track. Not a reconstructed tip either. Three
candidate positions were measured on 247 clean real tracks by RMS residual
around a constant-velocity fit:

    mask centroid            0.61 px
    bounding-box centre      0.68 px
    centroid + half-length   2.61 px   <- four times worse

The tip loses because it inherits the full frame-to-frame jitter of the mask's
ENDS, while the centroid averages that jitter down over every pixel. So the
centroid wins on noise. What it costs is a known bias: mask length drifts by a
median 6.8 px over a track, so centroid displacement understates tip
displacement by about half of that. That bias is REAL and is not corrected here.
It is recorded in the track output as ``length_drift`` so a later head
estimator can correct it rather than having it silently baked in.

No length state. V1 carried tail half-length as a filtered state; on SAM3 masks
that state would largely be tracking the detector's confidence, since mask
length is partly a readout of presence (major axis 9.4 -> 17.6 px across
presence bins while minor stays flat).
"""
from __future__ import annotations

import numpy as np

__all__ = ["KalmanState", "predict", "update", "rts_smooth"]

_F = np.array([[1.0, 0.0, 1.0, 0.0],
               [0.0, 1.0, 0.0, 1.0],
               [0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]])
_H = np.array([[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0]])


def transition(dt: int = 1) -> np.ndarray:
    """State transition over ``dt`` frames. dt > 1 is used by gap closing."""
    F = np.eye(4)
    F[0, 2] = F[1, 3] = float(dt)
    return F


def process_noise(q: float, dt: int = 1) -> np.ndarray:
    """Discrete white-noise acceleration model, scaled by ``q`` (px^2/frame^4).

    Using the standard DWNA block rather than a diagonal: the diagonal form
    lets position and velocity errors drift apart, which shows up as a filter
    that trusts a stale velocity too much across a multi-frame gap.
    """
    dt = float(dt)
    t2, t3, t4 = dt ** 2, dt ** 3, dt ** 4
    blk = np.array([[t4 / 4.0, t3 / 2.0], [t3 / 2.0, t2]])
    Q = np.zeros((4, 4))
    for i in (0, 1):
        Q[np.ix_([i, i + 2], [i, i + 2])] = blk
    return Q * float(q)


class KalmanState:
    """One track's filter, keeping the history an RTS pass needs.

    ``xp``/``Pp`` are the PRIOR (predicted) mean and covariance at each step and
    ``xf``/``Pf`` the POSTERIOR. Both are required: the smoother needs the prior
    to form the gain, and keeping only the posterior is the usual way an RTS
    implementation ends up subtly wrong.
    """

    __slots__ = ("x", "P", "q", "r", "xp", "Pp", "xf", "Pf", "dts")

    def __init__(self, y: float, x: float, q: float, r: float,
                 velocity_var: float = 25.0):
        self.x = np.array([y, x, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([r, r, velocity_var, velocity_var]).astype(np.float64)
        self.q = float(q)
        self.r = float(r)
        # history: element k is the state at the k-th recorded frame
        self.xp: list[np.ndarray] = [self.x.copy()]
        self.Pp: list[np.ndarray] = [self.P.copy()]
        self.xf: list[np.ndarray] = [self.x.copy()]
        self.Pf: list[np.ndarray] = [self.P.copy()]
        self.dts: list[int] = [1]

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:].copy()

    def predict(self, dt: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Return the predicted (position, position covariance) WITHOUT
        committing. The linker calls this to build costs for candidates it may
        not take, so it must not mutate the filter."""
        F = transition(dt)
        xp = F @ self.x
        Pp = F @ self.P @ F.T + process_noise(self.q, dt)
        return xp[:2], Pp[:2, :2] + np.eye(2) * self.r

    def step(self, measurement: np.ndarray, dt: int = 1) -> None:
        """Commit one predict+update with a matched measurement."""
        F = transition(dt)
        xp = F @ self.x
        Pp = F @ self.P @ F.T + process_noise(self.q, dt)

        R = np.eye(2) * self.r
        S = _H @ Pp @ _H.T + R
        K = Pp @ _H.T @ np.linalg.inv(S)
        innov = np.asarray(measurement, np.float64) - _H @ xp
        self.x = xp + K @ innov
        # Joseph form: stays positive-definite under repeated updates, which the
        # plain (I-KH)P form does not always do over a 90-frame movie.
        IKH = np.eye(4) - K @ _H
        self.P = IKH @ Pp @ IKH.T + K @ R @ K.T

        self.xp.append(xp)
        self.Pp.append(Pp)
        self.xf.append(self.x.copy())
        self.Pf.append(self.P.copy())
        self.dts.append(int(dt))


def rts_smooth(state: KalmanState) -> np.ndarray:
    """Rauch-Tung-Striebel backward pass. Returns (n, 4) smoothed states.

    Cannot change a link -- it runs after linking is settled. V6 measured the
    equivalent pass at position RMSE 2.04 -> 1.73 px and velocity RMSE
    8.95 -> 7.18 um/min.

    The frame-0 velocity is the clearest win: forward filtering initialises it
    to zero with no evidence, and only the backward pass can fill it in.
    """
    n = len(state.xf)
    out = [None] * n
    out[n - 1] = state.xf[n - 1].copy()
    P = [None] * n
    P[n - 1] = state.Pf[n - 1].copy()
    for k in range(n - 2, -1, -1):
        F = transition(state.dts[k + 1])
        Pp = state.Pp[k + 1]
        try:
            C = state.Pf[k] @ F.T @ np.linalg.inv(Pp)
        except np.linalg.LinAlgError:
            C = state.Pf[k] @ F.T @ np.linalg.pinv(Pp)
        out[k] = state.xf[k] + C @ (out[k + 1] - state.xp[k + 1])
        P[k] = state.Pf[k] + C @ (P[k + 1] - Pp) @ C.T
    return np.array(out)
