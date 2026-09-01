"""Stage 1 -- frame-to-frame linking.

THE CORRIDOR GATE, WHICH IS THE POINT OF THIS TRACKER

A microtubule grows along a line, and a SAM3 mask is a rendered centreline, so
the mask's principal axis IS that line. Measured on real predictions over 247
clean tracks:

    mask axis vs net direction of travel   median  5.4 deg   (96.8% within 30)
    mask axis vs that frame's step         median  6.8 deg
    across-axis component |q| of a step    median  0.17 px, p90 0.52, p99 2.16

So a candidate whose step runs sideways out of the comet's own axis is not that
comet. The gate is ``|q| <= corridor_min + mult * |step| * sigma_theta``.

Three properties, all of which matter:

  * IT EXISTS AT BIRTH. The axis comes from the mask's shape, not from motion,
    so it gates a track's very first link -- which V6 measured to be 22.8% of
    all links, and only 3.2% of those correct. No motion-derived gate can reach
    them, because at birth there is no motion yet. This is also why V7 does not
    need u-track's forward/backward/forward pass scheme, whose purpose is to
    bootstrap exactly this missing initial direction.

  * IT IS INDEPENDENT EVIDENCE. A motion-derived axis is bent by the very
    mis-links the gate is meant to catch. Shape cannot be.

  * IT DOES NOT SCALE WITH THE FRAME GAP. A comet going straight stays on its
    line however many frames pass; only its position ALONG the line becomes
    uncertain. V6 measured the constant corridor at 1.96x lift vs 1.42x for one
    scaled per frame. The ``sigma_theta`` term here is not a per-frame scaling:
    it widens the corridor for masks whose axis is genuinely uncertain (round
    ones), which is a different thing.

The corridor's value comes from letting ``max_disp`` stay LOOSE without paying
for it in precision. V6 measured that at max_disp 5 the corridor is inert, and
at 9 it removes 36% of spurious tracks at no recall cost.

WHAT THE COST IS

Squared Mahalanobis distance between the Kalman prediction and the detection,
plus ``score_weight * (1 - score)``. Presence is a COST, never a gate -- see
lap.py for why that distinction is the whole design.
"""
from __future__ import annotations

import numpy as np

from .config import Config, LinkConfig
from .detections import DetectionTable
from .geometry import decompose, directed_angle
from .kalman import KalmanState, rts_smooth
from .lap import solve
from .tracks import Track

__all__ = ["run_linking"]


class _Live:
    """A track under construction, with the bits the gates need."""

    __slots__ = ("track", "kf", "misses", "last_pos", "last_theta", "hits")

    def __init__(self, track_id: int, det: int, frame: int, pos: np.ndarray,
                 theta: float, cfg: LinkConfig, velocity: np.ndarray | None):
        self.track = Track(track_id=track_id, det_index=[det], frames=[frame],
                           measured=[pos.copy()])
        self.kf = KalmanState(pos[0], pos[1], cfg.process_noise, cfg.measurement_noise)
        if velocity is not None:
            # seeding from a previous pass -- this is what u-track's multi-pass
            # scheme achieves, and the only thing it achieves
            self.kf.x[2:] = velocity
            self.kf.xf[0] = self.kf.x.copy()
            self.kf.xp[0] = self.kf.x.copy()
        self.misses = 0
        self.hits = 1
        self.last_pos = pos.copy()
        self.last_theta = float(theta)


def _build_costs(live: list[_Live], det_idx: np.ndarray, dets: DetectionTable,
                 cfg: LinkConfig) -> np.ndarray:
    """(n_live, n_det) link costs, inf where gated."""
    n, m = len(live), det_idx.size
    cost = np.full((n, m), np.inf)
    if n == 0 or m == 0:
        return cost

    det_pos = np.stack([dets.cy[det_idx], dets.cx[det_idx]], axis=1)
    det_theta = dets.theta[det_idx]
    det_sigma = dets.sigma_theta[det_idx]
    det_score = dets.score[det_idx]
    max_angle = np.radians(cfg.max_link_angle_deg)

    for i, lv in enumerate(live):
        dt = lv.misses + 1
        pred, S = lv.kf.predict(dt)

        # --- gate 1: how far it could have gone ---------------------------------
        resid = det_pos - pred
        dist_pred = np.hypot(resid[:, 0], resid[:, 1])
        ok = dist_pred <= cfg.max_disp * dt

        # --- gate 2: the corridor ----------------------------------------------
        # measured from the track's LAST OBSERVED position, not the prediction:
        # the physical claim is that the comet moves along its own axis, and the
        # prediction already has velocity folded into it.
        step = det_pos - lv.last_pos
        step_len = np.hypot(step[:, 0], step[:, 1])
        _, q = decompose(step, det_theta)
        corridor = cfg.corridor_min + cfg.corridor_sigma_mult * step_len * det_sigma
        ok &= np.abs(q) <= corridor

        # --- gate 3: direction, once there is a direction to compare to ---------
        if lv.hits >= 2:
            v = lv.kf.velocity
            if np.hypot(*v) > 1e-6:
                ok &= directed_angle(np.broadcast_to(v, step.shape), step) <= max_angle

        if not ok.any():
            continue

        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)
        maha = np.einsum("ij,jk,ik->i", resid, Sinv, resid)
        cost[i] = np.where(ok, maha + cfg.score_weight * (1.0 - det_score), np.inf)
    return cost


def _one_pass(dets: DetectionTable, cfg: LinkConfig, frames: list[int],
              seed_velocity: dict[int, np.ndarray] | None,
              reverse: bool) -> list[Track]:
    by_frame = dets.indices_by_frame()
    live: list[_Live] = []
    done: list[Track] = []
    next_id = 0
    sign = -1.0 if reverse else 1.0

    for frame in frames:
        det_idx = by_frame.get(frame, np.zeros(0, dtype=np.int64))
        cost = _build_costs(live, det_idx, dets, cfg)

        if det_idx.size:
            score = dets.score[det_idx]
            birth = cfg.birth_cost + cfg.birth_score_weight * (1.0 - score)
        else:
            birth = np.zeros(0)
        death = np.full(len(live), cfg.death_cost)

        a = solve(cost, birth, death)

        n_old = len(live)
        matched_live, matched_det = set(), set()
        for i, j in a.matches:
            lv = live[i]
            d = det_idx[j]
            pos = np.array([dets.cy[d], dets.cx[d]])
            lv.kf.step(pos, dt=lv.misses + 1)
            lv.track.det_index.append(int(d))
            lv.track.frames.append(int(frame))
            lv.track.measured.append(pos)
            lv.last_pos = pos
            lv.last_theta = float(dets.theta[d])
            lv.misses = 0
            lv.hits += 1
            matched_live.add(i)
            matched_det.add(j)

        # unmatched detections start new tracks
        for j in range(det_idx.size):
            if j in matched_det:
                continue
            d = int(det_idx[j])
            v = None
            if seed_velocity is not None and d in seed_velocity:
                v = seed_velocity[d] * sign
            live.append(_Live(next_id, d, int(frame),
                              np.array([dets.cy[d], dets.cx[d]]),
                              float(dets.theta[d]), cfg, v))
            next_id += 1

        # retire. Only the tracks that already existed this frame can MISS;
        # the ones appended just above were born here and have not had a chance
        # to be matched, so n_old is captured before the births.
        still: list[_Live] = []
        for i, lv in enumerate(live):
            if i < n_old and i not in matched_live:
                lv.misses += 1
            if lv.misses > cfg.max_gap:
                done.append(lv.track)
            else:
                still.append(lv)
        live = still

    done.extend(lv.track for lv in live)
    return done


def run_linking(dets: DetectionTable, config: Config | None = None) -> list[Track]:
    """Link detections into track segments.

    Returns segments of at least ``min_track_length`` detections, each carrying
    its filtered and RTS-smoothed state. Gaps are NOT closed here -- with
    ``max_gap = 0`` (V6's measured default) a comet that blinks produces several
    segments, and rejoining them is ``gapclose.py``'s job.
    """
    cfg = (config or Config()).link
    if cfg.n_passes % 2 == 0:
        raise ValueError("n_passes must be odd so the final pass runs forwards")
    if len(dets) == 0:
        return []

    frames = list(dets.frame_range())
    seed: dict[int, np.ndarray] | None = None
    segments: list[Track] = []

    for p in range(cfg.n_passes):
        reverse = (p % 2 == 1)
        order = frames[::-1] if reverse else frames
        segments = _one_pass(dets, cfg, order, seed, reverse)
        if p < cfg.n_passes - 1:
            seed = {}
            for tr in segments:
                if len(tr) < 2:
                    continue
                pos = np.asarray(tr.measured)
                v = (pos[-1] - pos[0]) / max(len(pos) - 1, 1)
                for d in tr.det_index:
                    seed[int(d)] = v

    out: list[Track] = []
    for tr in segments:
        if len(tr) < cfg.min_track_length:
            continue
        # n_passes is odd so the final pass ran forwards, but assert it rather
        # than assume: a reversed track would give the RTS pass a negative dt
        # and silently flip every velocity sign.
        if tr.frames[0] > tr.frames[-1]:
            tr.det_index.reverse()
            tr.frames.reverse()
            tr.measured.reverse()
        out.append(tr)

    for i, tr in enumerate(out):
        tr.track_id = i
        kf = KalmanState(tr.measured[0][0], tr.measured[0][1],
                         cfg.process_noise, cfg.measurement_noise)
        for k in range(1, len(tr.measured)):
            kf.step(tr.measured[k], dt=tr.frames[k] - tr.frames[k - 1])
        tr.kalman = kf
        tr.filtered = np.array(kf.xf)
        tr.smoothed = rts_smooth(kf)
    return out
