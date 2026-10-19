"""Stage 2 -- closing gaps, and calling each one a pause or a shrinkage.

THIS IS THE PART THAT MAKES IT plusTipTracker.

EB3 binds only the GROWING plus-end. When a microtubule stops growing the comet
disappears -- not because the microtubule went away, but because the marker did.
So a nearest-neighbour tracker, and stage 1 on its own, report one microtubule
as several unrelated tracks and can never measure a catastrophe.

Gap closing solves one global assignment over every track END against every
track START within ``time_window`` frames, and labels each accepted join:

  fgap  the start is AHEAD, along the same direction of growth. The microtubule
        paused (or slowed below the threshold that keeps EB3 bound), then
        resumed growing. plusTip's "forward gap".

  bgap  the start is BEHIND, back down the line the microtubule just built. It
        depolymerised. plusTip's "backward gap". This is a catastrophe, and
        recovering it is the whole reason for the asymmetric geometry below.

THE GEOMETRY, AND WHY THE TWO CONES ARE DIFFERENT SIZES

Let ``u`` be the unit vector of the ending track's final velocity, and ``d`` the
displacement from that track's last position to the candidate's first. Split
``d`` into ``p`` along ``u`` and ``q`` across it.

  forward:  p >= -fluct_rad and |angle(d, u)| <= max_f_angle (30 deg)
  backward: p <  -fluct_rad and |angle(d, -u)| <= max_b_angle (10 deg)

The backward wedge is three times tighter on purpose, and it is not a tuning
choice -- it is the physics. A growing microtubule can wander a little, so a
pause may resume slightly off-axis. A SHRINKING one retreats along the lattice
it just built, which is a line it cannot deviate from. A backward candidate
that is off-axis is therefore not the same microtubule.

``fluct_rad`` is the "slightly behind still counts as forward" allowance, so a
track that restarts from a fluctuation during a pause is not misread as a
shrinkage. Without it every pause with a hair of backward jitter becomes a
catastrophe, and the catastrophe frequency -- the headline number -- inflates.

Shrinkage also runs faster than growth, so the backward search reaches
``back_vel_mult`` (1.5) times further than the forward one.

WHAT THIS CANNOT DO WITHOUT A HEAD

Direction here comes from motion: ``u`` is the ending segment's own velocity,
which needs at least two frames. A single-detection segment has no direction and
can only be joined as a passive target, never as a source. With a trustworthy
plus-end this restriction would lift, since the head/tail asymmetry gives
direction from one frame. That is the main thing a working head estimator would
buy stage 2.
"""
from __future__ import annotations

import numpy as np

from .config import Config, GapConfig
from .geometry import directed_angle
from .lap import solve
from .tracks import GAP_BACKWARD, GAP_FORWARD, CompoundTrack, Track

__all__ = ["close_gaps", "break_nonlinear"]


def _segment_speed(tr: Track) -> float:
    v = tr.velocities()
    return float(np.median(np.hypot(v[:, 0], v[:, 1]))) if len(v) else 0.0


def _candidate(end: Track, start: Track, dt: int, cfg: GapConfig):
    """Classify one (end, start) pair. Returns (kind, cost) or None if gated."""
    u = end.end_velocity
    speed = np.hypot(*u)
    if speed < 1e-6:
        return None                      # no direction -> cannot be a source
    u = u / speed

    p_end = end.positions()[-1]
    p_start = start.positions()[0]
    d = p_start - p_end
    dist = float(np.hypot(*d))
    p = float(d @ u)
    q = float(np.hypot(*(d - p * u)))

    # the two growth segments must point the same way, whichever kind of gap it
    # is: a pause and a resumption of the SAME microtubule keep their direction.
    v_start = start.start_velocity
    if np.hypot(*v_start) > 1e-6:
        if np.degrees(directed_angle(u, v_start)) > cfg.max_f_angle_deg:
            return None

    if p >= -cfg.fluct_rad:
        # forward: paused, then resumed
        reach = cfg.fwd_vel_mult * speed * dt + cfg.fluct_rad
        if p > reach:
            return None
        if dist > cfg.fluct_rad:
            if np.degrees(directed_angle(d, u)) > cfg.max_f_angle_deg:
                return None
        return GAP_FORWARD, dist ** 2 + q ** 2
    # backward: shrank
    reach = cfg.back_vel_mult * speed * dt + cfg.fluct_rad
    if -p > reach:
        return None
    if np.degrees(directed_angle(d, -u)) > cfg.max_b_angle_deg:
        return None
    # q is penalised harder going backward: off-lattice retreat is not this MT
    return GAP_BACKWARD, dist ** 2 + 4.0 * q ** 2


def close_gaps(segments: list[Track], config: Config | None = None,
               end_cost: float | None = None) -> list[CompoundTrack]:
    """Join track segments into compound tracks, labelling each gap.

    ``end_cost`` prices "this really is where the track ends". Left as None it
    is set from the accepted-cost distribution (the 75th percentile), which is
    u-track's approach -- deriving the alternative's price from the costs
    actually on offer rather than fixing it in advance.
    """
    cfg = (config or Config()).gap
    n = len(segments)
    if n == 0:
        return []

    starts = np.array([s.start_frame for s in segments])
    ends = np.array([s.end_frame for s in segments])

    cost = np.full((n, n), np.inf)
    kind: dict[tuple[int, int], str] = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dt = int(starts[j] - ends[i])
            if dt <= 0 or dt > cfg.time_window:
                continue
            got = _candidate(segments[i], segments[j], dt, cfg)
            if got is None:
                continue
            k, c = got
            cost[i, j] = c
            kind[(i, j)] = k

    finite = cost[np.isfinite(cost)]
    if end_cost is None:
        end_cost = float(np.percentile(finite, 75)) if finite.size else 1.0
    a = solve(cost, np.full(n, end_cost), np.full(n, end_cost))

    successor = {int(i): int(j) for i, j in a.matches}
    has_pred = set(successor.values())

    compounds: list[CompoundTrack] = []
    for i in range(n):
        if i in has_pred:
            continue
        chain, guard = [i], 0
        while chain[-1] in successor and guard < n:
            chain.append(successor[chain[-1]])
            guard += 1
        gaps = []
        for a_, b_ in zip(chain, chain[1:]):
            gaps.append({
                "kind": kind[(a_, b_)],
                "from_frame": int(segments[a_].end_frame),
                "to_frame": int(segments[b_].start_frame),
                "n_frames": int(segments[b_].start_frame - segments[a_].end_frame),
                "cost": float(cost[a_, b_]),
            })
        compounds.append(CompoundTrack(
            compound_id=len(compounds),
            segments=[segments[k] for k in chain],
            gaps=gaps,
        ))
    return compounds


def break_nonlinear(segments: list[Track], config: Config | None = None) -> list[Track]:
    """Split segments where consecutive steps turn too sharply.

    A port of ``plusTipBreakNonlinearTracks.m``: split wherever the displacement
    vectors of consecutive frame pairs differ in direction by more than
    ``break_angle_deg`` (45), EXCEPT where one of the two steps is very short
    (below the 3rd percentile of all step lengths), since a short step's
    direction is mostly localisation noise rather than a real turn.
    """
    cfg = (config or Config()).gap
    if not cfg.break_nonlinear:
        return segments

    all_steps = []
    for tr in segments:
        p = tr.positions()
        if len(p) >= 2:
            all_steps.append(np.hypot(*np.diff(p, axis=0).T))
    if not all_steps:
        return segments
    short = float(np.percentile(np.concatenate(all_steps), 3))
    limit = np.radians(cfg.break_angle_deg)

    out: list[Track] = []
    for tr in segments:
        p = tr.positions()
        if len(p) < 3:
            out.append(tr)
            continue
        steps = np.diff(p, axis=0)
        lens = np.hypot(steps[:, 0], steps[:, 1])
        cuts = []
        for k in range(len(steps) - 1):
            if lens[k] < short or lens[k + 1] < short:
                continue
            if directed_angle(steps[k], steps[k + 1]) > limit:
                cuts.append(k + 1)
        if not cuts:
            out.append(tr)
            continue
        bounds = [0, *cuts, len(p)]
        for lo, hi in zip(bounds, bounds[1:]):
            if hi - lo < 2:
                continue
            piece = Track(track_id=len(out),
                          det_index=tr.det_index[lo:hi],
                          frames=tr.frames[lo:hi],
                          measured=tr.measured[lo:hi])
            if tr.smoothed is not None:
                piece.smoothed = tr.smoothed[lo:hi]
            if tr.filtered is not None:
                piece.filtered = tr.filtered[lo:hi]
            out.append(piece)
    for i, tr in enumerate(out):
        tr.track_id = i
    return out
