"""Synthetic detections with known provenance, for scoring the linker alone.

This scores LINKING, not detection. It emits a DetectionTable directly -- no
images, no SAM3 -- so a change in the numbers is a change in the tracker.

Every parameter is set from something measured, and the source is named. The
two that are NOT measured are the clutter density and the miss rate; V6 flagged
the same two as the parameters its results were most sensitive to, and that
warning carries over verbatim.

WHAT IS SIMULATED

  * comets travel in a straight line at a constant speed (EB3 growth is
    persistent; direction changes are what plusTipBreakNonlinearTracks exists
    to catch, and are added by ``turn_rate`` when asked for)
  * the mask axis is the direction of travel plus noise, because that is the
    relationship the corridor gate depends on and the thing most worth stressing
  * two kinds of clutter, because they fail differently: TRANSIENT clutter
    appears for one frame and cannot form a track, while STATIC clutter persists
    and jiggles, which is what actually welds itself into spurious tracks
  * detections are dropped at ``miss_rate``, which is what fragments a comet
    into several segments and gives gap closing something to do
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detections import DetectionTable
from .geometry import orientation_sigma

__all__ = ["BenchConfig", "make_movie", "score", "run_benchmark"]


@dataclass
class BenchConfig:
    size: int = 512                 # [real] 512x512 frames
    n_frames: int = 30              # [V6] matches V6's benchmark length
    n_comets: int = 120             # [V6] matches V6's benchmark
    speed_mean: float = 2.36        # [HANDOFF] median 2.36 px/frame
    speed_sd: float = 0.8           # [guess]
    lifetime_min: int = 3           # [HANDOFF] tracks live 3-25 frames
    lifetime_max: int = 25
    axis_noise_deg: float = 5.4     # [V7] median mask-axis vs travel angle
    centroid_noise: float = 0.61    # [V7] RMS residual of the mask centroid
    major_mean: float = 14.1        # [V7] median component major axis
    minor_mean: float = 5.04        # [V7] median component minor axis
    shape_sd: float = 3.0           # [guess]
    score_mean: float = 0.92        # [V7] median score of detections in
                                    # straight moving real tracks
    score_sd: float = 0.10
    miss_rate: float = 0.25         # [V6 ESTIMATE, not a measurement]
    clutter_transient: int = 100    # [V6 ESTIMATE] per frame
    clutter_static: int = 40        # [V6 ESTIMATE] persistent blobs
    static_jitter: float = 0.5      # [guess] px
    clutter_score_mean: float = 0.90
    # [V7] MEASURED, and it is the important one. On real SAM3 predictions the
    # score of a detection inside a straight moving track (median 0.922) is
    # barely above one inside a track that goes nowhere (0.898): AUC 0.608,
    # where 0.5 means no information at all. Any threshold below 0.8 keeps 100%
    # of BOTH populations.
    #
    # An earlier version of this file guessed 0.70 here, which invented a
    # separation that does not exist and made a hard presence threshold look
    # like it cut spurious tracks from 21% to 7%. It does not. Do not raise this
    # gap without a measurement to back it.
    turn_rate: float = 0.0          # [guess] rad/frame of direction change
    seed: int = 0


@dataclass
class Truth:
    """Which comet (if any) produced each detection. -1 means clutter."""
    source: np.ndarray
    n_comets: int
    positions: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)


def _shape(rng, cfg, n):
    major = np.maximum(rng.normal(cfg.major_mean, cfg.shape_sd, n), 6.0)
    minor = np.clip(rng.normal(cfg.minor_mean, 0.6, n), 3.0, major - 0.5)
    area = np.maximum((np.pi / 4.0) * major * minor * 0.55, 6).astype(int)
    return major, minor, area


def make_movie(config: BenchConfig | None = None) -> tuple[DetectionTable, Truth]:
    cfg = config or BenchConfig()
    rng = np.random.default_rng(cfg.seed)

    frame, cy, cx, theta, major, minor, area, score, source = ([] for _ in range(9))
    truth_pos: dict[int, dict[int, np.ndarray]] = {}

    # ── comets ────────────────────────────────────────────────────────────────
    for cid in range(cfg.n_comets):
        life = int(rng.integers(cfg.lifetime_min, cfg.lifetime_max + 1))
        t0 = int(rng.integers(0, max(cfg.n_frames - 2, 1)))
        life = min(life, cfg.n_frames - t0)
        if life < 2:
            continue
        direction = rng.uniform(-np.pi, np.pi)
        speed = max(rng.normal(cfg.speed_mean, cfg.speed_sd), 0.3)
        margin = speed * life + 20
        pos = np.array([rng.uniform(margin, cfg.size - margin),
                        rng.uniform(margin, cfg.size - margin)])
        truth_pos[cid] = {}
        mj, mn, ar = _shape(rng, cfg, life)
        for k in range(life):
            direction += rng.normal(0.0, cfg.turn_rate)
            step = speed * np.array([np.cos(direction), np.sin(direction)])
            truth_pos[cid][t0 + k] = pos.copy()
            if rng.random() < cfg.miss_rate:
                pos = pos + step
                continue
            noisy = pos + rng.normal(0, cfg.centroid_noise, 2)
            frame.append(t0 + k)
            cy.append(noisy[0]); cx.append(noisy[1])
            # the mask axis is the direction of travel, plus noise. it is an
            # undirected line, so it is stored folded into [-pi/2, pi/2].
            a = direction + np.radians(rng.normal(0, cfg.axis_noise_deg))
            theta.append((a + np.pi / 2) % np.pi - np.pi / 2)
            major.append(mj[k]); minor.append(mn[k]); area.append(ar[k])
            score.append(np.clip(rng.normal(cfg.score_mean, cfg.score_sd), 0.05, 1.0))
            source.append(cid)
            pos = pos + step

    # ── static clutter: persists and jiggles. this is the dangerous kind ──────
    for _ in range(cfg.clutter_static):
        home = rng.uniform(20, cfg.size - 20, 2)
        th = rng.uniform(-np.pi / 2, np.pi / 2)
        mj, mn, ar = _shape(rng, cfg, cfg.n_frames)
        for t in range(cfg.n_frames):
            if rng.random() < cfg.miss_rate:
                continue
            p = home + rng.normal(0, cfg.static_jitter, 2)
            frame.append(t); cy.append(p[0]); cx.append(p[1])
            theta.append(th + rng.normal(0, 0.2))
            major.append(mj[t]); minor.append(mn[t]); area.append(ar[t])
            score.append(np.clip(rng.normal(cfg.clutter_score_mean, cfg.score_sd), 0.05, 1.0))
            source.append(-1)

    # ── transient clutter: one frame each ────────────────────────────────────
    n_tr = cfg.clutter_transient * cfg.n_frames
    if n_tr:
        mj, mn, ar = _shape(rng, cfg, n_tr)
        frame.extend(rng.integers(0, cfg.n_frames, n_tr).tolist())
        cy.extend(rng.uniform(0, cfg.size, n_tr).tolist())
        cx.extend(rng.uniform(0, cfg.size, n_tr).tolist())
        theta.extend(rng.uniform(-np.pi / 2, np.pi / 2, n_tr).tolist())
        major.extend(mj.tolist()); minor.extend(mn.tolist()); area.extend(ar.tolist())
        score.extend(np.clip(rng.normal(cfg.clutter_score_mean, cfg.score_sd, n_tr),
                             0.05, 1.0).tolist())
        source.extend([-1] * n_tr)

    order = np.argsort(np.asarray(frame), kind="stable")
    major = np.asarray(major)[order]
    minor = np.asarray(minor)[order]
    area = np.asarray(area, dtype=np.int64)[order]
    dets = DetectionTable(
        frame=np.asarray(frame, np.int64)[order],
        cy=np.asarray(cy)[order], cx=np.asarray(cx)[order],
        theta=np.asarray(theta)[order],
        sigma_theta=np.array([orientation_sigma(a, b, c)
                              for a, b, c in zip(major, minor, area)]),
        major=major, minor=minor, area=area,
        score=np.asarray(score)[order],
        det_id=np.arange(len(order), dtype=np.int64),
    )
    return dets, Truth(np.asarray(source, np.int64)[order], cfg.n_comets, truth_pos)


def score(segments, dets: DetectionTable, truth: Truth,
          purity_threshold: float = 0.75) -> dict:
    """Score output tracks against known provenance.

    A track is REAL if at least ``purity_threshold`` of its detections come from
    one comet. Anything else is spurious -- either stitched from clutter or
    welded across two comets, which are both wrong in the same way.
    """
    src = truth.source
    n_real = 0
    purities, claimed, speed_err = [], {}, []
    for tr in segments:
        s = src[np.asarray(tr.det_index)]
        vals, counts = np.unique(s[s >= 0], return_counts=True)
        if vals.size == 0:
            purities.append(0.0)
            continue
        best = int(vals[counts.argmax()])
        pur = counts.max() / len(s)
        purities.append(float(pur))
        if pur >= purity_threshold:
            n_real += 1
            claimed.setdefault(best, []).append(tr)
            tp = truth.positions.get(best, {})
            common = [f for f in tr.frames if f in tp]
            if len(common) >= 2:
                gt = np.array([tp[f] for f in common])
                gt_v = np.hypot(*(gt[-1] - gt[0])) / (common[-1] - common[0])
                p = tr.positions()
                idx = [tr.frames.index(f) for f in common]
                got_v = np.hypot(*(p[idx[-1]] - p[idx[0]])) / (common[-1] - common[0])
                speed_err.append(got_v - gt_v)

    n_out = len(segments)
    return {
        "n_tracks": n_out,
        "n_real_tracks": n_real,
        "spurious_frac": 1.0 - (n_real / n_out) if n_out else 0.0,
        "mean_purity": float(np.mean(purities)) if purities else 0.0,
        "comets_recovered": len(claimed),
        "comets_total": len({c for c in np.unique(src) if c >= 0}),
        "fragments_per_comet": (n_real / len(claimed)) if claimed else 0.0,
        "speed_bias": float(np.mean(speed_err)) if speed_err else float("nan"),
        "speed_rmse": float(np.sqrt(np.mean(np.square(speed_err)))) if speed_err else float("nan"),
    }


def run_benchmark(config=None, bench: BenchConfig | None = None,
                  seeds=(0, 1, 2)) -> dict:
    """Mean of ``score`` over several seeds."""
    from .link import run_linking

    base = bench or BenchConfig()
    rows = []
    for s in seeds:
        cfg = BenchConfig(**{**base.__dict__, "seed": s})
        dets, truth = make_movie(cfg)
        rows.append(score(run_linking(dets, config), dets, truth))
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}
