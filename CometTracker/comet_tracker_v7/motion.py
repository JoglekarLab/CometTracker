"""Telling a growing microtubule from a blob that jiggles.

Two mechanisms, both from u-track, and they answer different questions.

MSS (moment scaling spectrum), ``trackMSSAnalysis.m``
    For a self-similar process the nu-th moment of displacement scales as
    ``<|dr|^nu> ~ dt^(nu*S)``. Fit the scaling power for each moment order,
    then take the slope of those powers against the order. That slope ``S`` is
    the MSS slope::

        S ~ 0     immobile
        S < 0.5   confined
        S ~ 0.5   pure Brownian diffusion
        S > 0.5   directed (Brownian with drift)

    u-track's classes are exactly these four.

WHY THIS MATTERS HERE AND NOT ONLY AT THE END
    On the current real SAM3 track set, 34% of tracks move under 3 px in total
    and 17% have straightness below 0.3. That population is not microtubule
    growth, and it is currently mixed into every velocity number this project
    produces -- which is the most likely explanation for the median step of
    1.25 px/frame (~6 um/min) against the ~11 um/min the measured 2.36 px/frame
    constant implies. Classification separates them WITHOUT needing labels.

THE SIMPLIFICATION, STATED PLAINLY
    u-track classifies by comparing each track's MSS slope against confidence
    intervals from simulated Brownian motion at ``alphaMSS``. This module uses
    fixed thresholds on the slope instead. That is cruder, and it means the
    class boundaries are not calibrated to track length -- a short Brownian
    track can score a high slope by chance. Hence ``min_frames_for_mss``: below
    it, no MSS class is claimed and the straightness fallback is used, labelled
    as such in the output so the two are never silently mixed.
"""
from __future__ import annotations

import numpy as np

from .config import Config, MotionConfig

__all__ = ["mss_slope", "classify", "classify_all",
           "IMMOBILE", "CONFINED", "BROWNIAN", "DIRECTED", "UNCLASSIFIED"]

IMMOBILE = "immobile"
CONFINED = "confined"
BROWNIAN = "brownian"
DIRECTED = "directed"
UNCLASSIFIED = "unclassified"


def mss_slope(positions: np.ndarray, orders=(1, 2, 3, 4, 5, 6)) -> float:
    """MSS slope for a (n, 2) position series sampled at unit intervals.

    Returns nan when the track is too short for the fit to mean anything.
    """
    p = np.asarray(positions, np.float64)
    n = len(p)
    if n < 6:
        return float("nan")
    max_lag = max(2, min(n // 4, 20))
    lags = np.arange(1, max_lag + 1)
    if lags.size < 3:
        return float("nan")

    powers = []
    log_lags = np.log(lags)
    for nu in orders:
        moments = []
        for dt in lags:
            d = p[dt:] - p[:-dt]
            r = np.hypot(d[:, 0], d[:, 1])
            moments.append(np.mean(r ** nu))
        m = np.asarray(moments)
        if np.any(m <= 0):
            return float("nan")
        powers.append(np.polyfit(log_lags, np.log(m), 1)[0])
    return float(np.polyfit(np.asarray(orders, float), np.asarray(powers), 1)[0])


def classify(positions: np.ndarray, config: MotionConfig | None = None
             ) -> tuple[str, float, str]:
    """Return (class, mss_slope, method).

    ``method`` is "mss" or "straightness" and must be reported alongside the
    class: the two are not the same measurement and should never be pooled
    without saying which produced which.
    """
    cfg = config or MotionConfig()
    p = np.asarray(positions, np.float64)
    n = len(p)
    net = float(np.hypot(*(p[-1] - p[0]))) if n > 1 else 0.0
    path = float(np.hypot(*np.diff(p, axis=0).T).sum()) if n > 1 else 0.0

    if n >= cfg.min_frames_for_mss:
        s = mss_slope(p)
        if np.isfinite(s):
            if s < 0.15:
                return IMMOBILE, s, "mss"
            if s < 0.4:
                return CONFINED, s, "mss"
            if s <= 0.6:
                return BROWNIAN, s, "mss"
            return DIRECTED, s, "mss"

    # fallback for short tracks -- most of them, since lifetimes are 3-25 frames
    if net < cfg.min_net_displacement:
        return IMMOBILE, float("nan"), "straightness"
    straight = net / path if path > 0 else 0.0
    if straight >= cfg.straightness_directed:
        return DIRECTED, float("nan"), "straightness"
    return BROWNIAN, float("nan"), "straightness"


def classify_all(compounds, config: Config | None = None) -> None:
    """Classify each compound track in place.

    The whole compound is classified, not each segment: a microtubule that grows,
    pauses and grows again is one directed object, and scoring its pieces
    separately would call the short ones confined.
    """
    cfg = (config or Config()).motion
    for c in compounds:
        pos = np.vstack([s.positions() for s in c.segments])
        cls, slope, method = classify(pos, cfg)
        c.motion_class = cls
        c.mss_slope = slope
        if not hasattr(c, "motion_method"):
            pass
        setattr(c, "motion_method", method)
