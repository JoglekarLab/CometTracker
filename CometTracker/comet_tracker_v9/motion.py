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
    it, no MSS class is claimed and a short-track criterion is used, labelled as
    such in the output so the two are never silently mixed.

THREE SHORT-TRACK CRITERIA, AND WHY THERE ARE THREE
    ``min_frames_for_mss`` is 20 and real lifetimes are 3-25 frames, so the
    short-track criterion decides the class of nearly every track this project
    produces. It is therefore not a detail. Three are implemented and the
    configured one sets ``motion_class``; ``classify_every_criterion`` runs all
    three so their disagreement can be measured rather than assumed.

    ``net_disp`` -- the original, kept so published numbers stay reproducible.
        Immobile below a fixed 5 px of net displacement, then straightness.
        This is a SPEED floor scaling as 1/length: a 3-frame track must cover
        2.5 px/frame to qualify, a 9-frame track only 0.63. Measured medians
        for the directed class, both movies: 16.95/16.84 um/min at 3 frames,
        9.06/9.22 at 5, 10.44/6.89 at 9+. The short-track figure is the
        threshold showing through, not biology.

    ``speed_floor`` -- the minimal repair. Immobile below a fixed px/FRAME,
        then the same straightness test. Removes the 1/length scaling and
        changes nothing else, so it isolates that one defect.

    ``diffusion`` -- the criterion that asks the actual question. A diffusing
        object's net displacement grows as sqrt(time); a growing microtubule's
        grows linearly. So compare the track's net displacement against what
        its OWN step size would produce under a random walk::

            N  = n - 1 steps
            D  = net / (rms_step * sqrt(N))

        D has RMS 1 for a 2D random walk whatever the step size, and equals
        sqrt(N) for a perfectly straight track. Under the random-walk null D is
        Rayleigh with P(D > k) = exp(-k^2), so ``diffusion_k`` is a stated
        false-positive rate: 1.73 -> 5%, 2.15 -> 1%, 2.63 -> 0.1%. No global
        speed or displacement constant is needed, which is the point -- the
        track is measured against itself.

        NOTE THE CONSEQUENCE. Max D is sqrt(N), so at k = 1.73 a 3-frame track
        (N = 2, max D = 1.414) can never be called directed however straight it
        looks. That is correct, not a defect: two steps cannot separate a line
        from a random walk at 5% confidence, and the old criterion's answer for
        those tracks was an artefact of the threshold. It does mean this
        criterion reclassifies the entire 3-frame population, so the class
        counts are not comparable to the other two without saying so.
"""
from __future__ import annotations

import numpy as np

from .config import Config, MotionConfig

__all__ = ["mss_slope", "classify", "classify_all", "classify_every_criterion",
           "diffusion_ratio", "CRITERIA",
           "IMMOBILE", "CONFINED", "BROWNIAN", "DIRECTED", "UNCLASSIFIED"]

IMMOBILE = "immobile"
CONFINED = "confined"
BROWNIAN = "brownian"
DIRECTED = "directed"
UNCLASSIFIED = "unclassified"

CRITERIA = ("net_disp", "speed_floor", "diffusion")


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


# --- track geometry, computed once and shared by all three criteria ---------
def _geometry(p: np.ndarray) -> dict:
    n = len(p)
    if n < 2:
        return dict(n=n, net=0.0, path=0.0, rms_step=0.0, straight=0.0, D=0.0)
    steps = np.diff(p, axis=0)
    step_len = np.hypot(steps[:, 0], steps[:, 1])
    net = float(np.hypot(*(p[-1] - p[0])))
    path = float(step_len.sum())
    rms = float(np.sqrt(np.mean(step_len ** 2)))
    N = n - 1
    return dict(
        n=n, net=net, path=path, rms_step=rms,
        straight=(net / path) if path > 0 else 0.0,
        D=(net / (rms * np.sqrt(N))) if rms > 0 else 0.0,
    )


def diffusion_ratio(positions) -> float:
    """D = net / (rms_step * sqrt(N)).

    1 for a random walk, sqrt(N) for a straight line, independent of step size.
    """
    return _geometry(np.asarray(positions, np.float64))["D"]


# --- the three short-track criteria ----------------------------------------
def _crit_net_disp(g, cfg) -> str:
    if g["net"] < cfg.min_net_displacement:
        return IMMOBILE
    return DIRECTED if g["straight"] >= cfg.straightness_directed else BROWNIAN


def _crit_speed_floor(g, cfg) -> str:
    per_frame = g["net"] / max(g["n"] - 1, 1)
    if per_frame < cfg.min_speed_px_per_frame:
        return IMMOBILE
    return DIRECTED if g["straight"] >= cfg.straightness_directed else BROWNIAN


def _crit_diffusion(g, cfg) -> str:
    if g["rms_step"] < cfg.immobile_step_px:
        return IMMOBILE
    return DIRECTED if g["D"] >= cfg.diffusion_k else BROWNIAN


_CRIT_FN = {"net_disp": _crit_net_disp,
            "speed_floor": _crit_speed_floor,
            "diffusion": _crit_diffusion}


def _mss_class(p, cfg) -> tuple[str, float] | None:
    if len(p) < cfg.min_frames_for_mss:
        return None
    s = mss_slope(p)
    if not np.isfinite(s):
        return None
    if s < 0.15:
        return IMMOBILE, s
    if s < 0.4:
        return CONFINED, s
    if s <= 0.6:
        return BROWNIAN, s
    return DIRECTED, s


def classify(positions: np.ndarray, config: MotionConfig | None = None,
             criterion: str | None = None) -> tuple[str, float, str]:
    """Return (class, mss_slope, method).

    ``method`` is "mss" or the name of the short-track criterion used, and must
    be reported alongside the class: they are not the same measurement and
    should never be pooled without saying which produced which.
    """
    cfg = config or MotionConfig()
    crit = criterion or cfg.criterion
    if crit not in _CRIT_FN:
        raise ValueError(f"unknown motion criterion {crit!r}; "
                         f"expected one of {CRITERIA}")
    p = np.asarray(positions, np.float64)

    hit = _mss_class(p, cfg)
    if hit is not None:
        return hit[0], hit[1], "mss"
    return _CRIT_FN[crit](_geometry(p), cfg), float("nan"), crit


def classify_every_criterion(positions, config: MotionConfig | None = None
                             ) -> dict:
    """Class under all three criteria, plus the geometry behind them.

    One pass, so a comparison never costs a second run of the tracker. Tracks
    long enough for MSS get the same class in every entry -- the criteria only
    ever decide short tracks -- which is exactly what makes the comparison fair.
    """
    cfg = config or MotionConfig()
    p = np.asarray(positions, np.float64)
    g = _geometry(p)
    hit = _mss_class(p, cfg)
    if hit is not None:
        classes = {c: hit[0] for c in CRITERIA}
        method, slope = "mss", hit[1]
    else:
        classes = {c: _CRIT_FN[c](g, cfg) for c in CRITERIA}
        method, slope = "short", float("nan")
    return dict(classes=classes, method=method, mss_slope=slope, **g)


def classify_all(compounds, config: Config | None = None) -> None:
    """Classify each compound track in place.

    The whole compound is classified, not each segment: a microtubule that grows,
    pauses and grows again is one directed object, and scoring its pieces
    separately would call the short ones confined.

    Sets ``motion_class`` from the configured criterion, and ``motion_classes``
    with all three so downstream code can compare without re-running anything.
    """
    cfg = (config or Config()).motion
    for c in compounds:
        pos = np.vstack([s.positions() for s in c.segments])
        info = classify_every_criterion(pos, cfg)
        c.motion_class = info["classes"][cfg.criterion]
        c.mss_slope = info["mss_slope"]
        setattr(c, "motion_classes", info["classes"])
        setattr(c, "motion_method",
                "mss" if info["method"] == "mss" else cfg.criterion)
        setattr(c, "diffusion_ratio", info["D"])
        setattr(c, "straightness", info["straight"])
        setattr(c, "rms_step_px", info["rms_step"])
