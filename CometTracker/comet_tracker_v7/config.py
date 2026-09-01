"""Every tunable number, with where it came from.

Three provenances appear below and they are NOT equally trustworthy:

  [plusTip]  the default in u-track's own source, read out of
             ``@TrackingProcess/TrackingProcess.m`` lines 538-613. Tuned by its
             authors on their data, not ours.
  [V6]       measured on this project's synthetic benchmark, recorded in
             ``V6_CHANGES.md``. Real measurements, but of the LINKER only, with
             an estimated clutter density.
  [V7]       measured during V7's design on real SAM3 predictions for
             pAJV103_010 and _015 -- 247 tracks that pass a straightness filter.
             Real data, but the track set was built by ``sam3_tracks.py`` and is
             roughly a third static clutter, so these describe the population
             we can currently see, not verified ground truth.

  [guess]    nothing measured it. Flagged so it is obvious what to sweep first.

Nothing here has been scored end to end yet. Treat the defaults as a starting
point for a sweep, not as findings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ── acquisition, from HANDOFF.md ──────────────────────────────────────────────
MICRONS_PER_PIXEL = 0.158875614664932
SECONDS_PER_FRAME = 2.0


def px_per_frame_to_um_per_min(v: float) -> float:
    return v * MICRONS_PER_PIXEL * (60.0 / SECONDS_PER_FRAME)


@dataclass
class DetectConfig:
    """Turning masks into the instance table."""

    min_area: int = 6
    """[plusTip-independent] matches sam3_export's --min-area so V7 sees the
    same components the other tools do."""

    min_score: float = 0.0
    """[V7] presence floor. Deliberately 0, for two measured reasons:

    1. presence and mask quality are uncorrelated (r = 0.025 and 0.051 on the
       two movies), so a presence threshold discards good geometry;
    2. presence barely separates real comets from clutter (AUC 0.608), so the
       threshold does not buy precision either -- benchmarked, a cut at 0.7
       changes nothing and a cut at 0.9 costs 60% of the comets.

    Presence is carried into the cost matrix instead. Raising this floor cannot
    help and can only lose detections."""


@dataclass
class LinkConfig:
    """Stage 1: frame to frame."""

    max_disp: float = 7.0
    """[V6] the measured knee. V6 swept 5/7/9/12 and found 5 costs a quarter of
    the real comets (39.0 vs 48.3 recovered) for 3 points of precision, while 9
    doubles the spurious rate. plusTip's own maxSearchRadius is 10."""

    corridor_min: float = 2.5
    """[V7] floor on the across-axis corridor, in px. Measured |q| for a true
    single-frame step: median 0.17, p90 0.52, p99 2.16. 2.5 clears p99.
    V6 needed 4.92 because its axis came from a Gaussian fit on a probability
    map; a SAM3 mask axis is far better (5.4 deg median from travel), which is
    what buys the tighter gate."""

    corridor_sigma_mult: float = 3.0
    """[guess] the corridor widens by this many sigma_theta-induced lateral
    errors: corridor = corridor_min + mult * |displacement| * sigma_theta.
    Rationale: an axis angle error sigma over a step d puts the true position
    d*sin(sigma) ~ d*sigma off the line. This is what makes round masks (large
    sigma_theta) gate loosely instead of being silently trusted."""

    max_link_angle_deg: float = 30.0
    """[plusTip] maxFAngle. Only applied once a track has a velocity; a track's
    first link is gated by the corridor alone, which is the whole point of using
    the mask's own axis (it exists at birth)."""

    score_weight: float = 4.0
    """[V7] link cost += score_weight * (1 - presence), in px^2.

    MEASURED TO BE NEARLY INERT, AND KEPT ANYWAY. On real SAM3 predictions the
    score of a detection inside a straight moving track (median 0.922) is barely
    above one inside a track that goes nowhere (0.898): AUC 0.608 against 0.5
    for no information at all. Benchmarked at the measured distributions,
    presence-as-cost (20.9% spurious), presence ignored (20.3%) and a hard
    threshold at 0.7 (20.3%) are indistinguishable.

    So this parameter does not currently buy anything, and the design argument
    for it -- 'a weak detection where a strong track predicts should link, the
    same detection alone should not' -- is correct in structure but has nothing
    to act on, because SAM3's presence does not identify clutter. It is kept
    because it is the right place for the information IF a detector ever
    produces a confidence that separates the two, and because at this weight it
    costs nothing. Do not cite it as a feature that works."""

    birth_cost: float = 16.0
    """[guess] px^2. Cost of a detection starting a new track. Must exceed a
    typical good link cost or nothing ever links; 16 = (4 px)^2."""

    birth_score_weight: float = 48.0
    """[V7] birth cost += this * (1 - presence).

    The link/birth asymmetry a LAP can express and a threshold cannot. Swept at
    0 / 12 / 48 / 120 the spurious rate moves 20.5 / 20.2 / 21.0 / 21.2% -- i.e.
    not at all, for the same reason as score_weight: SAM3's presence does not
    separate comets from clutter. The mechanism is sound; the input is not
    informative. Left in place, honestly labelled, so that a better detector
    can be plugged in without redesigning the cost matrix."""

    death_cost: float = 16.0
    """[guess] px^2. Cost of a track ending. Symmetric with birth_cost."""

    max_gap: int = 0
    """[V6] NO COASTING. V6's single largest effect: share of links correct by
    gap was 37.7% at gap 1, 7.6% at gap 2, 2.3% at gap 3. A coasting track
    guesses, and in a dense field re-acquires clutter. Fragments are gap
    closing's problem, not the frame linker's."""

    n_passes: int = 1
    """[V6] forward / backward / forward is what u-track does
    (trackCloseGapsKalmanSparse.m lines 352-402), because a new track's initial
    velocity is zero (plusTipKalmanInitLinearMotion: "if not supplied, then
    initial velocity is taken as zero") so pass 1 is blind at every track start.
    V6 measured 1/3/5 passes at recall 53.5/54.6/53.4%, and identical to three
    decimals with coasting off. V7 should need it even less: the mask axis
    supplies direction at frame one, which is the thing the extra passes exist
    to bootstrap. Set to 3 to reproduce u-track. Must be odd."""

    process_noise: float = 1.0
    """[guess] Kalman Q scale."""

    measurement_noise: float = 0.4
    """[V7] Kalman R scale, px^2. Measured RMS residual of the mask centroid
    around a constant-velocity fit is 0.61 px on clean tracks, so ~0.37 px^2.
    Note this is centroid noise only; it does NOT include the ~1.3 px/frame
    wobble in mask LENGTH, which moves the centroid and is why V7 does not
    reconstruct a tip from centroid + half-length (that estimator measured 2.61
    px RMS, four times worse than the centroid itself)."""

    min_track_length: int = 3
    """[plusTip] minTrackLen, the newer u-track default."""


@dataclass
class GapConfig:
    """Stage 2: closing gaps, and calling each one a pause or a shrinkage."""

    time_window: int = 5
    """[plusTip] gapCloseParam.timeWindow."""

    max_f_angle_deg: float = 30.0
    """[plusTip] maxFAngle. Forward cone: angle between the ending track's final
    velocity and the displacement to the candidate start."""

    max_b_angle_deg: float = 10.0
    """[plusTip] maxBAngle. Backward wedge, deliberately much tighter -- a
    shrinking microtubule retreats along the lattice it just built, so a
    backward link that is off-axis is not a shrinkage."""

    fluct_rad: float = 1.0
    """[plusTip] tube radius around the trajectory, px."""

    back_vel_mult: float = 1.5
    """[plusTip] backVelMultFactor. Shrinkage runs faster than growth, so the
    backward search reaches further than the forward one."""

    fwd_vel_mult: float = 1.5
    """[V7] forward reach = fwd_vel_mult * speed * dt + fluct_rad.

    plusTip has no such parameter because its forward search radius comes from
    the Kalman filter's own predicted displacement std. V7 needs it because the
    reach is measured from the segment's LAST OBSERVED position using its OWN
    median speed, and a gap caused by a detector MISS (rather than a biological
    pause) puts the comet at almost exactly speed*dt ahead -- i.e. right on the
    boundary of a reach of speed*dt + fluct_rad, so any speed variation pushes a
    perfectly good join outside the gate.

    Diagnosed on real data: of 8333 candidate pairs that agreed in direction,
    only 288 survived a reach of 1.0, which was throwing away most of gap
    closing's job. Set to 1.0 to reproduce that behaviour."""

    break_nonlinear: bool = False
    """[plusTip] breakNonLinearTracks. plusTipBreakNonlinearTracks splits a
    track wherever consecutive steps turn by more than 45 deg, exempting steps
    below the 3rd percentile as localisation noise. Off by default, as in
    plusTip."""

    break_angle_deg: float = 45.0
    """[plusTip] the 45 deg in plusTipBreakNonlinearTracks."""


@dataclass
class MotionConfig:
    """Telling a growing comet from a blob that jiggles."""

    mss_alpha: float = 0.1
    """[plusTip] trackMSSAnalysis alphaMSS default."""

    min_frames_for_mss: int = 20
    """[guess] MSS needs a decent number of points. Your tracks are 3-25 frames
    (HANDOFF), so most will fall short and be reported unclassified. A
    straightness fallback is applied for those -- see motion.py."""

    straightness_directed: float = 0.6
    """[V7] net displacement / path length above which a short track is called
    directed. 0.6 is the cut used in V7's own design measurements; on the
    current SAM3 track set it keeps 247 of 795 tracks."""

    min_net_displacement: float = 5.0
    """[V7] px. On the current SAM3 track set 34% of tracks move under 3 px in
    total, which is not a growing microtubule."""


@dataclass
class Config:
    detect: DetectConfig = field(default_factory=DetectConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    gap: GapConfig = field(default_factory=GapConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
