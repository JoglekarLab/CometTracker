"""Head (plus-end) estimation from the causal RGB's own colour.

THE IDEA, AND WHY IT WORKS

The model's input is a causal pseudo-RGB: ``R = I(t-2), G = I(t-1), B = I(t)``.
A comet is 20-30 px long but advances only ~2.4 px per frame, so the three
copies overlap heavily and only the EDGES separate::

    leading ~2.4 px   present at t only          -> blue
    next    ~2.4 px   present at t-1 and t       -> cyan
    the bulk          present in all three       -> white
    trailing ~2.4 px  present at t-2 only        -> red

So the plus-end is the blue/cyan end, and it can be located from ONE causal
window -- no track history, so it exists on a track's first frame.

MEASURED, on 5009 detections across both pAJV103 movies, against the direction
of travel of straight moving tracks:

    intensity-profile skew (V1's method, 1 frame)   75.9% correct
    temporal shift along the axis (3 frames)        89.3% correct
      ... gated to |shift| >= 0.5 px/frame          96.2% correct (73% kept)
      ... gated to |shift| >= 1.0 px/frame          97.7% correct (47% kept)

The colour cue is the stronger of the two, and the skew cue adds nothing on top
of it: where the two disagree, the colour cue is still right 78% of the time.
V8 therefore uses colour and does not use skew.

THE CONFOUND, AND WHY THE WEIGHTING IS SHAPED THE WAY IT IS

EB3 comet brightness tracks growth rate, so a comet that brightens across the
window is blue-shifted everywhere and one that dims is red-shifted everywhere,
with no motion involved. Both evidence maps are therefore DIFFERENCES between
channels, not ratios or absolute levels: a uniform scaling of all three channels
leaves ``B - max(R, G)`` positive only where B genuinely exceeds the earlier
frames, which is the newly occupied region.

THE CONFOUND IS REDUCED, NOT ELIMINATED

A stationary comet that brightens across the window still yields positive blue
evidence everywhere, and the estimator then degrades to an intensity-weighted
centroid -- a head displaced forward of the mask centroid with no motion behind
it. Measured on synthetic scenes it lands short of the true tip and carries less
evidence than real motion does, so it is a smaller error than it could be.

``evidence`` does NOT reliably separate the two on real data: 0.309 median for
directed tracks against 0.243 for immobile ones, heavily overlapping. Treat it
as a quality signal, not as a filter for this confound. The evidence that the
estimator finds real heads is empirical rather than mechanical: against 11
hand-labelled heads it lands 1.77 px out (median) where the mask centroid lands
4.34 px out.

This is a port of the estimator developed in ``TrajectoryAxisLabeling/`` --
``evidence_maps`` and ``current_blue_centroid_weight`` are that code's
arithmetic, unchanged. What differs is the search region: the original took a
click and searched a radius-18 disc, while V8 searches inside the comet's own
SAM3 mask (dilated, because the leading edge is exactly the part a p>=0.5
contour tends to clip).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, label

__all__ = ["HeadEstimate", "evidence_maps", "blue_weight", "head_from_rgb"]

SMOOTH_SIGMA = 0.8      # as developed in TrajectoryAxisLabeling
MIN_EVIDENCE = 0.025    # below this there is no usable colour signal at all
REL_THRESHOLD = 0.42    # fraction of the local max that defines the region
MIN_HEAD_PIXELS = 7     # MIN_HEAD_PIXELS in the original


@dataclass
class HeadEstimate:
    """``head`` is (row, col), or None when no region cleared the threshold."""
    head: np.ndarray | None
    area: int
    evidence: float          # peak smoothed evidence in the search region
    purity: float            # mean blue fraction over the chosen region
    region: np.ndarray | None = None

    @property
    def ok(self) -> bool:
        return self.head is not None


def evidence_maps(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(blue, cyan) evidence. Verbatim arithmetic from the labelling tool.

    blue = B - max(R, G)      newly occupied at t, absent at t-1 and t-2
    cyan = min(G, B) - R      occupied at t-1 and t, absent at t-2
    """
    red, green, blue = np.moveaxis(np.asarray(rgb, np.float32), -1, 0)
    return (np.clip(blue - np.maximum(red, green), 0, None),
            np.clip(np.minimum(green, blue) - red, 0, None))


def blue_weight(rgb: np.ndarray) -> np.ndarray:
    """Centroid weight ``B * B/(R+G+B)``.

    Two factors doing two jobs: the leading ``B`` favours bright current-frame
    pixels, and the blue fraction favours pixels that are PURELY new. Pure blue
    gets full weight; an equal-green cyan pixel gets about half, so it still
    pulls the centroid but less than a pixel the comet only just reached.
    """
    red, green, blue = np.moveaxis(np.asarray(rgb, np.float32), -1, 0)
    return blue * (blue / (red + green + blue + 1e-6))


def head_from_rgb(rgb: np.ndarray, mask: np.ndarray,
                  centroid: tuple[float, float] | None = None,
                  dilate: int = 3) -> HeadEstimate:
    """Locate the plus-end inside one comet's mask.

    Parameters
    ----------
    rgb      : (H, W, 3) causal pseudo-RGB for this frame, channels in [0, 1].
    mask     : (H, W) bool, this comet's mask.
    centroid : tie-break centre; the mask's own centroid if omitted.
    dilate   : how far to grow the mask before searching. The leading edge is
               the newest and faintest part of the comet, so it is exactly what
               a p >= 0.5 contour clips -- searching only inside the mask would
               systematically pull the head backwards.
    """
    rgb = np.asarray(rgb, np.float32)
    mask = np.asarray(mask, bool)
    if not mask.any():
        return HeadEstimate(None, 0, 0.0, 0.0)

    blue_score, cyan_score = evidence_maps(rgb)
    raw = np.maximum(blue_score, cyan_score)
    smooth = gaussian_filter(raw, SMOOTH_SIGMA)

    search = binary_dilation(mask, iterations=int(dilate)) if dilate > 0 else mask
    local_max = float(smooth[search].max())
    if local_max < MIN_EVIDENCE:
        return HeadEstimate(None, 0, local_max, 0.0)

    threshold = max(MIN_EVIDENCE, REL_THRESHOLD * local_max)
    regions, count = label((smooth >= threshold) & search, np.ones((3, 3), int))
    if count == 0:
        return HeadEstimate(None, 0, local_max, 0.0)

    if centroid is None:
        ys, xs = np.nonzero(mask)
        centroid = (ys.mean(), xs.mean())
    cy, cx = float(centroid[0]), float(centroid[1])

    weight_map = blue_weight(rgb)
    yy, xx = np.indices(smooth.shape)
    red, green, blue = np.moveaxis(rgb, -1, 0)

    best = None
    for rid in range(1, count + 1):
        region = regions == rid
        area = int(region.sum())
        if area < MIN_HEAD_PIXELS:
            continue
        # position from the blue-affinity weight; SELECTION from the raw
        # evidence, so a region is chosen for how much colour evidence it holds
        # and only then measured with the purity weighting.
        w = weight_map * region
        if w.sum() <= 0:
            w = smooth * region
        if w.sum() <= 0:
            continue
        ry = float((yy * w).sum() / w.sum())
        rx = float((xx * w).sum() / w.sum())

        sel = raw * region
        sel_total = float(sel.sum())
        if sel_total <= 0:
            sel, sel_total = smooth * region, float((smooth * region).sum())
        if sel_total <= 0:
            continue
        sy = float((yy * sel).sum() / sel_total)
        sx = float((xx * sel).sum() / sel_total)
        # closer regions preferred, gently -- the original's 0.03/px
        rank = sel_total / (1.0 + 0.03 * float(np.hypot(sy - cy, sx - cx)))

        denom = (red + green + blue + 1e-6)[region]
        purity = float(np.mean(blue[region] / denom))
        if best is None or rank > best[0]:
            best = (rank, np.array([ry, rx], np.float32), area, purity, region)

    if best is None:
        return HeadEstimate(None, 0, local_max, 0.0)
    _, head, area, purity, region = best
    return HeadEstimate(head, area, local_max, purity, region)


# ══════════════════════════════════════════════════════════════════════════════
#  Post-linking pass: heads along already-linked tracks
# ══════════════════════════════════════════════════════════════════════════════
#
# THIS RUNS AFTER LINKING AND CANNOT CHANGE IT.
#
# Linking stays on the mask centroid, which is measurably the steadier
# measurement: RMS residual around a constant-velocity fit is 0.47 px for the
# centroid against 1.04 px for the head. Feeding the LAP the noisier of the two
# would degrade the thing V7 was tuned around for no benefit.
#
# But the centroid is the steadier estimate of the WRONG point. Against
# hand-labelled heads it lands 4.34 px out (median) where this estimator lands
# 1.77 px out, and its error is not random: mask length drifts a median 6.8 px
# over a track, so the centroid accumulates a systematic displacement pointing
# the same way every frame. The head's error is per-frame noise.
#
# That difference is the whole argument for this pass. For a TRACK-LEVEL
# velocity, independent noise averages down under a least-squares fit and bias
# does not. So: link on the centroid, measure speed on the head.

MAX_HEAD_STEP = 7.0
"""px per frame. Above this a head step is not a comet.

EB3 comets run 1.3-6.3 px/frame (the range recorded in sam3_tracks.py), so 7
sits just above the physical maximum. Measured head steps: median 2.75, p90
7.09, p99 18.22, max 47.7 px/frame -- that tail is the estimator jumping onto a
neighbouring comet's leading edge inside the dilated search region, not growth.
The gate discards 10.2% of head steps. For comparison the centroid, which the
linker gates at max_disp, exceeds 7 px/frame on 0.23% of steps.
"""

MAX_HEAD_BACKWARD = None
"""px per frame a head may move AGAINST its track's direction. OFF by default.

An earlier version defaulted this to 2.0 px/frame, on the argument that EB3
binds only the growing plus-end so a shrinking microtubule loses its comet and
backward motion must be an error. That argument was too strong. EB3 dissociates
with a time constant of roughly 2.5-4 s, so at 2 s/frame the comet stays visible
for about one to two frames after growth stops -- a brief backward excursion is
observable, and a gate would delete exactly that signal before it could be
measured.

It is off for two further reasons, both measured:

  * it is not needed. Backward steps beyond 2 sigma occur on 11.5% and 12.3% of
    directed tracks, against 11.0% expected from step noise alone -- no excess.
    Sustained backward runs (>1 frame) occur on 0.6-1.0% of tracks, and the mask
    centroid corroborates NONE of them.
  * it is asymmetric. Rejecting only backward outliers while keeping forward
    ones biases speed upward, and the gate moved the median head speed by under
    0.05 um/min anyway -- so it costs direction-neutrality and buys nothing.

The magnitude gate (MAX_HEAD_STEP) stands on its own: 47 px/frame is not a
comet under any interpretation. Set this to a number to re-enable direction
gating, but not for anything that measures backward motion.
"""


@dataclass
class TrackHeads:
    """Per-frame head estimates along one already-linked track."""
    frames: list[int]
    heads: np.ndarray          # (n, 2) rows of (row, col); nan where unavailable
    offsets: np.ndarray        # signed head-minus-centroid along the track axis
    n_missing: int
    centroids: np.ndarray | None = None   # the linker's positions, same order
    direction: np.ndarray | None = None   # unit vector of the track's travel
    n_rejected: int = 0

    @property
    def valid(self) -> np.ndarray:
        return ~np.isnan(self.heads[:, 0])

    def gate(self, max_step: float = MAX_HEAD_STEP,
             max_backward: float | None = MAX_HEAD_BACKWARD) -> "TrackHeads":
        """Drop head measurements that cannot be growth. Returns a new object.

        Two rejections, both against the CENTROID track rather than against the
        neighbouring head. That matters: if a head is compared only to its
        neighbour, one bad estimate poisons the two steps either side of it and
        there is no way to tell which of the pair was wrong. The centroid track
        is independent of the head estimator and is already gated by the linker,
        so it is the stable reference.

        1. step magnitude -- the head's displacement per frame, relative to the
           centroid's own displacement over the same interval, must not exceed
           ``max_step``;
        2. direction -- a head step may not run more than ``max_backward``
           against the track's direction of travel.
        """
        out = TrackHeads(list(self.frames), self.heads.copy(),
                         self.offsets.copy(), self.n_missing,
                         self.centroids, self.direction, self.n_rejected)
        idx = np.flatnonzero(out.valid)
        if idx.size < 2 or out.direction is None:
            return out
        f = np.asarray(out.frames, float)
        rejected = 0
        for a, b in zip(idx[:-1], idx[1:]):
            dt = max(f[b] - f[a], 1.0)
            step = out.heads[b] - out.heads[a]
            per_frame = float(np.hypot(*step)) / dt
            along = float(step @ out.direction) / dt
            too_far = per_frame > max_step
            too_backward = max_backward is not None and along < -max_backward
            if too_far or too_backward:
                # drop whichever endpoint sits further from where the centroid
                # track says the head should be
                ref = out.centroids
                da = np.hypot(*(out.heads[a] - ref[a])) if ref is not None else 0.0
                db = np.hypot(*(out.heads[b] - ref[b])) if ref is not None else 1.0
                drop = a if da > db else b
                out.heads[drop] = np.nan
                out.offsets[drop] = np.nan
                rejected += 1
        out.n_rejected = self.n_rejected + rejected
        return out

    def velocity_px_per_frame(self) -> float:
        """Speed from a least-squares fit of head position against frame.

        A fit rather than first-to-last: the head is a per-frame estimate with
        ~1 px of independent noise, and a fit over n points suppresses that by
        roughly sqrt(n) while two endpoints do not suppress it at all.
        """
        m = self.valid
        if m.sum() < 2:
            return float("nan")
        f = np.asarray(self.frames, float)[m]
        if f.max() - f.min() < 1:
            return float("nan")
        p = self.heads[m]
        vy = np.polyfit(f, p[:, 0], 1)[0]
        vx = np.polyfit(f, p[:, 1], 1)[0]
        return float(np.hypot(vy, vx))

    def net_displacement_px(self) -> float:
        m = self.valid
        if m.sum() < 2:
            return float("nan")
        p = self.heads[m]
        return float(np.hypot(*(p[-1] - p[0])))


def heads_along_track(track, dets, rgb_by_frame, labels,
                      dilate: int = 3) -> TrackHeads:
    """Estimate the head at every frame of one linked track.

    ``rgb_by_frame`` maps frame index -> the causal pseudo-RGB for that frame.
    Frames whose mask is missing, or where no colour region clears the
    threshold, yield nan rather than a guess -- roughly 1.2% of detections on
    the real movies. A guessed head would quietly corrupt the velocity fit.
    """
    pos = track.positions()
    travel = pos[-1] - pos[0]
    norm = float(np.hypot(*travel))
    u = travel / norm if norm > 0 else np.zeros(2)

    heads = np.full((len(track.frames), 2), np.nan)
    offsets = np.full(len(track.frames), np.nan)
    centroids = np.full((len(track.frames), 2), np.nan)
    missing = 0
    for k, (f, di) in enumerate(zip(track.frames, track.det_index)):
        rgb = rgb_by_frame.get(int(f))
        if rgb is None:
            missing += 1
            continue
        cy, cx = float(dets.cy[di]), float(dets.cx[di])
        centroids[k] = (cy, cx)
        yi, xi = int(round(cy)), int(round(cx))
        lab = labels[int(f)]
        if not (0 <= yi < lab.shape[0] and 0 <= xi < lab.shape[1]):
            missing += 1
            continue
        cid = lab[yi, xi]
        if cid == 0:
            missing += 1
            continue
        est = head_from_rgb(rgb, lab == cid, centroid=(cy, cx), dilate=dilate)
        if not est.ok:
            missing += 1
            continue
        heads[k] = est.head
        offsets[k] = float((est.head - np.array([cy, cx])) @ u)
    return TrackHeads(list(track.frames), heads, offsets, missing,
                      centroids=centroids, direction=u)
