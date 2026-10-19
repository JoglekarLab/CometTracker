"""Comet splitting: one mask becoming two, and whether that means anything.

    from .splits import detect
    out = detect(labels, movie)          # movie optional, enables the photometry test

WHAT A SPLIT EVENT IS HERE

A connected component at frame t whose pixels overlap TWO OR MORE separate
components at t+1. Nothing about tracking is used -- this is read straight off
the label stack, so it is independent of the linker, of max_disp, and of every
gate in link.py. That matters because the tracker has no split model at all: a
split currently shows up as one track dying and another being born, which is
indistinguishable from ordinary fragmentation once the tracks are written.

THE PROBLEM, AND THE THREE DISCRIMINATORS

Finding candidates is easy and almost meaningless on its own. A segmentation
model that breaks one comet's mask into two components produces exactly this
signature, and we already know SAM3 does that sometimes. So every candidate
carries three measurements that separate a real branch from a broken mask:

  angle       A broken mask leaves two pieces lying on the SAME axis, because
              they are two halves of one comet. Two microtubules growing apart
              do not. Recorded as the angle between the children's own mask
              axes, and as the angle between the directions they then travel.

  divergence  Two real comets moving apart get further apart every frame. Two
              halves of one mask stay the same distance apart and usually
              re-merge within a frame or two. Recorded as the separation at the
              event and its slope over the following frames, plus whether the
              children fall back into one component later.

  photometry  A broken mask CONSERVES signal: the two pieces sum to what the
              parent had. A genuinely new comet adds signal that was not there
              before. Recorded as sum(children) / parent, on raw background-
              subtracted pixels, when the movie is supplied.

THE NULL, WHICH IS THE POINT

Two unrelated comets that happen to pass close enough also produce an overlap
with two components. That rate is not zero and it is not guessable, so it is
measured: the identical detector is run on NON-CONSECUTIVE frame pairs
(t, t + lag), where by construction no real correspondence exists but the
density, shape and geometry of the field are exactly the same. The number it
returns is how many "splits" this movie yields from coincidence alone.

A split count without that null is uninterpretable, so ``detect`` always
returns both.
"""
from __future__ import annotations

import numpy as np

__all__ = ["detect", "EVENT_FIELDS"]

MIN_AREA = 6           # same floor detections.py uses
MIN_OVERLAP_PX = 3     # a child must share at least this much with the parent
MIN_OVERLAP_FRAC = .10 # ...and this fraction of its own area
FOLLOW = 5             # frames to follow the children forward
NULL_LAGS = (11, 17, 23)   # frame gaps used for the coincidence null

EVENT_FIELDS = ("frame", "n_children", "parent_area", "child_area_a",
                "child_area_b", "angle_axes_deg", "angle_motion_deg",
                "sep0_px", "sep_slope_px_per_frame", "followed_frames",
                "remerged", "intensity_ratio")


def _overlap_pairs(a, b):
    """{(label_a, label_b): shared pixel count} for two label frames."""
    m = (a > 0) & (b > 0)
    if not m.any():
        return {}
    av, bv = a[m].astype(np.int64), b[m].astype(np.int64)
    key = av * (int(b.max()) + 1) + bv
    u, c = np.unique(key, return_counts=True)
    div = int(b.max()) + 1
    return {(int(k // div), int(k % div)): int(n) for k, n in zip(u, c)}


def _props(lab):
    """{label: (area, cy, cx, theta)} for one frame."""
    from skimage.measure import regionprops
    out = {}
    for p in regionprops(lab):
        if p.area < MIN_AREA:
            continue
        cy, cx = p.centroid
        out[int(p.label)] = (int(p.area), float(cy), float(cx),
                             float(p.orientation))
    return out


def _children_of(lab_a, lab_b, pa, pb):
    """{parent: [children]} keeping only parents that split into 2+."""
    ov = _overlap_pairs(lab_a, lab_b)
    byparent: dict[int, list[int]] = {}
    for (la, lb), n in ov.items():
        if la not in pa or lb not in pb:
            continue
        if n < MIN_OVERLAP_PX or n < MIN_OVERLAP_FRAC * pb[lb][0]:
            continue
        byparent.setdefault(la, []).append(lb)
    return {k: v for k, v in byparent.items() if len(v) >= 2}


def _ang(t1, t2):
    """Angle between two orientations, mod pi, in degrees (0..90)."""
    d = abs(t1 - t2) % np.pi
    return float(np.degrees(min(d, np.pi - d)))


def _vec_ang(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))
    return float(np.degrees(np.arccos(c)))


def _successor(lab_a, lab_b, pa, pb, label):
    """The component at t+1 sharing the most pixels with `label` at t."""
    m = lab_a == label
    if not m.any():
        return None
    vals = lab_b[m]
    vals = vals[vals > 0]
    if vals.size == 0:
        return None
    u, c = np.unique(vals, return_counts=True)
    best = int(u[int(np.argmax(c))])
    return best if best in pb else None


def _events(labels, props, movie, lag, follow):
    """Every split candidate at a given frame LAG, fully measured.

    lag=1 is the real detector. A larger lag is the null: the same code, the
    same field, frame pairs that cannot be causally related. The children are
    followed with the same stride, so the null events go through an identical
    measurement and can be put through identical thresholds -- which is the
    only way the filtered counts mean anything.
    """
    T = labels.shape[0]
    ev = {k: [] for k in EVENT_FIELDS}
    for t in range(T - lag):
        if t not in props or (t + lag) not in props:
            continue
        la, lb = labels[t], labels[t + lag]
        pa, pb = props[t], props[t + lag]
        for parent, kids in _children_of(la, lb, pa, pb).items():
            # the two largest children; a 3-way split is recorded by
            # n_children but measured on its two biggest pieces
            kids = sorted(kids, key=lambda k: -pb[k][0])[:2]
            a, b = kids
            aa, ay, ax, at = pb[a]
            ba, by, bx, bt = pb[b]
            pA, py, px, pt = pa[parent]

            ev["frame"].append(t)
            ev["n_children"].append(len(kids))
            ev["parent_area"].append(pA)
            ev["child_area_a"].append(aa)
            ev["child_area_b"].append(ba)
            ev["angle_axes_deg"].append(_ang(at, bt))

            # follow both children forward, one greedy overlap step at a time
            pos_a = [(ay, ax)]; pos_b = [(by, bx)]
            ca, cb = a, b
            k = 0
            remerged = 0
            s = t + lag
            while k < follow and s + lag <= T - 1:
                if s not in props or (s + lag) not in props:
                    break
                na = _successor(labels[s], labels[s + lag], props[s],
                                props[s + lag], ca)
                nb = _successor(labels[s], labels[s + lag], props[s],
                                props[s + lag], cb)
                if na is None or nb is None:
                    break
                if na == nb:                 # fell back into one component
                    remerged = 1
                    break
                ca, cb = na, nb
                pos_a.append(props[s + lag][ca][1:3])
                pos_b.append(props[s + lag][cb][1:3])
                k += 1
                s += lag

            pa_arr = np.asarray(pos_a, float); pb_arr = np.asarray(pos_b, float)
            sep = np.hypot(*(pa_arr - pb_arr).T)
            ev["sep0_px"].append(float(sep[0]))
            ev["followed_frames"].append(int(k))
            ev["remerged"].append(int(remerged))
            if len(sep) >= 3:
                ev["sep_slope_px_per_frame"].append(
                    float(np.polyfit(np.arange(len(sep)), sep, 1)[0]))
                ev["angle_motion_deg"].append(
                    _vec_ang(pa_arr[-1] - pa_arr[0], pb_arr[-1] - pb_arr[0]))
            else:
                ev["sep_slope_px_per_frame"].append(float("nan"))
                ev["angle_motion_deg"].append(float("nan"))

            # photometry: does the signal add up, or was it just divided?
            if movie is not None:
                f0 = movie[t].astype(np.float64)
                f1 = movie[t + lag].astype(np.float64)
                bg0 = float(np.median(f0[la == 0])) if (la == 0).any() else 0.0
                bg1 = float(np.median(f1[lb == 0])) if (lb == 0).any() else 0.0
                ip = float((f0[la == parent] - bg0).sum())
                ic = float((f1[lb == a] - bg1).sum() + (f1[lb == b] - bg1).sum())
                ev["intensity_ratio"].append(ic / ip if ip > 0 else float("nan"))
            else:
                ev["intensity_ratio"].append(float("nan"))

    return {k: np.asarray(v, float) for k, v in ev.items()}


def detect(labels, movie=None, follow=FOLLOW, angle_min=25.0,
           sep_slope_min=0.3):
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError("labels must be (T, Y, X)")
    if movie is not None:
        movie = np.asarray(movie)
        if movie.ndim != 3:
            raise ValueError(f"movie must be (T, Y, X), got {movie.shape}; "
                             "read it with movie.read_movie (488 only)")
        if movie.shape != labels.shape:
            movie = None                      # mismatched: skip photometry

    T = labels.shape[0]
    props = {t: _props(labels[t]) for t in range(T) if labels[t].any()}
    n_det = sum(len(v) for v in props.values())

    out = _events(labels, props, movie, 1, follow)
    real = summarize(out, angle_min, sep_slope_min)

    # The coincidence null. Same detector, same measurements, same thresholds,
    # on frame pairs that cannot be causally related. Counts are scaled by the
    # number of usable frame pairs, since a lag of 23 has fewer of them.
    pairs_real = max(T - 1, 1)
    nulls, nullq = [], []
    for lag in NULL_LAGS:
        if T <= lag + follow:
            continue
        no = _events(labels, props, movie, lag, follow)
        q = summarize(no, angle_min, sep_slope_min)
        scale = pairs_real / max(T - lag, 1)
        nulls.append({k: v * scale for k, v in q.items()
                      if k.startswith("n_")})
        nullq.append(q)

    def nmean(key):
        v = [n[key] for n in nulls if np.isfinite(n[key])]
        return float(np.mean(v)) if v else float("nan")

    out["_summary"] = dict(
        n_frames=int(T),
        n_detections=int(n_det),
        n_candidates=int(real["n_candidates"]),
        n_diverging=int(real["n_diverging"]),
        n_strict=int(real["n_strict"]),
        candidates_per_1000_det=float(1000.0 * real["n_candidates"] / max(n_det, 1)),
        strict_per_1000_det=float(1000.0 * real["n_strict"] / max(n_det, 1)),
        null_lags=list(NULL_LAGS),
        null_candidates=nmean("n_candidates"),
        null_diverging=nmean("n_diverging"),
        null_strict=nmean("n_strict"),
        excess_strict=float(real["n_strict"] - nmean("n_strict")),
        median_angle_axes=real.get("median_angle_axes", float("nan")),
        median_sep_slope=real.get("median_sep_slope", float("nan")),
        frac_remerged=real.get("frac_remerged", float("nan")),
        median_intensity_ratio=real.get("median_intensity_ratio", float("nan")),
        null_median_intensity_ratio=(float(np.nanmean(
            [q.get("median_intensity_ratio", np.nan) for q in nullq]))
            if nullq else float("nan")),
        angle_min=angle_min, sep_slope_min=sep_slope_min,
    )
    return out


def summarize(out, angle_min=25.0, sep_slope_min=0.3):
    """Counts after the discriminators, so a headline number can be quoted.

    A candidate is called REAL only if the children lie off each other's axis,
    move apart, and do not fall back together. The thresholds are arguments
    because they are judgement calls, not facts.
    """
    if len(out["frame"]) == 0:
        return dict(n_candidates=0, n_diverging=0, n_strict=0)
    ang = out["angle_axes_deg"]
    amo = out["angle_motion_deg"]
    slope = out["sep_slope_px_per_frame"]
    rem = out["remerged"]
    diverging = (np.nan_to_num(slope, nan=-1) >= sep_slope_min) & (rem == 0)
    strict = diverging & (np.nan_to_num(ang, nan=0) >= angle_min) & \
        (np.nan_to_num(amo, nan=0) >= angle_min)
    return dict(n_candidates=int(len(ang)),
                n_diverging=int(diverging.sum()),
                n_strict=int(strict.sum()),
                median_angle_axes=float(np.nanmedian(ang)),
                median_sep_slope=float(np.nanmedian(slope)),
                frac_remerged=float(np.mean(rem)),
                median_intensity_ratio=float(np.nanmedian(out["intensity_ratio"])))
