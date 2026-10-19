"""One command, one predictions folder, every number this project reports.

    python -m comet_tracker_v9.postprocess --pred PREDS --nd2 MOVIES --out OUT

Until now each analysis lived in its own throwaway script and its numbers lived
in a chat window. That does not survive contact with twenty more movies. This
module runs the whole list in one pass per movie and writes a JSON per movie
plus a combined file, so the per-cell-line comparison at the end is a group-by
over files that already exist rather than a re-run of everything.

WHAT IT COMPUTES

  1. velocity   -- centroid and head speed, per track, WITH the motion class
                   under all three criteria attached to every row (see the note
                   below). Splitting happens at read time, not here.
  2. length     -- mask length distribution swept over detector confidence,
                   because mask length is partly a readout of that confidence
                   and a single number would hide it.
  3. lifetime   -- how many frames each comet is detected for.
  4. detections -- count per frame, which is the bleaching check: a downward
                   trend means later frames are not comparable to earlier ones.
  5. catastrophe -- OFF by default (``--catastrophe``). On the two pilot movies
                   neither the centroid nor the head measurement was convincing
                   and the movies show little real catastrophe, so it is opt-in
                   rather than a default output nobody should trust.

WHY EVERY ROW CARRIES THREE CLASSES

``MotionConfig.criterion`` picks which one is authoritative, but all three are
written to every row. The short-track criterion decides the class of nearly
every track here (min_frames_for_mss is 20; real lifetimes are 3-25 frames), so
it is not a detail that can be settled once and forgotten. Carrying all three
costs nothing at write time and means the sensitivity of any conclusion to that
choice can be read straight off the file. If a result holds under all three,
say so; if it only holds under one, that is the finding.

WHAT IS NOT HERE

Comet length as a physical measurement. What is written is the MASK length,
under the name ``mask_major_px``, swept over confidence. The mask is a
fixed-width rendered centreline whose ends erode as presence falls, so this is
a detector readout with a length-like shape, not the fluorescent comet's
length. Reading it off the raw image with the steerable ridge profile is the
open route to a real number (see tools/steerable_probe.py).
"""
from __future__ import annotations

import argparse
import glob
import re
import json
import os

import numpy as np

from .config import MICRONS_PER_PIXEL, SECONDS_PER_FRAME, Config
from . import summarize as _summarize
from .movie import ZoomTable, read_movie
from .detections import from_prediction_folder
from .motion import CRITERIA
from .photometry import measure as _measure_photometry
from .splits import detect as _detect_splits
from .pipeline import attach_heads, run
from .summarize import fit_speed_px_per_frame

__all__ = ["postprocess_movie", "postprocess_folder", "find_stems"]

# The scale is set PER MOVIE by _set_scale() from the zoom table -- these
# module values are only the fallback when nothing sets it. NIS records the 1x
# pixel for every file whatever the optovar was, so MICRONS_PER_PIXEL from the
# config is right only for movies taken at 1x. See movie.py.
_PX, _SEC = MICRONS_PER_PIXEL, SECONDS_PER_FRAME
_UM_MIN = _PX / _SEC * 60.0          # px/frame -> um/min


def _set_scale(px_um: float, sec: float):
    """Point every px->um conversion at one movie's scale; returns the old one.

    Both this module and summarize.py convert with module-level constants, so
    both are set together. Always paired with a restore in ``finally``.
    """
    global _PX, _SEC, _UM_MIN
    old = (_PX, _SEC)
    _PX, _SEC = float(px_um), float(sec)
    _UM_MIN = _PX / _SEC * 60.0
    _summarize._PX, _summarize._SEC = _PX, _SEC
    return old

CONFIDENCE_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)

EMG_MIN_FRAMES = 3      # a track needs this many usable EMG frames for an EMG speed


def _robust_speed_px_per_frame(frames, positions) -> float:
    """Speed from a Theil-Sen line through position vs frame, per coordinate.

    Theil-Sen takes the median of all pairwise slopes, so one frame whose
    position jumped -- a neighbour pulling the fit, a mask-only refit landing
    a few px off -- cannot drag the speed the way it drags least squares.
    """
    from scipy.stats import theilslopes

    f = np.asarray(frames, float)
    p = np.asarray(positions, float)
    if len(f) < 2 or np.ptp(f) <= 0:
        return float("nan")
    vy = theilslopes(p[:, 0], f)[0]
    vx = theilslopes(p[:, 1], f)[0]
    return float(np.hypot(vy, vx))


def _emg_speeds(phot, di, frames_of_det, theta_of_det, travel) -> dict:
    """v_emg_mu / v_emg_max for one track, from its usable EMG frames only.

    Which end of each comet is the plus end comes from the TRACK: photometry
    fits every comet both ways round, and here each frame takes the fit whose
    plus end faces the track's direction of travel (from the centroid fit).
    A comet's own profile is a poor judge of that -- choosing by R^2 put the
    plus end behind the comet's peak in 30% of frames on the test movie.

    A frame without a usable EMG position (mask < 4 px, fit failed, or failed
    the precision / blur checks) is simply LEFT OUT -- the centroid is never
    swapped in for it, because mu sits ~3 px ahead of the centroid, about one
    frame of travel, and mixing the two would bend the line.
    """
    out = dict(v_emg_mu_um_min=float("nan"), v_emg_max_um_min=float("nan"),
               n_emg_frames=0, n_emg_mask_frames=0, n_emg_short_frames=0)
    if phot is None or "emg_ok_p" not in phot:
        return out
    T = np.asarray(travel, float)
    if not np.all(np.isfinite(T)) or np.hypot(*T) == 0:
        return out
    th = np.asarray(theta_of_det, float)
    along = np.cos(th) * T[0] + np.sin(th) * T[1]       # axis . travel
    use_p = along >= 0
    pick = lambda k: np.where(use_p, phot[f"emg_{k}_p"][di], phot[f"emg_{k}_n"][di])
    ok = pick("ok") > 0
    win = pick("window")
    out["n_emg_frames"] = int(ok.sum())
    out["n_emg_mask_frames"] = int((ok & (win == 2)).sum())
    out["n_emg_short_frames"] = int((win == -1).sum())
    if ok.sum() < EMG_MIN_FRAMES:
        return out
    fr = np.asarray(frames_of_det, float)[ok]
    for kind in ("mu", "max"):
        pos = np.c_[pick(f"{kind}_y")[ok], pick(f"{kind}_x")[ok]]
        out[f"v_emg_{kind}_um_min"] = float(_robust_speed_px_per_frame(fr, pos) * _UM_MIN)
    return out


def find_stems(pred_dir: str) -> list[str]:
    """Every movie with a label stack in this folder."""
    return sorted(os.path.basename(p)[:-len("_labels.tif")]
                  for p in glob.glob(os.path.join(pred_dir, "*_labels.tif")))


_DAY_RE = re.compile(r"(?:^|[^0-9])(20[0-9]{6})(?:[^0-9]|$)")


def day_from_name(name: str):
    """The imaging day as a YYYYMMDD string, or None.

    Every movie in this set is named <date>_<construct>_<condition>_<field>,
    so the date is already there. Read rather than assumed: anything that is
    not eight digits starting 20 is ignored.
    """
    m = _DAY_RE.search(os.path.basename(str(name)))
    return m.group(1) if m else None


def locate_movie(stem: str, nd2_dir=None, nd2_root=None):
    """Find <stem>.nd2, the design it belongs to, and the repeat it came from.

    Returns (folder, group, day, day_source) or (None, None, None, None).

    ``nd2_root`` is the useful one: point it at Data/ and every movie finds its
    own design from the folder it sits in, so the whole set runs as one command
    instead of one invocation per design with --group typed by hand. Typing it
    by hand is how a mislabelled group gets into a final comparison.

    Two layouts are accepted under nd2_root, because both are in use:

        Data/<design>/<stem>.nd2              flat
        Data/<design>/<YYYYMMDD>/<stem>.nd2   one subfolder per repeat

    A repeat is an imaging day. With the nested layout the day comes from the
    folder; otherwise from the date the filename already starts with. When both
    are available they must agree -- a movie filed under the wrong day would
    otherwise travel all the way into a figure.
    """
    def _finish(d, group):
        f_day = day_from_name(stem)
        leaf = os.path.basename(os.path.normpath(d))
        d_day = leaf if re.fullmatch(r"20[0-9]{6}", leaf) else None
        if d_day and f_day and d_day != f_day:
            raise SystemExit(
                f"{stem}.nd2 sits in folder {d_day} but its name says {f_day}"
                " -- one of them is wrong, fix it before running")
        if d_day and f_day:
            return d, group, d_day, "both"
        if d_day:
            return d, group, d_day, "folder"
        if f_day:
            return d, group, f_day, "filename"
        return d, group, None, None

    if nd2_dir and os.path.exists(os.path.join(nd2_dir, stem + ".nd2")):
        leaf = os.path.basename(os.path.normpath(nd2_dir))
        parent = os.path.basename(os.path.dirname(os.path.normpath(nd2_dir)))
        grp = parent if re.fullmatch(r"20[0-9]{6}", leaf) else leaf
        return _finish(nd2_dir, grp)
    if nd2_root:
        # one level = design/, two = design/day/. Both are searched, and a stem
        # found in more than one place is an error rather than a guess.
        hits = sorted(set(glob.glob(os.path.join(nd2_root, "*", stem + ".nd2")))
                      | set(glob.glob(os.path.join(nd2_root, "*", "*",
                                                   stem + ".nd2"))))
        if len(hits) > 1:
            raise SystemExit(
                f"{stem}.nd2 exists in more than one folder under {nd2_root}: "
                + ", ".join(os.path.dirname(h) for h in hits)
                + " -- cannot pick a design, resolve the duplicate first")
        if hits:
            d = os.path.dirname(hits[0])
            leaf = os.path.basename(d)
            grp = (os.path.basename(os.path.dirname(d))
                   if re.fullmatch(r"20[0-9]{6}", leaf) else leaf)
            return _finish(d, grp)
    return None, None, None, None


def _pct(v, qs=(5, 25, 50, 75, 95)) -> dict:
    v = np.asarray([x for x in np.ravel(v) if np.isfinite(x)], float)
    if v.size == 0:
        return dict(n=0)
    return dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std(ddof=1))
                if v.size > 1 else 0.0,
                **{f"p{q}": float(np.percentile(v, q)) for q in qs})


# --- 1. velocity -----------------------------------------------------------
def _velocity_rows(result, phot=None) -> list[dict]:
    """One row per compound track.

    Carries the track's MASK SHAPE alongside its speed, because the two are
    only useful together. Comet length rises with growth rate (r = +0.47 at the
    track level on the pilot movie), so a length difference between cell lines
    means nothing until the lines are compared at matched speed -- and that
    join is impossible unless both numbers sit on the same row.

    Length is summarised as a median over the track's own detections. That is a
    fair summary because a comet's length is stable along its track: measured
    within-track CV is 0.156 against 0.55 between tracks. ``score`` comes along
    for the same reason -- mask length tracks detector confidence almost as
    strongly as it tracks speed, so any comparison has to be able to control
    for it.
    """
    rows = []
    heads = result.heads or {}
    dets = result.detections
    maj = np.asarray(dets.major, float) * _PX
    mnr = np.asarray(dets.minor, float) * _PX
    sco = np.asarray(dets.score, float)
    for c in result.compounds:
        pos = np.vstack([s.positions() for s in c.segments])
        # s.frames, NOT arange(start, end+1): with max_gap > 0 a segment
        # coasts over missed frames, so its frame list is not contiguous and
        # rebuilding it from the endpoints silently misaligns the fit.
        fr = np.concatenate([np.asarray(s.frames, float) for s in c.segments])
        v_cent = fit_speed_px_per_frame(fr, pos)

        # how straight-and-steady that track actually was. Two coordinates,
        # one number: sum the residuals and the deviations over both, so a
        # track moving mostly in x is not scored differently from one moving
        # mostly in y.
        if len(fr) >= 3:
            _t = fr - fr.mean()
            _ss_res = _ss_tot = 0.0
            for _k in (0, 1):
                _c = np.polyfit(fr, pos[:, _k], 1)
                _r = pos[:, _k] - np.polyval(_c, fr)
                _ss_res += float((_r ** 2).sum())
                _ss_tot += float(((pos[:, _k] - pos[:, _k].mean()) ** 2).sum())
            fit_r2 = float(1.0 - _ss_res / _ss_tot) if _ss_tot > 0 else float("nan")
            fit_se = float(np.sqrt(_ss_res / max(len(fr) - 2, 1)))
        else:
            fit_r2, fit_se = float("nan"), float("nan")

        # head speed, from the gated head track of every segment pooled
        hf, hp = [], []
        for s in c.segments:
            th = heads.get(s.track_id)
            if th is None:
                continue
            g = th.gate()
            ok = g.valid
            if ok.sum() >= 2:
                hf.append(np.asarray(g.frames)[ok])
                hp.append(g.heads[ok])
        v_head = (fit_speed_px_per_frame(np.concatenate(hf), np.vstack(hp))
                  if hf else float("nan"))

        di = np.concatenate([np.asarray(sg.det_index, int) for sg in c.segments])
        L, W, SC = maj[di], mnr[di], sco[di]
        # direction of travel = the centroid's own fitted velocity vector
        if len(fr) >= 2 and np.ptp(fr) > 0:
            travel = (np.polyfit(fr, pos[:, 0], 1)[0], np.polyfit(fr, pos[:, 1], 1)[0])
        else:
            travel = (np.nan, np.nan)
        emg = _emg_speeds(phot, di, np.asarray(dets.frame, float)[di],
                          np.asarray(dets.theta, float)[di], travel)
        v_c_um = float(v_cent * _UM_MIN)
        # No fallback: a track without enough usable EMG frames keeps NaN for
        # its EMG speeds. Filling them with the centroid was rejected -- fast
        # comets lose their EMG speed far more often (37% vs 74% on the test
        # movie), so the mix would bias any comparison toward slow comets.

        # photometry, measured on raw pixels -- see photometry.py. Aggregated
        # the same way as mask length: median over the track's own detections,
        # and only over the detections whose fit survived its quality gates, so
        # a failed fit becomes a smaller n rather than a wrong number.
        ph = {}
        if phot is not None:
            for key, out in (("peak", "int_peak"), ("integrated", "int_integrated"),
                             ("mean", "int_mean"), ("bg", "int_bg"),
                             ("peak_prof", "int_peak_prof"),
                             ("bg_side", "int_bg_side"),
                             ("bg_side_n", "bg_side_n"), ("area", "mask_area_px"),
                             ("peak_side", "int_peak_side"),
                             ("integrated_side", "int_integrated_side"),
                             ("mean_side", "int_mean_side"),
                             ("peak_prof_side", "int_peak_prof_side")):
                v = phot[key][di]
                ph[out] = float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")
            # auc and its extent are measured in px along the axis, so both
            # carry a pixel size into physical units -- counts*um and um.
            for key, out in (("auc", "int_auc"), ("auc_extent_px", "auc_extent_um"),
                             ("auc_side", "int_auc_side"),
                             ("auc_extent_side_px", "auc_extent_side_um")):
                v = phot[key][di]
                ph[out] = (float(np.nanmedian(v) * _PX)
                           if np.isfinite(v).any() else float("nan"))
            # (source field, output name, R^2 gate). The Bieling route carries
            # its own gate because that protocol SPECIFIES one: only growth
            # events whose exponential fits with R^2 > 0.9 are kept. The other
            # routes keep every finite fit and write their R^2 alongside, so a
            # gate can be applied later instead of baked in.
            # Gates applied PER FIT, before aggregating -- not to the track's
            # median R^2 afterwards. Those are different operations: a track
            # with three good frames and three bad ones should contribute its
            # three good frames, not be kept or dropped whole.
            for src, out, r2min in (("gauss_fwhm_px", "gauss_len_um", 0.88),
                                    ("exp_lambda_px", "exp_len_um", None),
                                    ("emg_len_px", "emg_len_um", 0.92),
                                    ("emg_fwhm_px", "emg_fwhm_um", 0.92),
                                    ("biel_len_px", "biel_len_um", 0.9),
                                    ("biel_len_side_px", "biel_side_len_um", 0.9),
                                    ("biel_len_off_px", "biel_off_len_um", 0.9)):
                v = phot[src][di]
                r2 = phot[{"biel_len_off_px": "biel_off_r2",
                           "biel_len_side_px": "biel_side_r2"}.get(
                              src, src.split("_")[0] + "_r2")][di]
                good = np.isfinite(v)
                if r2min is not None:
                    good = good & np.isfinite(r2) & (r2 > r2min)
                ph[out] = float(np.nanmedian(v[good]) * _PX) if good.any() else float("nan")
                ph[out + "_n"] = int(good.sum())
                ph[out.replace("_um", "_r2")] = (float(np.nanmedian(r2[good]))
                                                 if good.any() else float("nan"))
            snr = phot["profile_snr"][di]
            ph["profile_snr"] = float(np.nanmedian(snr)) if np.isfinite(snr).any() else float("nan")

        classes = getattr(c, "motion_classes", None) or {
            k: getattr(c, "motion_class", "unclassified") for k in CRITERIA}
        rows.append(dict(
            n_frames=int(len(pos)),
            fit_r2=fit_r2,
            fit_se_px=fit_se,
            method=getattr(c, "motion_method", "?"),
            mss_slope=float(getattr(c, "mss_slope", float("nan"))),
            straightness=float(getattr(c, "straightness", float("nan"))),
            diffusion_ratio=float(getattr(c, "diffusion_ratio", float("nan"))),
            rms_step_px=float(getattr(c, "rms_step_px", float("nan"))),
            v_centroid_um_min=v_c_um,
            v_head_um_min=float(v_head * _UM_MIN),
            **emg,
            mask_len_um=float(np.median(L)),
            mask_len_iqr=[float(np.percentile(L, 25)), float(np.percentile(L, 75))],
            mask_width_um=float(np.median(W)),
            mask_aspect=float(np.median(L / np.maximum(W, 1e-6))),
            # the per-track median of SAM3's own confidence in each mask.
            # Named in full so it is never confused with a fit quality.
            SAM3_confidence_score=float(np.median(SC)),
            n_det=int(len(di)),
            n_fgap=int(c.n_fgap()), n_bgap=int(c.n_bgap()),
            **ph,
            **{f"class_{k}": classes.get(k, "unclassified") for k in CRITERIA},
        ))
    return rows


def _velocity_summary(rows) -> dict:
    """Speed distributions per class, per criterion, for both estimators."""
    out = {}
    for crit in CRITERIA:
        per_class = {}
        for cls in sorted({r[f"class_{crit}"] for r in rows}):
            sel = [r for r in rows if r[f"class_{crit}"] == cls]
            per_class[cls] = dict(
                n=len(sel),
                centroid=_pct([r["v_centroid_um_min"] for r in sel]),
                head=_pct([r["v_head_um_min"] for r in sel]),
                emg_mu=_pct([r.get("v_emg_mu_um_min", float("nan")) for r in sel]),
                emg_max=_pct([r.get("v_emg_max_um_min", float("nan")) for r in sel]),
                n_with_emg=int(sum(np.isfinite(r.get("v_emg_mu_um_min", float("nan")))
                                   for r in sel)),
                n_frames=_pct([r["n_frames"] for r in sel]),
            )
        out[crit] = per_class
    return out


def _speed_by_length(rows, crit) -> dict:
    """Median directed speed against track length -- the diagnostic that
    exposed the net_disp defect. A criterion that is not length-dependent
    should give a flat-ish line here."""
    sel = [r for r in rows if r[f"class_{crit}"] == "directed"]
    out = {}
    for lo, hi in [(3, 4), (4, 6), (6, 9), (9, 15), (15, 10**6)]:
        s = [r for r in sel if lo <= r["n_frames"] < hi]
        if s:
            out[f"{lo}-{hi if hi < 10**6 else 'inf'}"] = dict(
                n=len(s),
                centroid=_pct([r["v_centroid_um_min"] for r in s], (50,)),
                head=_pct([r["v_head_um_min"] for r in s], (50,)))
    return out


# --- 2-4. detection-level analyses -----------------------------------------
def _detection_analyses(dets) -> dict:
    frames = np.asarray(dets.frame)
    n_frames = int(frames.max()) + 1 if frames.size else 0
    per_frame = np.bincount(frames, minlength=n_frames).tolist()

    length = {}
    for thr in CONFIDENCE_GRID:
        m = np.asarray(dets.score) >= thr
        length[str(thr)] = dict(
            kept=int(m.sum()),
            frac=float(m.mean()) if m.size else 0.0,
            mask_major_px=_pct(np.asarray(dets.major)[m]),
            mask_major_um=_pct(np.asarray(dets.major)[m] * _PX),
            aspect=_pct(np.asarray(dets.major)[m]
                        / np.maximum(np.asarray(dets.minor)[m], 1e-6)),
        )
    return dict(detections_per_frame=per_frame,
                detections_per_frame_summary=_pct(per_frame),
                length_by_confidence=length)


def _lifetime(result) -> dict:
    per_track = [int(sum(len(s) for s in c.segments)) for c in result.compounds]
    return dict(frames=_pct(per_track),
                seconds=_pct([v * _SEC for v in per_track]),
                histogram=np.bincount(per_track).tolist() if per_track else [])


# --- driver ----------------------------------------------------------------
def postprocess_movie(pred_dir, stem, nd2_dir=None, config=None,
                      with_heads=True, group=None, nd2_root=None,
                      photometry=False, splits=False, zoom_table=None) -> dict:
    """One movie, at its own scale.

    ``zoom_table`` is a ZoomTable or a CSV path; None means every movie is
    taken at the default 1.5x. The px->um scale is set for this movie only and
    restored on the way out, whatever happens.
    """
    zt = zoom_table if isinstance(zoom_table, ZoomTable) else ZoomTable(zoom_table)
    zoom, px_um, zoom_src = zt.lookup(stem)
    old = _set_scale(px_um, SECONDS_PER_FRAME)
    try:
        d = _postprocess_movie(pred_dir, stem, nd2_dir, config, with_heads,
                               group, nd2_root, photometry, splits)
    finally:
        _set_scale(*old)
    d["zoom"] = zoom
    d["zoom_source"] = zoom_src
    return d


def _postprocess_movie(pred_dir, stem, nd2_dir, config, with_heads, group,
                       nd2_root, photometry, splits) -> dict:
    cfg = config or Config()
    day = day_source = None
    if nd2_root or not nd2_dir:
        found, g, day, day_source = locate_movie(stem, nd2_dir, nd2_root)
        nd2_dir = found or nd2_dir
        group = group or g
    dets = from_prediction_folder(pred_dir, stem, cfg.detect)
    result = run(dets, cfg)

    # The movie is read ONCE, 488 channel only, and shared by heads,
    # photometry and splits. A two-channel file is (T, C, Y, X); read_movie
    # picks the channel by name so the 640 channel never reaches any of them.
    movie, minfo, movie_note = None, None, None
    if nd2_dir and (with_heads or photometry or splits):
        try:
            movie, minfo = read_movie(os.path.join(nd2_dir, stem + ".nd2"))
            if len(dets) and movie.shape[0] <= int(np.max(dets.frame)):
                raise ValueError(f"movie has {movie.shape[0]} frames but "
                                 f"detections reach frame {int(np.max(dets.frame))}")
        except Exception as exc:                      # noqa: BLE001
            movie, movie_note = None, f"{type(exc).__name__}: {exc}"

    head_note = None
    if with_heads and nd2_dir:
        try:
            import tifffile
            if movie is None:
                raise RuntimeError(f"movie unreadable -- {movie_note}")
            labels = tifffile.imread(os.path.join(pred_dir,
                                                  stem + "_labels.tif"))
            result = attach_heads(result, movie, labels)
        except Exception as exc:                      # noqa: BLE001
            # Heads need comet_sam3.preprocessing to rebuild the causal RGB
            # exactly as the model saw it. Without it the centroid numbers are
            # still correct, so this degrades rather than fails -- but it is
            # recorded, because a file full of NaN head speeds with no
            # explanation is worse than no file.
            head_note = f"{type(exc).__name__}: {exc}"

    phot, phot_note = None, None
    if photometry and nd2_dir:
        try:
            import tifffile
            if movie is None:
                raise RuntimeError(f"movie unreadable -- {movie_note}")
            labels = tifffile.imread(os.path.join(pred_dir, stem + "_labels.tif"))
            pp = os.path.join(pred_dir, stem + "_prob.tif")
            prob = tifffile.imread(pp) if os.path.exists(pp) else None
            phot = _measure_photometry(movie, labels, prob, cfg.detect)
            if len(phot["peak"]) != len(dets):
                raise RuntimeError(
                    f"photometry/detection mismatch: {len(phot['peak'])} vs {len(dets)}")
        except Exception as exc:                      # noqa: BLE001
            phot, phot_note = None, f"{type(exc).__name__}: {exc}"

    # Splitting comets, read off the label stack rather than the tracks: the
    # tracker has no split model, so a split is currently invisible in the
    # track output -- it looks like one track dying and another being born.
    split_summary, split_note = None, None
    if splits:
        try:
            import tifffile
            labels = tifffile.imread(os.path.join(pred_dir, stem + "_labels.tif"))
            split_summary = _detect_splits(labels, movie)["_summary"]
        except Exception as exc:                      # noqa: BLE001
            split_note = f"{type(exc).__name__}: {exc}"

    rows = _velocity_rows(result, phot)
    return dict(
        photometry_failure=phot_note,
        splits=split_summary,
        splits_failure=split_note,
        stem=stem,
        group=group,
        day=day or day_from_name(stem),
        day_source=day_source or ("filename" if day_from_name(stem) else None),
        min_track_length=cfg.link.min_track_length,
        max_gap=cfg.link.max_gap,
        min_score=cfg.detect.min_score,
        pixel_um=_PX, seconds_per_frame=_SEC,
        pixel_um_metadata=minfo.pixel_um_metadata if minfo else None,
        seconds_per_frame_metadata=minfo.seconds_per_frame if minfo else None,
        n_channels=minfo.n_channels if minfo else None,
        channel_used=minfo.channel_name if minfo else None,
        channel_note=minfo.channel_note if minfo else None,
        movie_failure=movie_note,
        criterion_default=cfg.motion.criterion,
        head_failure=head_note,
        n_detections=len(dets),
        n_tracks=len(result.compounds),
        velocity=dict(
            per_track=rows,
            by_class=_velocity_summary(rows),
            directed_speed_by_length={c: _speed_by_length(rows, c)
                                      for c in CRITERIA},
        ),
        lifetime=_lifetime(result),
        **_detection_analyses(dets),
        movie_stats=result.stats,
    )


def postprocess_folder(pred_dir, nd2_dir=None, out_dir=None, config=None,
                       with_heads=True, group=None, nd2_root=None,
                       skip_existing=False, shard=(0, 1),
                       photometry=False, splits=False, zoom_table=None,
                       default_zoom=None) -> dict:
    stems = find_stems(pred_dir)
    if not isinstance(zoom_table, ZoomTable):
        zoom_table = (ZoomTable(zoom_table, default_zoom) if default_zoom
                      else ZoomTable(zoom_table))
    print(f"zoom: {zoom_table.default:g}x unless listed; table "
          f"{zoom_table.path or '(none)'} lists {zoom_table.n_at(1.0)} at 1x, "
          f"{zoom_table.n_at(1.5)} at 1.5x", flush=True)
    if not stems:
        raise SystemExit(f"no *_labels.tif in {pred_dir}")
    # Interleaved, not contiguous: movie size varies ~10x across the set, and
    # contiguous blocks would put all the big ones in one task. stems[i::n]
    # spreads them, so the tasks finish at roughly the same time.
    i, n = shard
    if n > 1:
        stems = stems[i::n]
    # The cell line is what the final comparison groups by, and nothing in a
    # prediction folder records it -- so it is taken from the movie folder's
    # name unless given explicitly. Writing it into every file now is the
    # difference between a group-by at the end and re-deriving it from
    # filenames months later.
    if group is None and nd2_dir:
        group = os.path.basename(os.path.normpath(nd2_dir))
    out_dir = out_dir or os.path.join(pred_dir, "postprocess")
    os.makedirs(out_dir, exist_ok=True)

    combined = {}
    for stem in stems:
        dest = os.path.join(out_dir, stem + "_post.json")
        if skip_existing and os.path.exists(dest):
            print(f"  {stem} -- already done, skipped", flush=True)
            combined[stem] = json.load(open(dest))
            continue
        print(f"  {stem} ...", flush=True)
        d = postprocess_movie(pred_dir, stem, nd2_dir, config, with_heads,
                              group=group, nd2_root=nd2_root,
                              photometry=photometry, splits=splits,
                              zoom_table=zoom_table)
        combined[stem] = d
        with open(dest, "w") as fh:
            json.dump(d, fh)
        if d.get("movie_failure"):
            print(f"    MOVIE UNREADABLE -- {d['movie_failure']}")
        if d["head_failure"]:
            print(f"    heads unavailable -- {d['head_failure']}")
        sm = d.get("seconds_per_frame_metadata")
        if sm and abs(sm - d["seconds_per_frame"]) > 0.05 * d["seconds_per_frame"]:
            print(f"    WARNING: metadata says {sm:g} s/frame, "
                  f"using {d['seconds_per_frame']:g}")
        print(f"    [{d['group']}] {d['n_tracks']} tracks, "
              f"{d['n_detections']} detections, {d['zoom']:g}x "
              f"({d['zoom_source']}) {d['pixel_um']:.4f} um/px, "
              f"ch {d.get('channel_used')}")

    # Only the whole-set run writes combined.json. A shard writing it would
    # produce a file holding one sixth of the movies under a name that claims
    # to hold all of them, which is worse than not having it -- rebuild it with
    # --combine-only once every shard is done.
    if shard[1] == 1:
        write_combined(out_dir)
    print(f"wrote {len(stems)} movies to {out_dir}")
    return combined


def write_combined(out_dir: str) -> dict:
    """Rebuild combined.json from whatever per-movie files are present."""
    combined = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "*_post.json"))):
        d = json.load(open(f))
        combined[d["stem"]] = d
    with open(os.path.join(out_dir, "combined.json"), "w") as fh:
        json.dump(combined, fh)
    print(f"combined.json: {len(combined)} movies")
    return combined


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pred", required=True, help="folder of *_labels.tif")
    ap.add_argument("--nd2", help="folder of matching .nd2 movies (for heads)")
    ap.add_argument("--nd2-root", help="parent of per-cell-line movie folders "
                    "(e.g. Data/); each movie finds its own group from the "
                    "folder it sits in")
    ap.add_argument("--min-track-length", type=int, default=None)
    ap.add_argument("--max-gap", type=int, default=None)
    ap.add_argument("--photometry", action="store_true",
                    help="measure per-detection intensity and fit a comet "
                         "length from the raw pixels (needs the .nd2); adds "
                         "~40 s per movie")
    ap.add_argument("--splits", action="store_true",
                    help="look for one comet becoming two, with a coincidence "
                         "null from lagged frame pairs; ~6 s per movie")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave movies that already have an output file alone, "
                         "so an interrupted run resumes instead of restarting")
    ap.add_argument("--out", help="output folder (default PRED/postprocess)")
    ap.add_argument("--criterion", default=None, choices=CRITERIA,
                    help="which motion criterion is authoritative; all three "
                         "are written to every row regardless")
    ap.add_argument("--group", default=None,
                    help="cell line / condition label written into every "
                         "output file; defaults to the --nd2 folder name")
    ap.add_argument("--min-score", type=float, default=None,
                    help="SAM3 presence floor, 0-1. Only filters UPWARD from "
                         "the 0.5 already baked into labels.tif at export; a "
                         "value below 0.5 cannot recover anything and is "
                         "rejected. Default 0.0 (keep everything exported).")
    ap.add_argument("--shard", default="0/1", metavar="I/N",
                    help="process only every Nth movie starting at I, for "
                         "running as a job array; shards never collide because "
                         "each writes only its own movies")
    ap.add_argument("--combine-only", action="store_true",
                    help="rebuild combined.json from the per-movie files and "
                         "exit; run this once every shard has finished")
    ap.add_argument("--no-heads", action="store_true")
    ap.add_argument("--zoom-table", default=None,
                    help="CSV with stem,zoom (1 or 1.5). Movies not listed are "
                         "taken at --default-zoom. NIS does not record the "
                         "1.5x optovar, so this is the only source of scale")
    ap.add_argument("--default-zoom", type=float, default=1.5, choices=(1.0, 1.5),
                    help="zoom for movies not in the table (default 1.5)")
    a = ap.parse_args(argv)

    if a.combine_only:
        write_combined(a.out or os.path.join(a.pred, "postprocess"))
        return

    try:
        si, sn = (int(x) for x in a.shard.split("/"))
        assert 0 <= si < sn
    except Exception:
        raise SystemExit(f"--shard must be I/N with 0 <= I < N, got {a.shard!r}")

    cfg = Config()
    if a.criterion:
        cfg.motion.criterion = a.criterion
    if a.min_track_length is not None:
        cfg.link.min_track_length = a.min_track_length
    if a.max_gap is not None:
        cfg.link.max_gap = a.max_gap
    if a.min_score is not None:
        if a.min_score < 0.5 and a.min_score != 0.0:
            raise SystemExit(
                f"--min-score {a.min_score} is below the 0.5 threshold already "
                "applied by sam3_export when it built labels.tif. Nothing "
                "between 0 and 0.5 exists in the input, so this can only "
                "mislead. To go below 0.5, re-run the SAM3 export.")
        cfg.detect.min_score = a.min_score
    print(f"min_track_length={cfg.link.min_track_length} "
          f"max_gap={cfg.link.max_gap} criterion={cfg.motion.criterion} "
          f"min_score={cfg.detect.min_score}")
    postprocess_folder(a.pred, a.nd2, a.out, cfg, with_heads=not a.no_heads,
                       group=a.group, nd2_root=a.nd2_root,
                       skip_existing=a.skip_existing, shard=(si, sn),
                       photometry=a.photometry, splits=a.splits,
                       zoom_table=a.zoom_table, default_zoom=a.default_zoom)


if __name__ == "__main__":
    main()
