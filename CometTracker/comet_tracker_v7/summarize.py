"""Per-track and per-movie numbers, in physical units.

FREQUENCY DEFINITIONS ARE plusTip's, NOT AN INVENTION

``plusTipDynamParam.m`` lines 378-386 and 474-479 define gap frequency as the
mean over growth subtracks of the RECIPROCAL of that subtrack's own duration --
not one over the mean duration, and not a count divided by total time. Only
growth subtracks that actually END in a gap of that kind are included::

    fgap_freq_time = mean over growth subtracks followed by an fgap of
                     (1 / duration_of_that_subtrack_in_seconds)

with a matching "length" version using the subtrack's displacement in microns.
``bgap_freq_time`` is the catastrophe frequency.

The three definitions give different numbers on the same data, so mixing them
silently is a way to disagree with a published figure for no reason. This
module implements plusTip's.

WHAT IS DELIBERATELY NOT REPORTED

  * comet length. The mask is a fixed-width rendered centreline
    (``soft_uniform_axis``, "deliberately no taper"), and its length is partly a
    readout of detector presence. It is not the fluorescent comet's length and
    is not reported as if it were.
  * pause and shrinkage SPEED. EB3 marks growing ends only, so during an fgap or
    bgap there is nothing to see. plusTip reports a speed for these by dividing
    the gap displacement by the gap duration, which is an inference from the two
    endpoints, not a measurement. That number is emitted here as
    ``fgap_speed_inferred`` / ``bgap_speed_inferred`` with the name saying so.
  * any correction for the centroid-vs-tip bias. Mask length drifts a median
    6.8 px over a track, so centroid displacement understates plus-end
    displacement by roughly half of that. The per-track ``length_drift_px``
    column carries the raw material for that correction; it is not applied,
    because applying it needs a plus-end estimator this version does not have.
"""
from __future__ import annotations

import numpy as np

from .config import MICRONS_PER_PIXEL, SECONDS_PER_FRAME, Config
from .detections import DetectionTable
from .tracks import GAP_BACKWARD, GAP_FORWARD, CompoundTrack

__all__ = ["summarize_tracks", "movie_stats"]

_PX = MICRONS_PER_PIXEL
_SEC = SECONDS_PER_FRAME


def _segment_rows(seg, dets: DetectionTable | None) -> dict:
    p = seg.positions()
    n_steps = max(len(p) - 1, 1)
    frames = seg.frames
    duration_s = (frames[-1] - frames[0]) * _SEC
    disp_um = float(np.hypot(*(p[-1] - p[0]))) * _PX
    path_um = float(np.hypot(*np.diff(p, axis=0).T).sum()) * _PX if len(p) > 1 else 0.0
    speed = (path_um / duration_s * 60.0) if duration_s > 0 else float("nan")

    drift = float("nan")
    if dets is not None and len(seg.det_index) >= 3:
        major = dets.major[np.asarray(seg.det_index)]
        f = np.asarray(frames, float)
        drift = float(np.polyfit(f, major, 1)[0] * (f[-1] - f[0]))
    return {
        "duration_s": duration_s,
        "displacement_um": disp_um,
        "path_um": path_um,
        "speed_um_min": speed,
        "n_detections": len(frames),
        "length_drift_px": drift,
    }


def summarize_tracks(compounds: list[CompoundTrack],
                     dets: DetectionTable | None = None) -> list[dict]:
    """One row per compound track."""
    rows = []
    for c in compounds:
        segs = [_segment_rows(s, dets) for s in c.segments]
        growth_time = sum(s["duration_s"] for s in segs)
        growth_disp = sum(s["displacement_um"] for s in segs)
        speeds = [s["speed_um_min"] for s in segs if np.isfinite(s["speed_um_min"])]
        pos = np.vstack([s.positions() for s in c.segments])
        net = float(np.hypot(*(pos[-1] - pos[0]))) * _PX
        path = float(np.hypot(*np.diff(pos, axis=0).T).sum()) * _PX
        drifts = [s["length_drift_px"] for s in segs if np.isfinite(s["length_drift_px"])]
        rows.append({
            "compound_id": c.compound_id,
            "start_frame": c.start_frame,
            "end_frame": c.end_frame,
            "n_segments": len(c.segments),
            "n_detections": sum(s["n_detections"] for s in segs),
            "n_fgap": c.n_fgap(),
            "n_bgap": c.n_bgap(),
            "growth_time_s": growth_time,
            "growth_displacement_um": growth_disp,
            "growth_speed_um_min": float(np.mean(speeds)) if speeds else float("nan"),
            "net_displacement_um": net,
            "path_length_um": path,
            "straightness": (net / path) if path > 0 else 0.0,
            "motion_class": c.motion_class,
            "mss_slope": c.mss_slope,
            "motion_method": getattr(c, "motion_method", None),
            "length_drift_px": float(np.mean(drifts)) if drifts else float("nan"),
        })
    return rows


def movie_stats(compounds: list[CompoundTrack], dets: DetectionTable | None = None,
                config: Config | None = None) -> dict:
    """Population numbers for one movie, using plusTip's frequency definitions."""
    growth_speeds, growth_life, growth_disp = [], [], []
    f_freq_t, f_freq_l, b_freq_t, b_freq_l = [], [], [], []
    f_inferred, b_inferred, f_dur, b_dur = [], [], [], []
    classes: dict[str, int] = {}

    for c in compounds:
        classes[c.motion_class or "unclassified"] = \
            classes.get(c.motion_class or "unclassified", 0) + 1
        rows = [_segment_rows(s, dets) for s in c.segments]
        for r in rows:
            if np.isfinite(r["speed_um_min"]):
                growth_speeds.append(r["speed_um_min"])
            growth_life.append(r["duration_s"])
            growth_disp.append(r["displacement_um"])
        # a gap is attributed to the growth subtrack that PRECEDES it
        for k, g in enumerate(c.gaps):
            before = rows[k]
            dur_s = g["n_frames"] * _SEC
            p_end = c.segments[k].positions()[-1]
            p_next = c.segments[k + 1].positions()[0]
            gap_um = float(np.hypot(*(p_next - p_end))) * _PX
            if before["duration_s"] > 0:
                target_t = f_freq_t if g["kind"] == GAP_FORWARD else b_freq_t
                target_t.append(1.0 / before["duration_s"])
            if before["displacement_um"] > 0:
                target_l = f_freq_l if g["kind"] == GAP_FORWARD else b_freq_l
                target_l.append(1.0 / before["displacement_um"])
            if dur_s > 0:
                if g["kind"] == GAP_FORWARD:
                    f_inferred.append(gap_um / dur_s * 60.0)
                    f_dur.append(dur_s)
                else:
                    b_inferred.append(gap_um / dur_s * 60.0)
                    b_dur.append(dur_s)

    def _m(v):
        return float(np.mean(v)) if len(v) else float("nan")

    def _sem(v):
        return float(np.std(v) / np.sqrt(len(v))) if len(v) > 1 else float("nan")

    return {
        "n_compound_tracks": len(compounds),
        "n_growth_subtracks": len(growth_life),
        "n_fgap": len(f_freq_t),
        "n_bgap": len(b_freq_t),
        "growth_speed_um_min_mean": _m(growth_speeds),
        "growth_speed_um_min_sem": _sem(growth_speeds),
        "growth_lifetime_s_mean": _m(growth_life),
        "growth_displacement_um_mean": _m(growth_disp),
        "fgap_freq_time_mean": _m(f_freq_t),
        "fgap_freq_time_sem": _sem(f_freq_t),
        "fgap_freq_length_mean": _m(f_freq_l),
        "bgap_freq_time_mean": _m(b_freq_t),
        "bgap_freq_time_sem": _sem(b_freq_t),
        "bgap_freq_length_mean": _m(b_freq_l),
        "fgap_speed_inferred_um_min": _m(f_inferred),
        "bgap_speed_inferred_um_min": _m(b_inferred),
        "fgap_duration_s_mean": _m(f_dur),
        "bgap_duration_s_mean": _m(b_dur),
        "motion_classes": classes,
    }
