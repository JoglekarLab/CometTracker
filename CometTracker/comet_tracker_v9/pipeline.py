"""Detections -> segments -> compound tracks -> numbers.

Heads are a SEPARATE, LATER pass (``attach_heads``). Linking never sees them:
the mask centroid is the steadier measurement (0.47 px RMS around a
constant-velocity fit against the head's 1.04 px) and V7's gates are tuned
around it. The head is measured afterwards, along tracks that are already
settled, and is used for VELOCITY -- where the centroid has a systematic
problem the head does not. See ``head.py`` for the numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .detections import DetectionTable, from_prediction_folder
from .gapclose import break_nonlinear, close_gaps
from .link import run_linking
from .motion import classify_all
from .head import heads_along_track
from .summarize import movie_stats, summarize_tracks

__all__ = ["Result", "run"]


@dataclass
class Result:
    detections: DetectionTable
    segments: list
    compounds: list
    tracks: list
    stats: dict
    heads: dict | None = None      # track_id -> TrackHeads, once attached


def run(dets: DetectionTable, config: Config | None = None) -> Result:
    cfg = config or Config()
    segments = run_linking(dets, cfg)
    segments = break_nonlinear(segments, cfg)
    compounds = close_gaps(segments, cfg)
    classify_all(compounds, cfg)
    return Result(
        detections=dets,
        segments=segments,
        compounds=compounds,
        tracks=summarize_tracks(compounds, dets),
        stats=movie_stats(compounds, dets, cfg),
    )


def run_folder(folder: str, stem: str, config: Config | None = None) -> Result:
    cfg = config or Config()
    return run(from_prediction_folder(folder, stem, cfg.detect), cfg)


def attach_heads(result: Result, movie, labels, background=None) -> Result:
    """Measure the plus-end along every already-linked segment. Post-hoc only.

    This CANNOT change any link. It walks the segments the linker already
    produced and adds a head estimate per frame, from the colour of the causal
    pseudo-RGB (see head.py).

    Parameters
    ----------
    movie      : (T, Y, X) raw frames, as read from the .nd2.
    labels     : (T, Y, X) the same label stack the detections came from.
    background : temporal-median background; computed if omitted.

    Requires ``comet_sam3.preprocessing`` on the path, because the RGB must be
    built exactly as the model saw it -- same background blend, same joint
    normalisation. Rebuilding it a different way would change what "blue"
    means.
    """
    from comet_sam3.preprocessing import (causal_rgb_pair,
                                          temporal_median_background)
    import numpy as np

    movie = np.asarray(movie, np.float32)
    if background is None:
        background = temporal_median_background(movie)
    rgb = {f: causal_rgb_pair(movie, f, background=background)[0]
           for f in range(2, len(movie) - 1)}

    result.heads = {seg.track_id: heads_along_track(seg, result.detections,
                                                    rgb, labels)
                    for seg in result.segments}
    return result
