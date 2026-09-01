"""Detections -> segments -> compound tracks -> numbers."""
from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .detections import DetectionTable, from_prediction_folder
from .gapclose import break_nonlinear, close_gaps
from .link import run_linking
from .motion import classify_all
from .summarize import movie_stats, summarize_tracks

__all__ = ["Result", "run"]


@dataclass
class Result:
    detections: DetectionTable
    segments: list
    compounds: list
    tracks: list
    stats: dict


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
