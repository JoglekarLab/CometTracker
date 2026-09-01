"""Masks in, instance table out.

THE INTERFACE, AND WHY IT IS A TABLE AND NOT PIXELS

The tracker never sees a mask. It sees one row per detected comet::

    frame  cy  cx  theta  sigma_theta  major  minor  area  score  det_id

Two reasons. First, everything the linker needs from a mask is in its second
moments -- position, axis, and how much to trust the axis -- so carrying pixels
past this point would be carrying weight for nothing. Second, it makes the mask
source swappable. Today the rows are built from ``_labels.tif``, which is
connected components of SAM3's COLLAPSED probability map. If per-query instance
masks are exported later, only this module changes.

WHAT ``score`` IS, AND THE BUG IT INHERITS

``score`` is the component's peak value in ``_prob.tif``. That file holds
``max over queries of presence(q) * P_axis(q, pixel)``, and ``axis_peak`` is
>= 0.99 for 99% of queries, so the peak of a component is very close to the
presence of the best query on it. Good enough to use as a detection confidence.

But the same product is what BUILT the components, and that is a real defect
this module cannot repair:

  * a query with presence <= 0.5 can never reach the 0.5 threshold whatever its
    mask says, so ~31% of SAM3's detections are simply absent from labels.tif;
  * for the survivors the mask contour is a joint iso-contour, so the tube's
    ENDS erode as presence falls. Measured across presence bins the minor axis
    is flat (4.79 -> 5.06 px) while the major nearly doubles (9.36 -> 17.56 px).

So ``major`` here is partly a readout of detector confidence, not comet length.
V7 therefore uses ``major`` only for sigma_theta (a ratio, where the bias
largely cancels) and never as a length measurement or a filter state. Comet
length as a scientific output has to come from the image, not from a mask that
is a fixed-width centreline annotation to begin with (see
``comet_sam3/targets.py::soft_uniform_axis`` -- "deliberately no taper").

The fix belongs upstream: threshold ``P_axis`` alone and carry ``presence`` as a
column. ``score`` is already that column, so when the export is fixed this
module keeps working and simply gets better inputs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .config import DetectConfig
from .geometry import orientation_sigma

FIELDS = ("frame", "cy", "cx", "theta", "sigma_theta",
          "major", "minor", "area", "score", "det_id")


@dataclass
class DetectionTable:
    """Column arrays, plus a per-frame index. All arrays share a length."""

    frame: np.ndarray
    cy: np.ndarray
    cx: np.ndarray
    theta: np.ndarray
    sigma_theta: np.ndarray
    major: np.ndarray
    minor: np.ndarray
    area: np.ndarray
    score: np.ndarray
    det_id: np.ndarray

    def __len__(self) -> int:
        return int(self.frame.size)

    @property
    def xy(self) -> np.ndarray:
        """(n, 2) positions in (row, col)."""
        return np.stack([self.cy, self.cx], axis=1)

    def indices_by_frame(self) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        order = np.argsort(self.frame, kind="stable")
        f = self.frame[order]
        if f.size == 0:
            return out
        bounds = np.flatnonzero(np.diff(f)) + 1
        for chunk in np.split(order, bounds):
            out[int(self.frame[chunk[0]])] = chunk
        return out

    def frame_range(self) -> range:
        if len(self) == 0:
            return range(0)
        return range(int(self.frame.min()), int(self.frame.max()) + 1)

    def subset(self, idx: np.ndarray) -> "DetectionTable":
        idx = np.asarray(idx)
        return DetectionTable(**{k: getattr(self, k)[idx] for k in FIELDS})

    def to_rows(self) -> list[dict]:
        return [dict(zip(FIELDS, vals)) for vals in
                zip(*(getattr(self, k) for k in FIELDS))]


def from_label_stack(
    labels: np.ndarray,
    prob: np.ndarray | None = None,
    config: DetectConfig | None = None,
) -> DetectionTable:
    """Build the instance table from a (T, Y, X) label stack.

    ``labels`` is renumbered per frame (1..n), which is what sam3_export and
    ilastik_export both write. ``prob`` is the matching probability stack, used
    only to fill ``score``; uint8 0-255 is rescaled to 0-1. Without it every
    score is 1.0, which makes the presence terms in the cost matrix inert
    rather than wrong.
    """
    from skimage.measure import regionprops

    cfg = config or DetectConfig()
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError("labels must be (T, Y, X)")
    if prob is not None:
        prob = np.asarray(prob)
        if prob.shape != labels.shape:
            raise ValueError("prob and labels must have the same shape")
        if prob.dtype == np.uint8:
            prob = prob.astype(np.float32) / 255.0

    cols: dict[str, list] = {k: [] for k in FIELDS}
    next_id = 0
    for t in range(labels.shape[0]):
        frame_labels = labels[t]
        if not frame_labels.any():
            continue
        intensity = prob[t] if prob is not None else None
        for p in regionprops(frame_labels, intensity_image=intensity):
            if p.area < cfg.min_area:
                continue
            score = 1.0 if intensity is None else float(p.intensity_max)
            if score < cfg.min_score:
                continue
            cy, cx = p.centroid
            cols["frame"].append(t)
            cols["cy"].append(float(cy))
            cols["cx"].append(float(cx))
            cols["theta"].append(float(p.orientation))
            cols["sigma_theta"].append(
                orientation_sigma(p.axis_major_length, p.axis_minor_length, p.area))
            cols["major"].append(float(p.axis_major_length))
            cols["minor"].append(float(p.axis_minor_length))
            cols["area"].append(int(p.area))
            cols["score"].append(score)
            cols["det_id"].append(next_id)
            next_id += 1

    dtypes = {"frame": np.int64, "area": np.int64, "det_id": np.int64}
    return DetectionTable(**{
        k: np.asarray(v, dtype=dtypes.get(k, np.float64)) for k, v in cols.items()})


def from_prediction_folder(
    folder: str, stem: str, config: DetectConfig | None = None
) -> DetectionTable:
    """Read ``<stem>_labels.tif`` (+ ``_prob.tif`` if present) from a folder.

    This is the layout ``sam3_export.py`` writes and the ilastik/U-Net arms
    share, so V7 reads all three detectors' output without a special case.
    """
    import tifffile

    label_path = os.path.join(folder, stem + "_labels.tif")
    if not os.path.exists(label_path):
        raise FileNotFoundError(label_path)
    prob_path = os.path.join(folder, stem + "_prob.tif")
    prob = tifffile.imread(prob_path) if os.path.exists(prob_path) else None
    return from_label_stack(tifffile.imread(label_path), prob, config)
