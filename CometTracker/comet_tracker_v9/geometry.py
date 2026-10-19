"""Axis arithmetic.

Everything in V7 that involves a direction goes through this module, because
the one convention mistake that is easy to make here is silent and total.

THE CONVENTION, AND WHY IT IS PINNED BY A TEST

``skimage.measure.regionprops(...).orientation`` is documented as "the angle
between the 0th axis (rows) and the major axis". That sentence admits at least
five plausible unit vectors, and four of them are wrong. The correct one, in
(row, col) order, is::

    d = (cos(theta), sin(theta))

``tests/test_geometry.py::test_axis_unit_matches_regionprops`` builds synthetic
bars at known angles, runs regionprops over them, and asserts this formula
recovers the known direction to under a degree. During development the plausible
alternative ``(-sin, cos)`` was used instead, which is exactly 90 degrees off;
it made the mask axis look PERPENDICULAR to the direction of travel (median
84.7 deg) and would have inverted the corridor gate into a filter that keeps
only impossible links. The test exists so that cannot recur silently.

SIGN

``theta`` describes a LINE, not an arrow: theta and theta+pi are the same axis,
and regionprops returns a value in [-pi/2, pi/2] with no way to tell head from
tail. Every function here that compares an axis to something else therefore
folds its answer into [0, pi/2] -- an "undirected" angle. Recovering the arrow
is the tracker's job, from motion (see ``link.py``), not geometry's.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "axis_unit", "axis_normal", "decompose", "undirected_angle",
    "axis_vs_vector", "directed_angle", "orientation_sigma",
]


def axis_unit(theta: float | np.ndarray) -> np.ndarray:
    """Unit vector along the major axis, in (row, col) order.

    Pinned to ``regionprops.orientation`` by test_geometry. Accepts a scalar
    (returns shape (2,)) or an array (returns shape (n, 2)).
    """
    theta = np.asarray(theta, dtype=np.float64)
    out = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    return out


def axis_normal(theta: float | np.ndarray) -> np.ndarray:
    """Unit vector perpendicular to the major axis, in (row, col) order."""
    theta = np.asarray(theta, dtype=np.float64)
    return np.stack([-np.sin(theta), np.cos(theta)], axis=-1)


def decompose(step: np.ndarray, theta: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a displacement into (along, across) an axis.

    Parameters
    ----------
    step : (..., 2) displacement in (row, col).
    theta : matching orientation(s), radians.

    Returns
    -------
    (p, q) : along-axis and across-axis components, both SIGNED.

    ``q`` is the quantity the corridor gate bounds. Measured on real SAM3
    predictions over 247 clean tracks, |q| for a true single-frame step has
    median 0.17 px, p90 0.52 px, p99 2.16 px -- which is why V7's corridor can
    be far tighter than the 4.92 px V6 needed.
    """
    step = np.asarray(step, dtype=np.float64)
    u = axis_unit(theta)
    n = axis_normal(theta)
    return (step * u).sum(-1), (step * n).sum(-1)


def undirected_angle(theta_a, theta_b) -> np.ndarray:
    """Angle between two AXES, folded into [0, pi/2]."""
    d = np.abs(np.asarray(theta_a, np.float64) - np.asarray(theta_b, np.float64))
    d = np.mod(d, np.pi)
    return np.minimum(d, np.pi - d)


def axis_vs_vector(theta, vec: np.ndarray) -> np.ndarray:
    """Angle between an axis (a line) and a vector, folded into [0, pi/2].

    A zero-length vector has no direction; it returns pi/2 (maximally
    disagreeing) rather than nan, so callers gating on "angle <= limit" reject
    it instead of silently letting a nan through the comparison.
    """
    vec = np.asarray(vec, np.float64)
    norm = np.linalg.norm(vec, axis=-1)
    u = axis_unit(theta)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.abs((vec * u).sum(-1)) / norm
    c = np.where(norm > 0, np.clip(c, 0.0, 1.0), 0.0)
    return np.arccos(c)


def directed_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between two VECTORS, in [0, pi]. Direction matters here.

    This is the one used for gap closing, where "ahead, same way" (a pause)
    and "behind, back down the line" (a shrinkage) are different events and
    must not be folded together.
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (a * b).sum(-1) / (na * nb)
    c = np.where((na > 0) & (nb > 0), np.clip(c, -1.0, 1.0), 1.0)
    return np.arccos(c)


def orientation_sigma(major: float, minor: float, area: float) -> float:
    """Standard error of the principal-axis angle, in radians.

    For N points with principal variances l1 > l2, the asymptotic variance of
    the estimated axis angle is ``l1*l2 / (N * (l1-l2)^2)``. skimage reports
    ``axis_length = 4*sqrt(l)``, so ``l = (length/4)^2``.

    The formula does what is wanted at both extremes: a long thin mask gets a
    small sigma, and a round one (l1 -> l2) diverges. It is capped at pi/2,
    which is the largest an undirected angle error can be.

    CAVEAT: the derivation assumes N independent Gaussian samples. Mask pixels
    are neither independent nor Gaussian, so treat this as a well-behaved
    monotone proxy for axis reliability, not a calibrated error bar. It is used
    to WIDEN the corridor for round masks, so being approximate makes the gate
    conservative rather than wrong.
    """
    l1 = (float(major) / 4.0) ** 2
    l2 = (float(minor) / 4.0) ** 2
    n = max(float(area), 1.0)
    if l1 <= l2:
        return float(np.pi / 2)
    var = (l1 * l2) / (n * (l1 - l2) ** 2)
    return float(min(np.sqrt(max(var, 0.0)), np.pi / 2))
