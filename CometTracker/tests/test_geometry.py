"""The axis convention is pinned here. If this file fails, nothing else matters."""
import numpy as np
import pytest
from skimage.measure import label, regionprops

from comet_tracker_v7 import geometry as g


def _bar(angle_deg, half_len=40, half_wid=2, size=201):
    """A bar through the centre, elongated along a KNOWN (row, col) direction."""
    th = np.radians(angle_deg)
    ur, uc = np.sin(th), np.cos(th)
    img = np.zeros((size, size), np.uint8)
    for s in np.linspace(-half_len, half_len, 8 * half_len):
        for w in np.linspace(-half_wid, half_wid, 4 * half_wid + 1):
            img[int(round(size // 2 + s * ur - w * uc)),
                int(round(size // 2 + s * uc + w * ur))] = 1
    return img, np.array([ur, uc])


@pytest.mark.parametrize("deg", [0, 15, 30, 45, 60, 75, 90, 120, 150, 170])
def test_axis_unit_matches_regionprops(deg):
    """(cos theta, sin theta) in (row, col) recovers the true bar direction.

    This is THE test. During development ``(-sin, cos)`` was used instead, which
    is 90 degrees off, and it made the mask axis look perpendicular to the
    direction of travel (median 84.7 deg) -- inverting the corridor gate into a
    filter that keeps only impossible links.
    """
    img, truth = _bar(deg)
    prop = regionprops(label(img))[0]
    d = g.axis_unit(prop.orientation)
    # undirected: the axis is a line, so |cos| is the right comparison
    err = np.degrees(np.arccos(min(1.0, abs(float(d @ truth)))))
    assert err < 1.0, f"{deg} deg: recovered axis is {err:.2f} deg off"


def test_axis_normal_is_perpendicular_and_unit():
    for th in np.linspace(-np.pi / 2, np.pi / 2, 17):
        u, n = g.axis_unit(th), g.axis_normal(th)
        assert abs(float(u @ n)) < 1e-12
        assert abs(np.linalg.norm(u) - 1) < 1e-12
        assert abs(np.linalg.norm(n) - 1) < 1e-12


def test_decompose_recovers_pure_along_and_across():
    th = np.radians(37.0)
    u, n = g.axis_unit(th), g.axis_normal(th)
    p, q = g.decompose(3.0 * u, th)
    assert p == pytest.approx(3.0) and q == pytest.approx(0.0, abs=1e-12)
    p, q = g.decompose(2.0 * n, th)
    assert p == pytest.approx(0.0, abs=1e-12) and q == pytest.approx(2.0)


def test_decompose_is_vectorised():
    th = np.array([0.0, np.pi / 2])
    steps = np.array([[1.0, 0.0], [1.0, 0.0]])
    p, q = g.decompose(steps, th)
    assert p.shape == (2,) and q.shape == (2,)
    assert p[0] == pytest.approx(1.0)   # axis along rows
    assert q[1] == pytest.approx(-1.0)  # axis along cols -> step is across


def test_undirected_angle_folds_at_90():
    assert g.undirected_angle(0.0, np.pi) == pytest.approx(0.0)
    assert np.degrees(g.undirected_angle(0.0, np.radians(170))) == pytest.approx(10.0)
    assert np.degrees(g.undirected_angle(0.0, np.radians(90))) == pytest.approx(90.0)


def test_axis_vs_vector_rejects_zero_length():
    """A zero step must gate OUT, not produce nan that slips through a <= test."""
    a = g.axis_vs_vector(0.0, np.array([0.0, 0.0]))
    assert np.isfinite(a) and a == pytest.approx(np.pi / 2)
    assert not (a <= np.radians(30))


def test_directed_angle_keeps_the_sign_gap_closing_needs():
    """Forward and backward must NOT fold together -- pause vs shrinkage."""
    fwd = np.array([1.0, 0.0])
    assert np.degrees(g.directed_angle(fwd, fwd)) == pytest.approx(0.0)
    assert np.degrees(g.directed_angle(fwd, -fwd)) == pytest.approx(180.0)
    # the undirected version deliberately loses exactly this distinction
    assert np.degrees(g.undirected_angle(0.0, np.pi)) == pytest.approx(0.0)


def test_orientation_sigma_shrinks_with_elongation():
    thin = g.orientation_sigma(major=30.0, minor=5.0, area=100)
    fat = g.orientation_sigma(major=8.0, minor=6.0, area=100)
    assert thin < fat
    assert g.orientation_sigma(10.0, 10.0, 100) == pytest.approx(np.pi / 2)
    assert g.orientation_sigma(30.0, 5.0, 400) < g.orientation_sigma(30.0, 5.0, 100)
    assert 0 < thin < np.pi / 2


def test_orientation_sigma_on_the_real_median_mask():
    """Median real component: major 14.1, minor 5.0, area 54."""
    s = g.orientation_sigma(14.1, 5.0, 54)
    assert np.degrees(s) < 15.0, f"median mask axis sigma is {np.degrees(s):.1f} deg"
