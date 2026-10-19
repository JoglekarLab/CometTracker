"""Scale and channel: the two things the .nd2 metadata gets wrong."""
import numpy as np
import pytest

from comet_tracker_v9 import postprocess as P
from comet_tracker_v9 import summarize as S
from comet_tracker_v9.movie import MICRONS_PER_PIXEL_1X, ZoomTable


def _table(tmp_path, body):
    p = tmp_path / "z.csv"
    p.write_text(body)
    return str(p)


def test_default_is_1p5x_without_a_table():
    z, px, src = ZoomTable().lookup("anything_001")
    assert (z, src) == (1.5, "default")
    assert px == pytest.approx(0.105917, abs=1e-6)


def test_table_marks_1x_and_rest_default(tmp_path):
    zt = ZoomTable(_table(tmp_path,
        "# comment line\nfolder,stem,zoom,pixel_um\nA,m_001,1,0.158876\nA,m_002,1.5,0.105917\n"))
    assert zt.lookup("m_001") == (1.0, MICRONS_PER_PIXEL_1X, "table")
    assert zt.lookup("m_002")[:2] == (1.5, MICRONS_PER_PIXEL_1X / 1.5)
    assert zt.lookup("not_listed")[2] == "default"
    assert (zt.n_at(1.0), zt.n_at(1.5)) == (1, 1)


def test_stem_with_dot_is_kept_whole(tmp_path):
    zt = ZoomTable(_table(tmp_path, "stem,zoom\n20261007_EB3GW16+0.25DOX_001,1\n"))
    assert zt.lookup("20261007_EB3GW16+0.25DOX_001")[0] == 1.0


def test_rejects_other_zooms_and_missing_columns(tmp_path):
    with pytest.raises(ValueError):
        ZoomTable(_table(tmp_path, "stem,zoom\nm,2\n"))
    with pytest.raises(ValueError):
        ZoomTable(_table(tmp_path, "name,z\nm,1\n"))
    with pytest.raises(FileNotFoundError):
        ZoomTable(str(tmp_path / "nope.csv"))


def test_set_scale_moves_both_modules_and_restores():
    base = (P._PX, P._SEC, S._PX, S._SEC)
    old = P._set_scale(0.1, 2.0)
    try:
        assert (P._PX, S._PX) == (0.1, 0.1)
        assert P._UM_MIN == pytest.approx(0.1 / 2.0 * 60)
    finally:
        P._set_scale(*old)
    assert (P._PX, P._SEC, S._PX, S._SEC) == base


def test_photometry_refuses_a_two_channel_stack():
    from comet_tracker_v9.photometry import measure
    mov = np.zeros((3, 2, 16, 16), np.float32)       # (T, C, Y, X) as nd2 gives it
    lab = np.zeros((3, 16, 16), np.int32)
    with pytest.raises(ValueError, match="T, Y, X"):
        measure(mov, lab, None)


def test_splits_refuses_a_two_channel_stack():
    from comet_tracker_v9.splits import detect
    with pytest.raises(ValueError, match="T, Y, X"):
        detect(np.zeros((3, 16, 16), np.int32), np.zeros((3, 2, 16, 16)))
