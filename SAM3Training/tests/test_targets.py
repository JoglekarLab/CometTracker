import numpy as np

from comet_sam3.targets import gaussian_head, soft_uniform_axis


def test_axis_width_is_uniform_not_tapered():
    line = np.zeros((64, 64), bool)
    line[32, 10:55] = True
    target = soft_uniform_axis(line, width_pixels=3.0)
    assert np.allclose(target[31:34, 15].sum(), target[31:34, 50].sum())
    assert target[32, 10] == 1.0
    assert target[32, 54] == 1.0


def test_head_heatmap_peaks_at_subpixel_neighborhood():
    heatmap = gaussian_head((32, 32), (12.2, 19.7), sigma_pixels=1.5)
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    assert (y, x) == (12, 20)

