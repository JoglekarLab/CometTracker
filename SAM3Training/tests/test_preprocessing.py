import numpy as np

from comet_sam3.preprocessing import causal_rgb_pair, temporal_median_background


def test_causal_pair_reuses_identical_underlying_frame_values():
    movie = np.arange(15 * 8 * 8, dtype=np.float32).reshape(15, 8, 8)
    background = temporal_median_background(movie)
    image_t, image_p = causal_rgb_pair(movie, center=7, background=background)
    assert image_t.shape == (8, 8, 3)
    assert image_t.dtype == np.float32
    assert np.array_equal(image_t[..., 1], image_p[..., 0])
    assert np.array_equal(image_t[..., 2], image_p[..., 1])

