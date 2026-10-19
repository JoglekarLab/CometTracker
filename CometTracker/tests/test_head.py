"""The head estimator, on synthetic comets where the answer is known."""
import numpy as np
import pytest

from comet_tracker_v9.head import (MIN_HEAD_PIXELS, blue_weight, evidence_maps,
                                   head_from_rgb, heads_along_track)


def comet(shape, head_yx, direction, length=20.0, sigma=1.5, amp=1.0):
    """Gaussian head with an exponential tail trailing behind it."""
    yy, xx = np.indices(shape).astype(float)
    u = np.asarray(direction, float)
    u = u / np.hypot(*u)
    dy, dx = yy - head_yx[0], xx - head_yx[1]
    along = dy * u[0] + dx * u[1]          # >0 ahead of the head
    across = -dy * u[1] + dx * u[0]
    tail = np.exp(np.clip(along, None, 0) / (length / 3.0))
    tail[along > 0] = np.exp(-(along[along > 0] ** 2) / (2 * sigma ** 2))
    return amp * tail * np.exp(-(across ** 2) / (2 * sigma ** 2))


def scene(speed=2.4, direction=(0.0, 1.0), head=(32.0, 32.0), shape=(64, 64)):
    """RGB where the comet advances by `speed` per frame toward `direction`."""
    u = np.asarray(direction, float); u = u / np.hypot(*u)
    frames = [comet(shape, np.asarray(head) - u * speed * k, direction)
              for k in (2, 1, 0)]                   # t-2, t-1, t
    rgb = np.stack(frames, axis=-1).astype(np.float32)
    mask = frames[2] > 0.05
    return rgb, mask, np.asarray(head)


def test_leading_edge_is_blue_not_red():
    rgb, mask, _ = scene()
    blue, cyan = evidence_maps(rgb)
    assert blue.max() > 0, "a moving comet must produce blue evidence"
    # blue evidence must sit AHEAD of the mask centroid along the direction
    ys, xs = np.nonzero(mask)
    by, bx = np.unravel_index(np.argmax(blue), blue.shape)
    assert bx > xs.mean(), "blue evidence is behind the centroid"


def test_head_lands_near_the_true_head():
    rgb, mask, truth = scene()
    est = head_from_rgb(rgb, mask)
    assert est.ok
    assert np.hypot(*(est.head - truth)) < 4.0, f"head off by {est.head - truth}"


def test_head_is_ahead_of_the_centroid():
    rgb, mask, _ = scene()
    ys, xs = np.nonzero(mask)
    est = head_from_rgb(rgb, mask)
    assert est.head[1] > xs.mean() + 1.0


@pytest.mark.parametrize("direction", [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1)])
def test_head_found_in_every_direction(direction):
    rgb, mask, truth = scene(direction=direction)
    est = head_from_rgb(rgb, mask)
    assert est.ok, f"no head for direction {direction}"
    assert np.hypot(*(est.head - truth)) < 5.0


def test_a_stationary_comet_yields_little_or_no_head():
    """No motion -> no newly occupied region -> no colour evidence."""
    rgb, mask, _ = scene(speed=0.0)
    est = head_from_rgb(rgb, mask)
    assert not est.ok or est.evidence < 0.1


def test_brightening_alone_does_create_a_spurious_head():
    """THE KNOWN CONFOUND, pinned so it cannot be forgotten.

    A stationary comet that BRIGHTENS across the window produces positive blue
    evidence everywhere (B - max(R,G) is proportional to the comet's own
    profile), so the estimator degrades to an intensity-weighted centroid and
    reports a head displaced forward of the mask centroid -- with no motion at
    all.

    Two things bound the damage, and this test pins both:
      * the displacement is SMALLER than a genuinely moving comet's, because it
        lands on the intensity centroid rather than the leading edge;
      * the evidence value is lower.

    What this test deliberately does NOT assert is that evidence separates the
    two on real data. Measured on 3868 real detections, evidence is 0.309
    (median) for directed tracks against 0.243 for immobile ones -- overlapping
    heavily, nothing like the clean synthetic gap. Evidence is a quality signal,
    not a filter for this confound.
    """
    rgb, mask, truth = scene(speed=0.0)
    ys, xs = np.nonzero(mask)
    centre = np.array([ys.mean(), xs.mean()])
    true_offset = float(np.hypot(*(truth - centre)))

    bright = rgb * np.array([0.6, 0.8, 1.0], np.float32)
    spurious = head_from_rgb(bright, mask)
    assert spurious.ok, "brightening does produce a head -- that is the point"
    d_bright = float(np.hypot(*(spurious.head - centre)))

    moving = head_from_rgb(*scene(speed=2.4)[:2])
    d_move = float(np.hypot(*(moving.head - centre)))

    assert d_bright < true_offset, "spurious head should fall short of the tip"
    assert d_move > d_bright, "real motion must displace the head further"
    assert moving.evidence > spurious.evidence


def test_faster_comet_gives_more_evidence():
    slow = head_from_rgb(*scene(speed=1.0)[:2])
    fast = head_from_rgb(*scene(speed=4.0)[:2])
    assert fast.evidence > slow.evidence


def test_blue_weight_prefers_pure_blue_over_cyan():
    pure = np.array([[[0.0, 0.0, 1.0]]], np.float32)
    cyan = np.array([[[0.0, 1.0, 1.0]]], np.float32)
    assert blue_weight(pure)[0, 0] > blue_weight(cyan)[0, 0]
    assert blue_weight(cyan)[0, 0] > 0, "cyan must stay eligible, just weighted less"


def test_empty_mask_returns_no_head():
    rgb, _, _ = scene()
    est = head_from_rgb(rgb, np.zeros(rgb.shape[:2], bool))
    assert not est.ok and est.head is None


def test_min_area_rejects_a_speck():
    rgb, mask, _ = scene()
    tiny = np.zeros_like(mask); tiny[0, 0] = True
    assert not head_from_rgb(rgb, tiny, dilate=0).ok
    assert MIN_HEAD_PIXELS == 7


def test_heads_along_track_reports_missing_rather_than_guessing():
    """A frame with no usable RGB must yield nan, never an invented head."""
    class T:
        frames = [5, 6, 7]
        det_index = [0, 1, 2]
        def positions(self, source="smoothed"):
            return np.array([[10., 10.], [10., 12.], [10., 14.]])
    class D:
        cy = np.array([10., 10., 10.]); cx = np.array([10., 12., 14.])
    labels = np.zeros((10, 32, 32), int)
    th = heads_along_track(T(), D(), {}, labels)     # no RGB for any frame
    assert th.n_missing == 3
    assert np.isnan(th.heads).all()
    assert np.isnan(th.velocity_px_per_frame())


# ── the outlier gate ──────────────────────────────────────────────────────────

from comet_tracker_v9.head import MAX_HEAD_BACKWARD, MAX_HEAD_STEP, TrackHeads


def make_th(heads, frames=None, direction=(0.0, 1.0), centroids=None):
    heads = np.asarray(heads, float)
    frames = list(range(len(heads))) if frames is None else list(frames)
    if centroids is None:                    # centroid track: clean, forward
        centroids = np.stack([np.zeros(len(heads)),
                              np.arange(len(heads)) * 2.4], axis=1)
    u = np.asarray(direction, float); u = u / np.hypot(*u)
    return TrackHeads(frames, heads, np.zeros(len(heads)), 0,
                      centroids=np.asarray(centroids, float), direction=u)


def test_gate_keeps_a_clean_track_untouched():
    heads = [[0.0, 2.4 * k] for k in range(6)]
    g = make_th(heads).gate()
    assert g.valid.all() and g.n_rejected == 0


def test_gate_drops_an_impossible_jump():
    """47 px/frame was the real maximum measured. Nothing biological does that."""
    heads = [[0.0, 0.0], [0.0, 2.4], [0.0, 40.0], [0.0, 7.2], [0.0, 9.6]]
    g = make_th(heads).gate()
    assert not g.valid[2], "the 40 px outlier survived"
    assert g.n_rejected >= 1


def test_backward_gate_is_off_by_default():
    """It must be opt-in. EB3 dissociates over ~2.5-4 s, so at 2 s/frame a comet
    stays visible for a frame or two after growth stops -- gating backward
    motion by default would delete the excursion signal before it is measured.
    Measured: backward runs occur on 0.6-1.0% of tracks, matching noise."""
    # steps +2.4, -1.4, +3.0 -- all within the magnitude gate, so only a
    # direction gate could reject the middle one, and it must not.
    heads = [[0.0, 0.0], [0.0, 2.4], [0.0, 1.0], [0.0, 4.0]]
    assert make_th(heads).gate().valid.all()
    assert MAX_HEAD_BACKWARD is None


def test_backward_gate_works_when_asked_for():
    heads = [[0.0, 0.0], [0.0, 2.4], [0.0, -1.5], [0.0, 3.0]]
    g = make_th(heads).gate(max_backward=2.0)
    assert not g.valid[2]


def test_gate_tolerates_small_backward_jitter():
    """The tolerance is not zero: clipping jitter at zero would bias speed up."""
    heads = [[0.0, 0.0], [0.0, 2.4], [0.0, 1.4], [0.0, 4.0], [0.0, 6.4]]
    g = make_th(heads).gate(max_backward=2.0)
    assert g.valid.all(), "a 1 px wobble is jitter, not a rejection"


def test_gate_drops_the_endpoint_further_from_the_centroid_track():
    """Given a bad step, the wrong point is the one the centroid disagrees with
    -- comparing a head only to its neighbour cannot tell which of the pair
    was at fault."""
    centroids = np.stack([np.zeros(4), np.arange(4) * 2.4], axis=1)
    heads = [[0.0, 0.0], [0.0, 2.4], [0.0, 30.0], [0.0, 7.2]]
    g = make_th(heads, centroids=centroids).gate()
    assert g.valid[1] and not g.valid[2]


def test_gate_is_not_destructive_to_the_original():
    heads = [[0.0, 0.0], [0.0, 40.0], [0.0, 4.8]]
    th = make_th(heads)
    th.gate()
    assert th.valid.all(), "gate() must return a copy, not mutate"


def test_gate_needs_a_direction_to_act():
    th = TrackHeads([0, 1], np.array([[0.0, 0.0], [0.0, 50.0]]),
                    np.zeros(2), 0)
    assert th.gate().valid.all()      # no direction -> no rejection, no crash


def test_gate_defaults_sit_above_the_physical_maximum():
    """EB3 runs 1.3-6.3 px/frame, so the step gate must not cut real growth."""
    assert MAX_HEAD_STEP > 6.3


# ── speed estimator ───────────────────────────────────────────────────────────

from comet_tracker_v9.summarize import fit_speed_px_per_frame


def test_fit_speed_is_exact_on_a_clean_track():
    f = list(range(6))
    p = np.array([[0.0, 2.4 * k] for k in f])
    assert fit_speed_px_per_frame(f, p) == pytest.approx(2.4)


def test_fit_speed_is_not_inflated_by_noise_the_way_path_length_is():
    """The bug this replaced. Path length sums |step|, and a magnitude is always
    positive, so noise can only add to it. A fit averages the noise out.
    Measured on real data: 1.22x inflation on head positions."""
    rng = np.random.default_rng(0)
    f = np.arange(8)
    truth = np.stack([np.zeros(8), 2.4 * f], axis=1)
    noisy = truth + rng.normal(0, 1.5, truth.shape)
    path = np.hypot(*np.diff(noisy, axis=0).T).sum() / (f[-1] - f[0])
    fit = fit_speed_px_per_frame(f, noisy)
    assert path > 2.4 * 1.1, "path length should be visibly inflated here"
    assert abs(fit - 2.4) < abs(path - 2.4), "the fit must be the closer estimate"


def test_fit_speed_handles_degenerate_input():
    assert np.isnan(fit_speed_px_per_frame([3], np.array([[1.0, 1.0]])))
    assert np.isnan(fit_speed_px_per_frame([2, 2], np.array([[0., 0.], [1., 1.]])))
