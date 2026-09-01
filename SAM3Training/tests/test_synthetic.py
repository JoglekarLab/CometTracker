import inspect
import json
from dataclasses import replace

import numpy as np

from comet_sam3.geometry import transform_d8_image, transform_d8_yx
from comet_sam3.data.synthetic import (
    SyntheticConfig,
    SyntheticPairSource,
    _sample_decay_length,
    _select_lineage_event,
    build_synthetic_pair_sample,
    generate_synthetic_pair,
    synthetic_records,
)


SMALL = SyntheticConfig(
    tile_size=64,
    n_frames=7,
    center_frame=3,
    n_comets=(1, 1),
    frozen_distractors=(0, 0),
    branch_probability=0.0,
    merge_probability=0.0,
    hotspot_probability=0.0,
    transient_blob_probability=0.0,
    microtubule_crowd_probability=0.0,
)


def test_default_tile_is_192_and_pair_is_deterministic():
    assert SyntheticConfig().tile_size == 192
    first = generate_synthetic_pair(17, config=SMALL)
    second = generate_synthetic_pair(17, config=SMALL)
    np.testing.assert_array_equal(first.image_t, second.image_t)
    np.testing.assert_array_equal(first.image_tp1, second.image_tp1)
    assert first.links == second.links
    assert [item.track_id for item in first.instances_t] == [
        item.track_id for item in second.instances_t
    ]


def test_decay_length_uses_agreed_four_component_mixture():
    config = SyntheticConfig().validate()
    assert config.decay_length_bins_pixels == (
        (3.0, 5.0),
        (5.0, 18.0),
        (18.0, 24.0),
        (24.0, 28.0),
    )
    np.testing.assert_allclose(
        config.decay_length_probabilities, (0.05, 0.70, 0.20, 0.05)
    )
    rng = np.random.default_rng(20260830)
    values = np.asarray([_sample_decay_length(rng, config) for _ in range(20_000)])
    observed = np.asarray(
        [
            np.mean((values >= low) & (values < high))
            for low, high in config.decay_length_bins_pixels[:-1]
        ]
        + [
            np.mean(
                (values >= config.decay_length_bins_pixels[-1][0])
                & (values <= config.decay_length_bins_pixels[-1][1])
            )
        ]
    )
    np.testing.assert_allclose(
        observed, config.decay_length_probabilities, atol=0.012
    )


def test_generator_api_accepts_worker_rng_and_tile_size():
    first = generate_synthetic_pair(np.random.default_rng(19), tile_size=72)
    second = generate_synthetic_pair(rng=np.random.default_rng(19), tile_size=72)
    assert first.image_t.shape == (72, 72, 3)
    np.testing.assert_array_equal(first.image_t, second.image_t)
    np.testing.assert_array_equal(first.image_tp1, second.image_tp1)


def test_causal_pair_reuses_identical_normalized_frames():
    sample = generate_synthetic_pair(21, rotation=0, reflect=False, config=SMALL)
    np.testing.assert_array_equal(sample.image_t[..., 1], sample.image_tp1[..., 0])
    np.testing.assert_array_equal(sample.image_t[..., 2], sample.image_tp1[..., 1])


def test_positive_pair_has_persistent_ids_links_and_exact_centerline():
    sample = generate_synthetic_pair(31, rotation=0, reflect=False, config=SMALL)
    assert len(sample.instances_t) == 1
    assert len(sample.instances_tp1) == 1
    before, after = sample.instances_t[0], sample.instances_tp1[0]
    assert before.track_id == after.track_id
    assert sample.links == [(before.track_id, after.track_id)]
    for instance in (before, after):
        assert instance.head_valid and instance.axis_valid
        axis = np.asarray(instance.axis_yx)
        assert axis.ndim == 2 and axis.shape[1] == 2 and len(axis) >= 2
        # Ordered from tail to head; the last raster pixel contains the rounded head.
        np.testing.assert_array_equal(axis[-1], np.rint(instance.head_yx))
        steps = np.abs(np.diff(axis, axis=0))
        assert np.all(steps.max(axis=1) == 1)
        assert instance.metadata["axis_kind"] == "exact_uniform_tail_to_head_centerline"
    assert sample.metadata["axis_tapered"] is False


def test_empty_and_frozen_scenes_are_explicit_exhaustive_negatives():
    for kind in ("empty", "frozen"):
        sample = generate_synthetic_pair(
            41, scene_kind=kind, rotation=0, reflect=False, config=SMALL
        )
        assert sample.instances_t == []
        assert sample.instances_tp1 == []
        assert sample.links == []
        assert sample.exhaustive_t and sample.exhaustive_tp1
        assert sample.metadata["scene_kind"] == kind
        assert sample.metadata["posthoc_noise"] is False
        assert sample.metadata["posthoc_intensity_change"] is False
        assert sample.metadata["posthoc_blur"] is False
        assert sample.metadata["pauses"] is False
    frozen = generate_synthetic_pair(
        41, scene_kind="frozen", rotation=0, reflect=False, config=SMALL
    )
    assert frozen.metadata["n_frozen_distractors"] >= 1


def test_split_transition_has_child_ids_and_masks_link_loss():
    sample = generate_synthetic_pair(
        51,
        force_branch=True,
        split_transition=True,
        rotation=0,
        reflect=False,
        config=SMALL,
    )
    assert len(sample.instances_t) == 1
    assert len(sample.instances_tp1) == 2
    parent = sample.instances_t[0].track_id
    children = {item.track_id for item in sample.instances_tp1}
    assert children == {f"{parent}/a", f"{parent}/b"}
    heads = np.asarray([item.head_yx for item in sample.instances_tp1])
    assert np.linalg.norm(heads[0] - heads[1]) >= 2.5
    assert sample.links == []
    assert sample.metadata["branch_transition"] is True
    assert sample.metadata["link_supervision_valid"] is False


def test_small_angle_branch_uses_total_opening_and_separates_children():
    config = replace(
        SMALL,
        small_branch_fraction=1.0,
        small_branch_angle_degrees=(10.0, 12.0),
    )
    sample = generate_synthetic_pair(
        52,
        force_branch=True,
        split_transition=False,
        rotation=0,
        reflect=False,
        config=config,
    )
    opening = sample.metadata["branch_openings_degrees"]
    assert len(opening) == 1 and 10.0 <= opening[0] <= 12.0
    assert sample.metadata["small_branch_count"] == 1
    heads_t = np.asarray([item.head_yx for item in sample.instances_t])
    heads_tp1 = np.asarray([item.head_yx for item in sample.instances_tp1])
    assert len(heads_t) == len(heads_tp1) == 2
    assert np.linalg.norm(heads_t[0] - heads_t[1]) >= 2.5
    assert np.linalg.norm(heads_tp1[0] - heads_tp1[1]) > np.linalg.norm(
        heads_t[0] - heads_t[1]
    )
    assert sample.metadata["link_supervision_valid"] is True
    assert sample.metadata["link_exhaustive"] is True
    assert len(sample.links) == 2


def test_merge_transition_is_five_percent_variant_with_ambiguous_link_masked():
    assert SyntheticConfig().merge_probability == 0.05
    sample = generate_synthetic_pair(
        53,
        force_merge=True,
        rotation=0,
        reflect=False,
        config=SMALL,
    )
    assert len(sample.instances_t) == 2
    assert len(sample.instances_tp1) == 1
    assert sample.metadata["merge_event"] is True
    assert sample.metadata["merge_transition"] is True
    assert sample.metadata["link_supervision_valid"] is False
    assert sample.metadata["link_exhaustive"] is False
    assert sample.metadata["ambiguous_lineage_event"] == "merge"
    assert sample.links == []
    heads_t = np.asarray([item.head_yx for item in sample.instances_t])
    assert np.linalg.norm(heads_t[0] - heads_t[1]) >= 3.0
    child = sample.instances_tp1[0]
    assert child.metadata["event"] == "post_merge_child"
    assert set(child.metadata["parent_ids"]) == {
        item.track_id for item in sample.instances_t
    }


def test_merge_and_branch_event_intervals_have_exact_configured_widths():
    config = SyntheticConfig()
    assert _select_lineage_event(0.0, config) == "merge"
    assert _select_lineage_event(0.049999, config) == "merge"
    assert _select_lineage_event(0.05, config) == "branch"
    assert _select_lineage_event(0.129999, config) == "branch"
    assert _select_lineage_event(0.13, config) is None


def test_dynamic_blob_and_microtubule_backgrounds_are_unlabelled_negatives():
    config = replace(
        SMALL,
        transient_blob_probability=1.0,
        transient_blob_count=(2, 2),
        microtubule_crowd_probability=1.0,
        microtubule_count=(4, 4),
        microtubule_length_pixels=(18.0, 35.0),
    )
    sample = generate_synthetic_pair(
        54,
        scene_kind="empty",
        rotation=0,
        reflect=False,
        config=config,
    )
    assert sample.instances_t == [] and sample.instances_tp1 == []
    assert sample.links == [] and sample.exhaustive_t and sample.exhaustive_tp1
    assert sample.metadata["background_kind"] == "dynamic_analytic"
    assert sample.metadata["n_background_blobs"] == 2
    assert sample.metadata["n_background_microtubules"] == 4
    assert sample.metadata["background_mean_abs_frame_change"] > 0.0
    assert not np.array_equal(sample.image_t[..., 0], sample.image_t[..., 1])


def test_background_feature_switches_do_not_change_object_geometry_or_links():
    plain_config = replace(
        SMALL,
        transient_blob_probability=0.0,
        microtubule_crowd_probability=0.0,
    )
    hard_config = replace(
        SMALL,
        transient_blob_probability=1.0,
        transient_blob_count=(2, 2),
        microtubule_crowd_probability=1.0,
        microtubule_count=(4, 4),
        microtubule_length_pixels=(18.0, 35.0),
    )
    plain = generate_synthetic_pair(55, rotation=0, reflect=False, config=plain_config)
    hard = generate_synthetic_pair(55, rotation=0, reflect=False, config=hard_config)
    assert plain.links == hard.links
    for first, second in zip(plain.instances_t + plain.instances_tp1, hard.instances_t + hard.instances_tp1):
        assert first.track_id == second.track_id
        np.testing.assert_array_equal(first.axis_yx, second.axis_yx)
        np.testing.assert_allclose(first.head_yx, second.head_yx)
    assert not np.array_equal(plain.image_t, hard.image_t)


def test_background_renderer_has_no_forbidden_posthoc_filter_or_noise_calls():
    import comet_sam3.data.synthetic as module

    source = inspect.getsource(module)
    assert "gaussian_filter" not in source
    assert ".poisson(" not in source
    assert "posthoc_noise\": True" not in source
    assert "posthoc_blur\": True" not in source


def test_one_d8_transform_is_shared_by_both_images_and_all_geometry():
    plain = generate_synthetic_pair(61, rotation=0, reflect=False, config=SMALL)
    changed = generate_synthetic_pair(61, rotation=1, reflect=True, config=SMALL)
    np.testing.assert_array_equal(
        changed.image_t, transform_d8_image(plain.image_t, 1, True)
    )
    np.testing.assert_array_equal(
        changed.image_tp1, transform_d8_image(plain.image_tp1, 1, True)
    )
    for original, transformed in zip(plain.instances_t, changed.instances_t):
        np.testing.assert_allclose(
            transformed.head_yx, transform_d8_yx(original.head_yx, 64, 1, True)
        )
        np.testing.assert_array_equal(
            transformed.axis_yx, transform_d8_yx(original.axis_yx, 64, 1, True)
        )


def test_manifest_recipes_are_json_serializable_and_materialize_on_access():
    records = synthetic_records(12, seed=71)
    json.dumps(records)
    assert records[0]["scene_kind"] == "empty"
    assert records[1]["scene_kind"] == "frozen"
    assert records[2]["scene_kind"] == "positive"
    sample = build_synthetic_pair_sample(records[2], config=SMALL)
    assert sample.sample_id == records[2]["sample_id"]
    source = SyntheticPairSource(records=records[2:4], config=SMALL)
    assert len(source) == 2
    assert source[0].source == "procedural"
