#!/usr/bin/env python3
"""Materialize one sample per source and verify manifest/data contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comet_sam3.config import load_config  # noqa: E402
from comet_sam3.data.current_annotations import build_current_pair_sample  # noqa: E402
from comet_sam3.data.sources import TrainingSources  # noqa: E402
from comet_sam3.data.synthetic import generate_synthetic_pair  # noqa: E402
from comet_sam3.data.unet_masks import (  # noqa: E402
    augment_unet_record,
    build_unet_pair_sample,
)


def _sample_summary(sample) -> dict:
    if not np.array_equal(sample.image_t[..., 1], sample.image_tp1[..., 0]):
        raise ValueError(f"causal overlap channel 1/0 differs for {sample.sample_id}")
    if not np.array_equal(sample.image_t[..., 2], sample.image_tp1[..., 1]):
        raise ValueError(f"causal overlap channel 2/1 differs for {sample.sample_id}")
    sample.validate()
    return {
        "sample_id": sample.sample_id,
        "source": sample.source,
        "shape": list(sample.image_t.shape),
        "range": [float(sample.image_t.min()), float(sample.image_t.max())],
        "instances_t": len(sample.instances_t),
        "instances_tp1": len(sample.instances_tp1),
        "heads_t": sum(item.head_valid for item in sample.instances_t),
        "axes_t": sum(item.axis_valid for item in sample.instances_t),
        "positive_links": len(sample.links),
        "exhaustive_t": sample.exhaustive_t,
        "certified_regions_t": len(
            sample.metadata.get("certified_background_regions_t", [])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    sources = TrainingSources(config)
    root = Path(config["paths"]["project_root"])

    # Fail before a GPU allocation if any manifest path is stale on the cluster.
    path_keys = (
        "movie_path",
        "source_movie",
        "source_mask",
        "donor_movie",
        "donor_background",
    )
    missing = []
    for record in [*sources.train_records, *sources.val_records]:
        for key in path_keys:
            value = record.get(key)
            if (
                key == "source_movie"
                and isinstance(value, str)
                and Path(value).suffix.lower() not in {".nd2", ".tif", ".tiff"}
            ):
                continue
            if isinstance(value, str) and value and not (root / value).is_file():
                missing.append(f"{record['sample_id']}:{key}={value}")
    if missing:
        raise FileNotFoundError(f"manifest references missing files: {missing[:10]}")

    current_positive = build_current_pair_sample(sources.current_positive[0], root)
    current_background = build_current_pair_sample(sources.current_background[0], root)
    unet_record = sources.unet_train[0]
    if not unet_record.get("augmented", False):
        unet_record = augment_unet_record(
            unet_record, root, rotation=0, reflect=False, seed=17
        )
    old = build_unet_pair_sample(
        unet_record,
        root,
        background_blend=float(config["input"]["background_blend"]),
    )
    procedural = generate_synthetic_pair(
        int(config["campaign"]["seed"]),
        tile_size=int(config["input"]["source_tile_size"]),
    )
    if old.instances_t[0].head_valid:
        raise ValueError("old mask source must not invent head supervision")
    if not current_background.metadata.get("certified_background_regions_t"):
        raise ValueError("current background sample lost its certified rectangle")

    leakage_path = Path(config["paths"]["manifest_dir"]) / "leakage_audit.json"
    leakage = json.loads(leakage_path.read_text())
    if leakage.get("status") != "passed":
        raise ValueError("leakage audit did not pass")
    report = {
        "status": "passed",
        "manifest_inventory": sources.inventory(),
        "leakage_audit": leakage,
        "materialized_examples": {
            "procedural": _sample_summary(procedural),
            "old_unet_paste": _sample_summary(old),
            "current_positive": _sample_summary(current_positive),
            "current_partial_background": _sample_summary(current_background),
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
