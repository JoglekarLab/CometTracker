#!/usr/bin/env python3
"""Build portable train/validation source-descriptor manifests.

The JSONL files contain recipes, not rendered arrays.  Causal RGB pairs and
lossless geometry augmentation are materialized at training time.  This keeps
the source movie provenance auditable and avoids duplicating large images.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


SAM3_TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(SAM3_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_TRAINING_ROOT))

from comet_sam3.config import load_config  # noqa: E402
from comet_sam3.data.current_annotations import (  # noqa: E402
    DEFAULT_VALIDATION_MOVIES,
    load_current_records,
)


def _record_movies(record: dict[str, Any]) -> set[str]:
    """Return every real movie used as signal or background by a recipe."""
    def identity(value: str) -> str:
        path = Path(value)
        return path.stem if path.suffix.lower() in {".nd2", ".tif", ".tiff"} else value

    movies: set[str] = set()
    for key in ("source_movie", "background_movie", "donor_movie"):
        value = record.get(key)
        if isinstance(value, str) and value:
            movies.add(identity(value))
    roles = record.get("movie_roles", {})
    if isinstance(roles, dict):
        for value in roles.values():
            if isinstance(value, str) and value:
                movies.add(identity(value))
            elif isinstance(value, (list, tuple)):
                movies.update(identity(str(item)) for item in value if item)
    return movies


def _portable_paths(value: Any, project_root: Path, key: str = "") -> Any:
    if isinstance(value, dict):
        return {
            child_key: _portable_paths(child, project_root, child_key)
            for child_key, child in value.items()
        }
    if isinstance(value, list):
        return [_portable_paths(child, project_root, key) for child in value]
    if isinstance(value, str) and (key.endswith("_path") or key.endswith("_file")):
        path = Path(value).expanduser()
        if path.is_absolute():
            try:
                return path.resolve().relative_to(project_root).as_posix()
            except ValueError as error:
                raise ValueError(f"manifest path is outside project root: {path}") from error
        normalized = Path(os.path.normpath(value))
        if normalized.parts and normalized.parts[0] == "..":
            raise ValueError(f"manifest path escapes project root: {value}")
        return normalized.as_posix()
    return value


def _is_background_record(record: dict[str, Any]) -> bool:
    kind = str(record.get("kind", "")).lower()
    return "background" in kind and "positive" not in kind


def _is_pajv101(movie: str) -> bool:
    return "pAJV101" in movie


def _movie_split_map(project_root: str | Path, validation_movies: Iterable[str]) -> dict[str, str]:
    """Map every primary Data movie path and stem to the frozen movie split."""
    root = Path(project_root).expanduser().resolve()
    held_out = set(map(str, validation_movies))
    mapping: dict[str, str] = {}
    skip_suffixes = ("_comet_masks", "_background", "_labels", "_prob", "_mask")
    for path in sorted((root / "Data").glob("*/*")):
        if not path.is_file() or path.suffix.lower() not in {".nd2", ".tif", ".tiff"}:
            continue
        if path.stem.endswith(skip_suffixes):
            continue
        split = "val" if path.stem in held_out else "train"
        relative = path.relative_to(root).as_posix()
        mapping[relative] = split
        mapping[path.name] = split
        mapping[path.stem] = split
    return mapping


def audit_leakage(
    train_records: Sequence[dict[str, Any]],
    val_records: Sequence[dict[str, Any]],
    *,
    validation_movies: Iterable[str] = DEFAULT_VALIDATION_MOVIES,
    pajv101_max_background_fraction: float = 0.15,
    require_all_validation_movies: bool = True,
) -> dict[str, Any]:
    """Raise on real-movie leakage or pAJV101 policy violations."""
    held_out = frozenset(map(str, validation_movies))
    if held_out != DEFAULT_VALIDATION_MOVIES:
        raise ValueError(
            "validation set must be exactly the two frozen movies: "
            f"{sorted(DEFAULT_VALIDATION_MOVIES)}"
        )
    train_movies = set().union(*(_record_movies(record) for record in train_records)) if train_records else set()
    val_movies = set().union(*(_record_movies(record) for record in val_records)) if val_records else set()
    overlap = train_movies & val_movies
    heldout_in_train = train_movies & held_out
    unexpected_val = val_movies - held_out
    missing_val = held_out - val_movies
    if overlap:
        raise ValueError(f"movie leakage across train/val: {sorted(overlap)}")
    if heldout_in_train:
        raise ValueError(f"held-out movies appear in training: {sorted(heldout_in_train)}")
    if unexpected_val:
        raise ValueError(f"non-held-out real movies appear in validation: {sorted(unexpected_val)}")
    if require_all_validation_movies and missing_val:
        raise ValueError(f"held-out movies missing from validation: {sorted(missing_val)}")

    pajv_positive = []
    pajv_val = []
    train_background = []
    pajv_background = []
    for record in [*train_records, *val_records]:
        movies = _record_movies(record)
        if not any(_is_pajv101(movie) for movie in movies):
            continue
        if record in val_records:
            pajv_val.append(record["sample_id"])
        if not _is_background_record(record):
            pajv_positive.append(record["sample_id"])
    for record in train_records:
        if _is_background_record(record):
            train_background.append(record)
            if any(_is_pajv101(movie) for movie in _record_movies(record)):
                pajv_background.append(record)
    if pajv_positive:
        raise ValueError(f"pAJV101 may not provide positives: {pajv_positive[:5]}")
    if pajv_val:
        raise ValueError(f"pAJV101 may not appear in validation: {pajv_val[:5]}")
    fraction = len(pajv_background) / len(train_background) if train_background else 0.0
    if fraction > float(pajv101_max_background_fraction) + 1e-12:
        raise ValueError(
            f"pAJV101 is {fraction:.3%} of training backgrounds, above "
            f"{float(pajv101_max_background_fraction):.1%}"
        )

    source_counts = Counter(
        f"{record.get('source', 'unknown')}:{record.get('kind', 'unknown')}"
        for record in [*train_records, *val_records]
    )
    return {
        "status": "passed",
        "validation_movies_expected": sorted(held_out),
        "validation_movies_observed": sorted(val_movies),
        "training_movie_count": len(train_movies),
        "validation_movie_count": len(val_movies),
        "train_val_movie_overlap": sorted(overlap),
        "heldout_movies_in_training": sorted(heldout_in_train),
        "unexpected_validation_movies": sorted(unexpected_val),
        "missing_validation_movies": sorted(missing_val),
        "train_records": len(train_records),
        "validation_records": len(val_records),
        "source_kind_counts": dict(sorted(source_counts.items())),
        "pajv101_background_only": True,
        "pajv101_max_background_fraction": float(pajv101_max_background_fraction),
        "pajv101_training_background_records": len(pajv_background),
        "training_background_records": len(train_background),
        "pajv101_training_background_fraction": fraction,
    }


def _atomic_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest is missing: {path}")
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def emit_manifests(
    records: Iterable[dict[str, Any]],
    manifest_dir: str | Path,
    project_root: str | Path,
    *,
    validation_movies: Iterable[str] = DEFAULT_VALIDATION_MOVIES,
    pajv101_max_background_fraction: float = 0.15,
    require_all_validation_movies: bool = True,
) -> dict[str, Any]:
    """Validate records and emit deterministic train/validation JSONL files."""
    root = Path(project_root).expanduser().resolve()
    prepared = [_portable_paths(copy.deepcopy(record), root) for record in records]
    identifiers = [str(record["sample_id"]) for record in prepared]
    duplicates = [key for key, count in Counter(identifiers).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate manifest sample IDs: {duplicates[:5]}")
    for record in prepared:
        split = record.get("split")
        if split not in {"train", "val"}:
            raise ValueError(f"invalid split {split!r} for {record['sample_id']}")
        movies = _record_movies(record)
        real_expected = "val" if movies & set(validation_movies) else "train"
        if movies and split != real_expected:
            raise ValueError(
                f"split mismatch for {record['sample_id']}: {split}, expected {real_expected}"
            )
    train = sorted(
        (record for record in prepared if record["split"] == "train"),
        key=lambda item: item["sample_id"],
    )
    val = sorted(
        (record for record in prepared if record["split"] == "val"),
        key=lambda item: item["sample_id"],
    )
    audit = audit_leakage(
        train,
        val,
        validation_movies=validation_movies,
        pajv101_max_background_fraction=pajv101_max_background_fraction,
        require_all_validation_movies=require_all_validation_movies,
    )
    output = Path(manifest_dir).expanduser().resolve()
    _atomic_jsonl(output / "train.jsonl", train)
    _atomic_jsonl(output / "val.jsonl", val)
    _atomic_json(output / "leakage_audit.json", audit)
    return audit


def records_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build every presently configured source descriptor.

    Current records are always available.  Procedural and old-mask source
    modules are imported lazily so this script remains usable while individual
    data sources are being developed and tested independently.
    """
    paths = config["paths"]
    constraints = config["constraints"]
    records = load_current_records(
        paths["current_labels"],
        paths["queue_csv"],
        paths["project_root"],
        source_tile_size=int(config["input"]["source_tile_size"]),
        validation_movies=config["validation"]["held_out_movies"],
        base_seed=int(config["campaign"]["seed"]),
        background_blend=float(config["input"]["background_blend"]),
        pajv101_max_background_fraction=float(
            constraints["pajv101_max_background_fraction"]
        ),
    )

    from comet_sam3.data.synthetic import synthetic_records

    # Training scenes are generated directly from the deterministic
    # epoch/sample RNG and therefore do not need a finite manifest bank.  Only
    # validation recipes are frozen on disk.
    records.extend(
        synthetic_records(
            int(config["validation"]["fixed_synthetic_pairs"]),
            seed=int(config["validation"]["fixed_synthetic_seed"]),
            split="val",
        )
    )

    try:
        from comet_sam3.data.unet_masks import build_unet_manifest
    except ImportError:
        build_unet_manifest = None
    if build_unet_manifest is not None:
        old_records = build_unet_manifest(
            paths["project_root"],
            split="train",
            movie_splits=_movie_split_map(
                paths["project_root"], config["validation"]["held_out_movies"]
            ),
            seed=int(config["campaign"]["seed"]),
            tile_size=int(config["input"]["source_tile_size"]),
        )
        for record in old_records:
            record["kind"] = "pasted_axis_pair"
            record["preprocessing"] = {
                "causal_channels_t": [-2, -1, 0],
                "causal_channels_tp1": [-1, 0, 1],
                "joint_robust_percentiles": [1.0, 99.7],
                "temporal_median_background": True,
                "positive_residual_clip": True,
                "background_blend": float(config["input"]["background_blend"]),
            }
        records.extend(old_records)
    return records


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=SAM3_TRAINING_ROOT / "configs/campaign.yaml",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="re-audit existing JSONL manifests without rebuilding recipes",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.audit_only:
        manifest_dir = Path(config["paths"]["manifest_dir"])
        audit = audit_leakage(
            _read_jsonl(manifest_dir / "train.jsonl"),
            _read_jsonl(manifest_dir / "val.jsonl"),
            validation_movies=config["validation"]["held_out_movies"],
            pajv101_max_background_fraction=float(
                config["constraints"]["pajv101_max_background_fraction"]
            ),
        )
        print(json.dumps(audit, indent=2, sort_keys=True))
        return
    records = records_from_config(config)
    audit = emit_manifests(
        records,
        config["paths"]["manifest_dir"],
        config["paths"]["project_root"],
        validation_movies=config["validation"]["held_out_movies"],
        pajv101_max_background_fraction=float(
            config["constraints"]["pajv101_max_background_fraction"]
        ),
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
