"""Manifest-backed real sources and deterministic on-the-fly procedural data."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..curriculum import CurriculumSampler
from ..schema import PairSample
from .current_annotations import augment_current_record, build_current_pair_sample
from .synthetic import (
    SyntheticConfig,
    build_synthetic_pair_sample,
    generate_synthetic_pair,
)
from .unet_masks import augment_unet_record, build_unet_pair_sample


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"training manifest is missing: {path}; run scripts/build_manifest.py"
        )
    records: list[dict] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return records


class TrainingSources:
    """Resolve one curriculum source name to one validated ``PairSample``.

    The source callable receives the exact per-epoch/per-index RNG constructed
    by :class:`EpochPairDataset`.  This makes random D8 transforms, translated
    crops, old-mask paste choices, and procedural scenes reproducible without
    saving rendered arrays.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.project_root = Path(config["paths"]["project_root"])
        manifest_dir = Path(config["paths"]["manifest_dir"])
        self.train_records = read_jsonl(manifest_dir / "train.jsonl")
        self.val_records = read_jsonl(manifest_dir / "val.jsonl")

        self.current_train = [
            record for record in self.train_records if record.get("source") == "current"
        ]
        self.current_positive = [
            record for record in self.current_train if record.get("kind") == "accepted_pair"
        ]
        self.current_background = [
            record
            for record in self.current_train
            if record.get("kind") == "background_negative"
        ]
        self.unet_train = [
            record
            for record in self.train_records
            if record.get("source") in {"unet_masks", "unet_paste"}
        ]
        self.current_val = [
            record for record in self.val_records if record.get("source") == "current"
        ]
        self.synthetic_val = [
            record
            for record in self.val_records
            if record.get("source") == "procedural"
        ]
        if not self.current_positive:
            raise ValueError("manifest contains no current positive annotations")
        if not self.current_background:
            raise ValueError("manifest contains no certified current backgrounds")
        if not self.unet_train:
            raise ValueError("manifest contains no old U-Net mask tracks")
        if not self.current_val or not self.synthetic_val:
            raise ValueError("validation manifest needs current and fixed procedural records")

        sampling = config["training"].get("source_sampling", {})
        self.current_positive_fraction = float(
            sampling.get("current_positive_fraction", 0.75)
        )
        self.procedural_empty_fraction = float(
            sampling.get("procedural_empty_fraction", 0.10)
        )
        self.procedural_frozen_fraction = float(
            sampling.get("procedural_frozen_fraction", 0.10)
        )
        if not 0.0 < self.current_positive_fraction < 1.0:
            raise ValueError("current_positive_fraction must be in (0, 1)")
        if (
            self.procedural_empty_fraction < 0
            or self.procedural_frozen_fraction < 0
            or self.procedural_empty_fraction + self.procedural_frozen_fraction >= 1
        ):
            raise ValueError("invalid procedural negative fractions")
        self.max_retries = int(sampling.get("materialization_retries", 12))
        procedural = dict(config.get("procedural", {}))
        self.synthetic_config = SyntheticConfig(
            tile_size=int(config["input"]["source_tile_size"]),
            background_blend=float(config["input"]["background_blend"]),
            **procedural,
        ).validate()

        self.callables: dict[str, Callable[[np.random.Generator], PairSample]] = {
            "procedural": self.procedural,
            "unet_paste": self.unet_paste,
            "current": self.current,
        }

    @staticmethod
    def _pick(records: Sequence[dict], rng: np.random.Generator) -> dict:
        if not records:
            raise ValueError("cannot sample an empty record collection")
        return records[int(rng.integers(0, len(records)))]

    def procedural(self, rng: np.random.Generator) -> PairSample:
        draw = float(rng.random())
        if draw < self.procedural_empty_fraction:
            scene_kind = "empty"
        elif draw < self.procedural_empty_fraction + self.procedural_frozen_fraction:
            scene_kind = "frozen"
        else:
            scene_kind = "positive"
        return generate_synthetic_pair(
            rng=rng,
            scene_kind=scene_kind,
            config=self.synthetic_config,
        )

    def current(self, rng: np.random.Generator) -> PairSample:
        records = (
            self.current_positive
            if float(rng.random()) < self.current_positive_fraction
            else self.current_background
        )
        last_error: Exception | None = None
        for _ in range(self.max_retries):
            record = self._pick(records, rng)
            augmented = augment_current_record(
                record,
                seed=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
            try:
                return build_current_pair_sample(augmented, self.project_root)
            except (ValueError, IndexError) as error:
                # A large translated crop can occasionally clip a long axis at
                # a movie edge.  Resampling is safe; silently dropping target
                # pixels is not.
                last_error = error
        raise RuntimeError("could not materialize a complete current target") from last_error

    def unet_paste(self, rng: np.random.Generator) -> PairSample:
        last_error: Exception | None = None
        for _ in range(self.max_retries):
            record = self._pick(self.unet_train, rng)
            if not bool(record.get("augmented", False)):
                try:
                    record = augment_unet_record(
                        record,
                        self.project_root,
                        rotation=int(rng.integers(0, 4)),
                        reflect=bool(rng.integers(0, 2)),
                        seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                    )
                except ValueError as error:
                    last_error = error
                    continue
            try:
                return build_unet_pair_sample(
                    record,
                    self.project_root,
                    background_blend=float(self.config["input"]["background_blend"]),
                )
            except (ValueError, IndexError) as error:
                last_error = error
        raise RuntimeError("could not materialize an old-mask paste") from last_error

    def inventory(self) -> dict[str, int | float]:
        counts = Counter(
            f"{record.get('source')}:{record.get('kind', 'recipe')}"
            for record in [*self.train_records, *self.val_records]
        )
        return {
            **dict(sorted(counts.items())),
            "current_positive_fraction_when_sampled": self.current_positive_fraction,
            "procedural_empty_fraction": self.procedural_empty_fraction,
            "procedural_frozen_fraction": self.procedural_frozen_fraction,
        }

    def validation_samples(self, full: bool) -> Iterator[PairSample]:
        current = self.current_val
        synthetic = self.synthetic_val
        if not full:
            positive = [record for record in current if record.get("kind") == "accepted_pair"][:8]
            background = [
                record for record in current if record.get("kind") == "background_negative"
            ][:8]
            current = [*positive, *background]
            synthetic = synthetic[:16]
        for record in current:
            # Validation geometry is frozen: no random translation or D8.
            yield build_current_pair_sample(record, self.project_root)
        for record in synthetic:
            yield build_synthetic_pair_sample(record, config=self.synthetic_config)


class EpochPairDataset(Dataset):
    """Map-style deterministic epoch plan, compatible with DataLoader workers."""

    def __init__(self, config: dict, sources: TrainingSources, epoch: int) -> None:
        self.config = config
        self.sources = sources
        self.epoch = int(epoch)
        self.seed = int(config["campaign"]["seed"])
        self.plan = CurriculumSampler(config, sources.callables).source_plan(self.epoch)

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, index: int) -> PairSample:
        source_name = self.plan[int(index)]
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, int(index)])
        )
        return self.sources.callables[source_name](rng).validate()


def collate_pair_samples(samples: list[PairSample]):
    """Stack only images; retain rich partial-label objects for the criterion."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    image_t = torch.from_numpy(np.stack([sample.image_t for sample in samples])).permute(
        0, 3, 1, 2
    )
    image_tp1 = torch.from_numpy(
        np.stack([sample.image_tp1 for sample in samples])
    ).permute(0, 3, 1, 2)
    return image_t.contiguous(), image_tp1.contiguous(), samples


__all__ = [
    "EpochPairDataset",
    "TrainingSources",
    "collate_pair_samples",
    "read_jsonl",
]
