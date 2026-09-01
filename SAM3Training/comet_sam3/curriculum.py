"""Exact, reproducible source mixtures for the finite campaign epochs."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .config import phase_for_epoch
from .schema import PairSample


def exact_source_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation so every epoch has the requested size."""
    raw = {name: float(value) * int(total) for name, value in fractions.items()}
    counts = {name: int(np.floor(value)) for name, value in raw.items()}
    remaining = int(total) - sum(counts.values())
    order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


class CurriculumSampler:
    def __init__(
        self,
        config: dict,
        sources: dict[str, Callable[[np.random.Generator], PairSample]],
    ) -> None:
        self.config = config
        self.sources = sources
        self.seed = int(config["campaign"]["seed"])

    def source_plan(self, epoch: int) -> list[str]:
        phase = phase_for_epoch(self.config, epoch)
        missing = set(phase["sources"]) - set(self.sources)
        if missing:
            raise KeyError(f"missing configured data sources: {sorted(missing)}")
        counts = exact_source_counts(phase["pairs_per_epoch"], phase["sources"])
        plan = [name for name, count in counts.items() for _ in range(count)]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(epoch)]))
        rng.shuffle(plan)
        return plan

    def iter_epoch(
        self,
        epoch: int,
        rank: int = 0,
        world_size: int = 1,
    ):
        plan = self.source_plan(epoch)
        for global_index in range(int(rank), len(plan), int(world_size)):
            seed = np.random.SeedSequence([self.seed, int(epoch), global_index])
            rng = np.random.default_rng(seed)
            yield self.sources[plan[global_index]](rng).validate()

