from pathlib import Path

from comet_sam3.config import load_config
from comet_sam3.curriculum import exact_source_counts


def test_first_epoch_has_exactly_ninety_percent_procedural():
    cfg = load_config(Path(__file__).parents[1] / "configs/campaign.yaml")
    phase = cfg["training"]["phases"][0]
    counts = exact_source_counts(phase["pairs_per_epoch"], phase["sources"])
    assert counts == {"procedural": 5400, "unet_paste": 480, "current": 120}
    assert sum(counts.values()) == 6000

