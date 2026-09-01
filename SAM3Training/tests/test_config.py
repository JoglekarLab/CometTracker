from pathlib import Path

from comet_sam3.config import load_config, phase_for_epoch


def test_campaign_schedule_is_complete_and_finishes_current_heavy():
    cfg = load_config(Path(__file__).parents[1] / "configs/campaign.yaml")
    assert phase_for_epoch(cfg, 1)["sources"]["procedural"] == 0.90
    assert phase_for_epoch(cfg, 5)["pairs_per_epoch"] == 6000
    assert phase_for_epoch(cfg, 25)["sources"]["current"] == 0.75
    assert cfg["targets"]["taper_axis"] is False

