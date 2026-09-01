from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

from comet_sam3.checkpointing import config_fingerprint, load_checkpoint, save_checkpoint
from comet_sam3.config import load_config
from comet_sam3.optim import build_optimizer, build_scheduler, optimizer_steps_per_campaign


class _Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(6)])
        self.ln_post = nn.LayerNorm(4)


class _Visual(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = _Trunk()
        self.convs = nn.ModuleList([nn.Conv2d(4, 4, 1)])


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = _Visual()


class _Segmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_predictor = nn.Linear(4, 4)


class _SAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _Backbone()
        self.transformer = nn.Linear(4, 4)
        self.segmentation_head = _Segmentation()


class _Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.sam3 = _SAM()
        self.class_embedding = nn.Parameter(torch.ones(1, 4))
        self.head_predictor = nn.Linear(4, 4)
        self.track_projector = nn.Linear(4, 4)
        self.link_scorer = nn.Linear(12, 1)
        self.base_checkpoint_sha256 = "a" * 64


def _late_model() -> _Wrapper:
    model = _Wrapper()
    visual = model.sam3.backbone.vision_backbone
    for parameter in visual.parameters():
        parameter.requires_grad_(False)
    for parameter in visual.convs.parameters():
        parameter.requires_grad_(True)
    for block in visual.trunk.blocks[-4:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    for parameter in visual.trunk.ln_post.parameters():
        parameter.requires_grad_(True)
    return model


def test_optimizer_includes_only_late_vision_and_has_no_decay_groups():
    config = load_config("SAM3Training/configs/campaign.yaml")
    model = _late_model()
    optimizer = build_optimizer(model, config)
    grouped = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    lower = {
        id(parameter)
        for block in model.sam3.backbone.vision_backbone.trunk.blocks[:2]
        for parameter in block.parameters()
    }
    upper = {
        id(parameter)
        for block in model.sam3.backbone.vision_backbone.trunk.blocks[-4:]
        for parameter in block.parameters()
    }
    assert not (grouped & lower)
    assert upper <= grouped
    assert any(group["weight_decay"] == 0.0 for group in optimizer.param_groups)
    assert any(group["weight_decay"] > 0.0 for group in optimizer.param_groups)


def test_checkpoint_roundtrip_validates_adapter_and_resume_state(tmp_path: Path):
    config = load_config("SAM3Training/configs/campaign.yaml")
    model = _late_model()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, optimizer_steps_per_campaign(config))
    path = tmp_path / "resume.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        epoch=6,
        global_step=37,
        config=config,
        next_epoch=6,
        next_pair_index=123,
    )
    restored = _late_model()
    restored_optimizer = build_optimizer(restored, config)
    restored_scheduler = build_scheduler(
        restored_optimizer, config, optimizer_steps_per_campaign(config)
    )
    payload = load_checkpoint(
        path,
        restored,
        restored_optimizer,
        restored_scheduler,
        config=config,
        restore_rng=False,
    )
    assert payload["next_epoch"] == 6
    assert payload["next_pair_index"] == 123
    assert payload["global_step"] == 37
    torch.testing.assert_close(restored.class_embedding, model.class_embedding)


def test_checkpoint_fingerprint_includes_procedural_campaign_settings():
    config = load_config("SAM3Training/configs/campaign.yaml")
    changed = copy.deepcopy(config)
    changed["procedural"]["decay_length_probabilities"] = [0.10, 0.65, 0.20, 0.05]
    assert config_fingerprint(config) != config_fingerprint(changed)


def test_training_job_requires_matching_two_regime_preflight():
    script = Path("SAM3Training/sbatch/train.sbatch").read_text()
    assert "missing GPU preflight report" in script
    assert 'regimes != {"frozen", "unfrozen"}' in script
    assert "config_fingerprint" in script
