from __future__ import annotations

import pytest
import torch
from torch import nn

from comet_sam3.model import (
    _install_autograd_vit_mlp_forwards,
    _without_optional_interactive_neck,
)


def _checkpoint_fixture():
    model_state = {
        "backbone.vision_backbone.convs.0.weight": torch.zeros(2, 3),
        "backbone.vision_backbone.convs.0.bias": torch.zeros(2),
        "core.weight": torch.zeros(1),
    }
    detector = {
        "backbone.vision_backbone.convs.0.weight": torch.ones(2, 3),
        "backbone.vision_backbone.convs.0.bias": torch.ones(2),
        "backbone.vision_backbone.sam2_convs.0.weight": torch.ones(2, 3),
        "backbone.vision_backbone.sam2_convs.0.bias": torch.ones(2),
        "core.weight": torch.ones(1),
    }
    return detector, model_state


def test_complete_optional_interactive_neck_is_excluded():
    detector, model_state = _checkpoint_fixture()

    filtered = _without_optional_interactive_neck(detector, model_state)

    assert "backbone.vision_backbone.sam2_convs.0.weight" not in filtered
    assert "backbone.vision_backbone.sam2_convs.0.bias" not in filtered
    assert "backbone.vision_backbone.convs.0.weight" in filtered
    assert "core.weight" in filtered


def test_incomplete_optional_interactive_neck_is_rejected():
    detector, model_state = _checkpoint_fixture()
    del detector["backbone.vision_backbone.sam2_convs.0.bias"]

    with pytest.raises(ValueError, match="optional interactive neck is malformed"):
        _without_optional_interactive_neck(detector, model_state)


def test_wrong_optional_interactive_neck_shape_is_rejected():
    detector, model_state = _checkpoint_fixture()
    detector["backbone.vision_backbone.sam2_convs.0.weight"] = torch.ones(3, 3)

    with pytest.raises(ValueError, match="wrong_shapes"):
        _without_optional_interactive_neck(detector, model_state)


def test_other_unexpected_keys_are_not_filtered():
    detector, model_state = _checkpoint_fixture()
    detector["unknown.branch.weight"] = torch.ones(1)

    filtered = _without_optional_interactive_neck(detector, model_state)

    assert "unknown.branch.weight" in filtered


def test_vit_mlp_replacement_preserves_autograd():
    class DummyMlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(3, 5)
            self.act = nn.GELU()
            self.drop1 = nn.Dropout(0.0)
            self.norm = nn.LayerNorm(5)
            self.fc2 = nn.Linear(5, 3)
            self.drop2 = nn.Dropout(0.0)

        def forward(self, _x):
            raise ValueError("inference-only fused forward was not replaced")

    block = nn.Module()
    block.mlp = DummyMlp()
    trunk = nn.Module()
    trunk.blocks = nn.ModuleList([block])
    vision = nn.Module()
    vision.trunk = trunk
    backbone = nn.Module()
    backbone.vision_backbone = vision
    model = nn.Module()
    model.backbone = backbone

    _install_autograd_vit_mlp_forwards(model)
    inputs = torch.randn(2, 3, requires_grad=True)
    model.backbone.vision_backbone.trunk.blocks[0].mlp(inputs).sum().backward()

    assert inputs.grad is not None
    assert block.mlp.fc1.weight.grad is not None
    assert block.mlp.fc2.weight.grad is not None
