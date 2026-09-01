import numpy as np
import torch
from torch import nn

from comet_sam3.config import load_config
from comet_sam3.losses import CometMultitaskLoss
from comet_sam3.model import FramePredictions, PairPredictions
from comet_sam3.schema import CometInstance, PairSample


def _frame(seed: int) -> FramePredictions:
    generator = torch.Generator().manual_seed(seed)
    return FramePredictions(
        presence_logits=torch.randn(1, 3, generator=generator, requires_grad=True),
        boxes_cxcywh=torch.rand(1, 3, 4, generator=generator, requires_grad=True),
        axis_logits=torch.randn(1, 3, 16, 16, generator=generator, requires_grad=True),
        head_logits=torch.randn(1, 3, 16, 16, generator=generator, requires_grad=True),
        track_embeddings=torch.nn.functional.normalize(
            torch.randn(1, 3, 8, generator=generator, requires_grad=True), dim=-1
        ),
        queries=torch.randn(1, 3, 16, generator=generator, requires_grad=True),
    )


def test_partial_nonexhaustive_pair_has_finite_loss():
    config = load_config("SAM3Training/configs/campaign.yaml")
    criterion = CometMultitaskLoss(config)
    axis = np.stack((np.arange(8, 20), np.arange(10, 22)), axis=-1).astype(np.float32)
    left = CometInstance("left", "track", (19.0, 21.0), axis, True, True)
    right = CometInstance("right", "track", (18.0, 23.0), axis + (np.array([-1, 2])), True, True)
    image = np.zeros((32, 32, 3), np.float32)
    sample = PairSample(
        "sample", "current", image, image.copy(), [left], [right], [("left", "right")]
    ).validate()
    model = nn.Module()
    model.link_scorer = nn.Sequential(nn.Linear(24, 1))
    total, pieces, _ = criterion(model, PairPredictions(_frame(1), _frame(2)), [sample])
    assert torch.isfinite(total)
    assert set(pieces) == {
        "presence", "box", "axis_focal", "axis_dice", "head_focal", "head_coordinate", "link"
    }
    total.backward()


def test_track_id_links_are_positive_and_regional_background_is_not_zero():
    config = load_config("SAM3Training/configs/campaign.yaml")
    criterion = CometMultitaskLoss(config)
    image = np.zeros((32, 32, 3), np.float32)
    axis = np.stack((np.arange(8, 16), np.arange(10, 18)), axis=-1).astype(np.float32)
    left = CometInstance("instance-left", "persistent-track", (15.0, 17.0), axis, True, True)
    right = CometInstance("instance-right", "persistent-track", (14.0, 19.0), axis + [-1, 2], True, True)
    linked = PairSample(
        "synthetic-link",
        "procedural",
        image,
        image.copy(),
        [left],
        [right],
        [("persistent-track", "persistent-track")],
        True,
        True,
        {"link_exhaustive": True},
    ).validate()
    model = nn.Module()
    model.link_scorer = nn.Sequential(nn.Linear(24, 1))
    total, pieces, _ = criterion(model, PairPredictions(_frame(3), _frame(4)), [linked])
    assert torch.isfinite(total)
    assert float(pieces["link"].detach()) > 0.0

    background = PairSample(
        "partial-background",
        "current",
        image,
        image.copy(),
        metadata={"certified_background_regions_t": [[0.0, 32.0, 0.0, 32.0]]},
    ).validate()
    total, pieces, _ = criterion(model, PairPredictions(_frame(5), _frame(6)), [background])
    assert torch.isfinite(total)
    assert float(pieces["presence"].detach()) > 0.0
    assert float(pieces["head_focal"].detach()) == 0.0
    assert float(pieces["axis_focal"].detach()) == 0.0
