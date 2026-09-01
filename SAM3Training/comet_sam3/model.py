"""Custom SAM3 query model for axis, head, presence, and pairwise identity.

The official SAM3 text encoder is intentionally bypassed.  A single learned
class token conditions every query, which is appropriate for this one-class
microscopy detector and avoids relying on the linguistic meaning of "comet".
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MethodType, SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


_IMAGE_NECK_PREFIX = "backbone.vision_backbone.convs."
_OPTIONAL_INTERACTIVE_NECK_PREFIX = "backbone.vision_backbone.sam2_convs."


def _without_optional_interactive_neck(
    detector: Mapping[str, torch.Tensor],
    model_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove only the checkpoint's unused, structurally verified SAM2 neck.

    Meta's official SAM3 checkpoint contains a second FPN neck for the optional
    interactive-image/tracker path even when ``build_sam3_image_model`` is
    called with ``enable_inst_interactivity=False``. The official loader
    ignores those unexpected keys. This campaign does not build or call that
    path, but otherwise requires an exact detector checkpoint load.

    If any optional-neck tensors are present, require the complete key set and
    the same shapes as the image neck before excluding them. This preserves
    strict validation instead of silently accepting arbitrary unexpected keys.
    """

    optional = {
        key: value
        for key, value in detector.items()
        if key.startswith(_OPTIONAL_INTERACTIVE_NECK_PREFIX)
    }
    if not optional:
        return dict(detector)

    image_neck = {
        key: value
        for key, value in model_state.items()
        if key.startswith(_IMAGE_NECK_PREFIX)
    }
    expected_optional = {
        _OPTIONAL_INTERACTIVE_NECK_PREFIX + key.removeprefix(_IMAGE_NECK_PREFIX): value
        for key, value in image_neck.items()
    }
    missing = sorted(set(expected_optional) - set(optional))
    extra = sorted(set(optional) - set(expected_optional))
    wrong_shapes = sorted(
        key
        for key in set(optional) & set(expected_optional)
        if tuple(optional[key].shape) != tuple(expected_optional[key].shape)
    )
    if missing or extra or wrong_shapes:
        raise ValueError(
            "official SAM3 optional interactive neck is malformed: "
            f"missing={missing[:20]}, extra={extra[:20]}, "
            f"wrong_shapes={wrong_shapes[:20]}"
        )

    return {
        key: value
        for key, value in detector.items()
        if not key.startswith(_OPTIONAL_INTERACTIVE_NECK_PREFIX)
    }


def _autograd_vit_mlp_forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run the pinned SAM3 ViT MLP with ordinary autograd operations."""

    x = module.fc1(x)
    x = module.act(x)
    x = module.drop1(x)
    x = module.norm(x)
    x = module.fc2(x)
    return module.drop2(x)


def _install_autograd_vit_mlp_forwards(sam3_image_model: nn.Module) -> None:
    """Replace SAM3's inference-only fused ViT MLP forward methods.

    The pinned OSS implementation always calls ``perflib.fused.addmm_act``;
    that kernel detaches its weights and explicitly rejects grad-enabled
    execution.  It is suitable for inference but cannot train the upper ViT
    blocks.  Rebinding only the MLP forward methods retains the exact modules,
    parameters, state-dict names, activation, normalization, and dropout while
    restoring standard PyTorch autograd.
    """

    blocks = sam3_image_model.backbone.vision_backbone.trunk.blocks
    if not blocks:
        raise ValueError("SAM3 vision trunk has no transformer blocks")
    required = ("fc1", "act", "drop1", "norm", "fc2", "drop2")
    for index, block in enumerate(blocks):
        mlp = block.mlp
        absent = [name for name in required if not hasattr(mlp, name)]
        if absent:
            raise ValueError(f"SAM3 ViT MLP block {index} is missing {absent}")
        mlp.forward = MethodType(_autograd_vit_mlp_forward, mlp)


@dataclass
class FramePredictions:
    presence_logits: torch.Tensor
    boxes_cxcywh: torch.Tensor
    axis_logits: torch.Tensor
    head_logits: torch.Tensor
    track_embeddings: torch.Tensor
    queries: torch.Tensor

    def probabilities(self, output_size: tuple[int, int] | None = None) -> dict[str, torch.Tensor]:
        axis = self.axis_logits
        head = self.head_logits
        if output_size is not None:
            axis = F.interpolate(axis, size=output_size, mode="bilinear", align_corners=False)
            head = F.interpolate(head, size=output_size, mode="bilinear", align_corners=False)
        return {
            "presence": self.presence_logits.sigmoid(),
            "axis": axis.sigmoid(),
            "head": head.sigmoid(),
        }


@dataclass
class PairPredictions:
    t: FramePredictions
    tp1: FramePredictions


class CometSAM3(nn.Module):
    """Use SAM3 object queries with a second mask predictor and link head."""

    def __init__(
        self,
        sam3_image_model: nn.Module,
        sam_input_size: int = 1008,
        track_embedding_dim: int = 128,
        disable_dac_query_duplication: bool = True,
    ) -> None:
        super().__init__()
        self.sam3 = sam3_image_model
        _install_autograd_vit_mlp_forwards(self.sam3)
        self.sam_input_size = int(sam_input_size)
        hidden_dim = int(self.sam3.hidden_dim)
        # The checkpoint has already been loaded.  No forward path below uses
        # language features, so release this large frozen module instead of
        # carrying it in GPU memory for a one-class detector.
        self.sam3.backbone.language_backbone = None
        self.class_embedding = nn.Parameter(torch.empty(1, hidden_dim))
        nn.init.normal_(self.class_embedding, std=0.02)

        segmentation_head = self.sam3.segmentation_head
        if segmentation_head is None or not hasattr(segmentation_head, "mask_predictor"):
            raise ValueError("SAM3 must be built with instance segmentation enabled")
        # Start from the pretrained mask mapping.  The original branch becomes
        # the uniformly widened axis predictor; this copy becomes the head map.
        self.head_predictor = copy.deepcopy(segmentation_head.mask_predictor)
        self.track_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, int(track_embedding_dim)),
        )
        self.link_scorer = nn.Sequential(
            nn.Linear(3 * int(track_embedding_dim), int(track_embedding_dim)),
            nn.GELU(),
            nn.Linear(int(track_embedding_dim), 1),
        )

        if disable_dac_query_duplication:
            self.sam3.dac = False
            self.sam3.transformer.decoder.dac = False
            self.sam3.transformer.decoder.num_o2m_queries = 0
        self._freeze_permanently_unused_branches()

    def _freeze_permanently_unused_branches(self) -> None:
        """Freeze checkpoint modules bypassed by the custom one-class path."""
        for parameter in self.sam3.geometry_encoder.parameters():
            parameter.requires_grad_(False)
        segmentation = self.sam3.segmentation_head
        if getattr(segmentation, "semantic_seg_head", None) is not None:
            for parameter in segmentation.semantic_seg_head.parameters():
                parameter.requires_grad_(False)
        if getattr(segmentation, "presence_head", None) is not None:
            for parameter in segmentation.presence_head.parameters():
                parameter.requires_grad_(False)
        # The decoder presence token itself participates in object-query self
        # attention and stays trainable.  Its unused auxiliary scoring head and
        # output norm do not affect our logits.
        decoder = self.sam3.transformer.decoder
        for attribute in ("presence_token_head", "presence_token_out_norm"):
            module = getattr(decoder, attribute, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        sam_input_size: int = 1008,
        track_embedding_dim: int = 128,
        disable_dac_query_duplication: bool = True,
        device: str = "cuda",
    ) -> "CometSAM3":
        """Build and strictly load the pinned official image checkpoint.

        This pinned SAM3 commit constructs one decoder coordinate cache on a
        CUDA device even when the module is initially built on CPU.  The final
        campaign is CUDA-only, so reject a misleading CPU request explicitly.
        """
        from sam3.model_builder import build_sam3_image_model

        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("the pinned SAM3 training wrapper requires CUDA")

        base = build_sam3_image_model(
            checkpoint_path=None,
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            eval_mode=False,
            # Load on CPU, remove the unused text tower, then move the smaller
            # wrapper.  This avoids a large transient GPU-memory spike.
            device="cpu",
            compile=False,
        )
        with open(checkpoint_path, "rb") as handle:
            digest = hashlib.sha256()
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        checkpoint_sha256 = digest.hexdigest()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model" in payload and isinstance(payload["model"], dict):
            payload = payload["model"]
        if not isinstance(payload, dict):
            raise ValueError("official SAM3 checkpoint must contain a state dictionary")
        detector = {
            key.replace("detector.", ""): value
            for key, value in payload.items()
            if "detector" in key
        }
        required_prefixes = (
            "backbone.vision_backbone.trunk.",
            "transformer.decoder.query_embed.",
            "dot_prod_scoring.",
            "segmentation_head.mask_predictor.",
        )
        absent = [prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in detector)]
        if absent:
            raise ValueError(f"checkpoint is not the expected SAM3 image checkpoint; missing {absent}")
        detector = _without_optional_interactive_neck(detector, base.state_dict())
        incompatibility = base.load_state_dict(detector, strict=False)
        if incompatibility.missing_keys or incompatibility.unexpected_keys:
            raise ValueError(
                "official SAM3 checkpoint did not load exactly: "
                f"missing={incompatibility.missing_keys[:20]}, "
                f"unexpected={incompatibility.unexpected_keys[:20]}"
            )
        del payload, detector
        model = cls(
            base,
            sam_input_size=sam_input_size,
            track_embedding_dim=track_embedding_dim,
            disable_dac_query_duplication=disable_dac_query_duplication,
        )
        model.base_checkpoint_sha256 = checkpoint_sha256
        return model.to(device)

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (N, 3, H, W)")
        images = F.interpolate(
            images.float(),
            size=(self.sam_input_size, self.sam_input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return (images - 0.5) / 0.5

    def _forward_images(self, images: torch.Tensor) -> FramePredictions:
        images = self._prepare_images(images)
        batch_size = images.shape[0]
        device = images.device

        backbone_out: dict[str, Any] = {"img_batch_all_stages": images}
        backbone_out.update(self.sam3.backbone.forward_image(images))
        image_ids = torch.arange(batch_size, device=device, dtype=torch.long)
        find_input = SimpleNamespace(img_ids=image_ids)
        prompt = self.class_embedding.view(1, 1, -1).expand(1, batch_size, -1)
        prompt_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)

        backbone_out, encoder_out, _ = self.sam3._run_encoder(
            backbone_out=backbone_out,
            find_input=find_input,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )
        out: dict[str, Any] = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
        }
        out, _ = self.sam3._run_decoder(
            memory=out["encoder_hidden_states"],
            pos_embed=encoder_out["pos_embed"],
            src_mask=encoder_out["padding_mask"],
            out=out,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_out=encoder_out,
        )

        seg = self.sam3.segmentation_head
        encoder_hidden = out["encoder_hidden_states"]
        if seg.cross_attend_prompt is not None:
            attended = seg.cross_attn_norm(encoder_hidden)
            attended = seg.cross_attend_prompt(
                query=attended,
                key=prompt,
                value=prompt,
                key_padding_mask=prompt_mask,
            )[0]
            encoder_hidden = encoder_hidden + attended

        pixel_embed = seg._embed_pixels(
            backbone_feats=backbone_out["backbone_fpn"],
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden,
        )
        instance_embed = seg.instance_seg_head(pixel_embed)
        queries = out["queries"]
        axis_logits = seg.mask_predictor(queries, instance_embed)
        head_logits = self.head_predictor(queries, instance_embed)
        tracks = F.normalize(self.track_projector(queries), dim=-1)

        return FramePredictions(
            presence_logits=out["pred_logits"].squeeze(-1),
            boxes_cxcywh=out["pred_boxes"],
            axis_logits=axis_logits,
            head_logits=head_logits,
            track_embeddings=tracks,
            queries=queries,
        )

    def forward(self, image_t: torch.Tensor, image_tp1: torch.Tensor) -> PairPredictions:
        if image_t.shape != image_tp1.shape:
            raise ValueError("paired image batches must have the same shape")
        batch_size = image_t.shape[0]
        combined = self._forward_images(torch.cat((image_t, image_tp1), dim=0))

        def split(frame: FramePredictions, part: slice) -> FramePredictions:
            return FramePredictions(**{
                name: getattr(frame, name)[part]
                for name in FramePredictions.__dataclass_fields__
            })

        return PairPredictions(
            t=split(combined, slice(0, batch_size)),
            tp1=split(combined, slice(batch_size, 2 * batch_size)),
        )

    def pairwise_link_logits(
        self,
        embeddings_t: torch.Tensor,
        embeddings_tp1: torch.Tensor,
    ) -> torch.Tensor:
        """Return all ``t`` by ``t+1`` same-comet logits."""
        left = embeddings_t.unsqueeze(-2).expand(*embeddings_t.shape[:-2], embeddings_t.shape[-2], embeddings_tp1.shape[-2], embeddings_t.shape[-1])
        right = embeddings_tp1.unsqueeze(-3).expand_as(left)
        features = torch.cat((left, right, (left - right).abs()), dim=-1)
        return self.link_scorer(features).squeeze(-1)

    def configure_trainable(
        self,
        epoch: int,
        unfreeze_upper_vision_blocks: int = 4,
        freeze_vision_through_epoch: int = 5,
    ) -> None:
        """Freeze text always; expose only the final ViT blocks after epoch five."""
        language = self.sam3.backbone.language_backbone
        if language is not None:
            for parameter in language.parameters():
                parameter.requires_grad_(False)

        visual = self.sam3.backbone.vision_backbone
        for parameter in visual.parameters():
            parameter.requires_grad_(False)

        # The FPN neck adapts sooner than the large ViT trunk.
        for parameter in visual.convs.parameters():
            parameter.requires_grad_(True)

        if int(epoch) > int(freeze_vision_through_epoch):
            blocks = visual.trunk.blocks
            count = min(int(unfreeze_upper_vision_blocks), len(blocks))
            if count > 0:
                for block in blocks[-count:]:
                    for parameter in block.parameters():
                        parameter.requires_grad_(True)
                for parameter in visual.trunk.ln_post.parameters():
                    parameter.requires_grad_(True)
        self._freeze_permanently_unused_branches()
