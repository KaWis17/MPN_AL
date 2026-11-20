"""The base-model segmenter: a frozen SSL encoder plus a trained :class:`CellDecoder`.

Geometry
--------
The decoder upsamples the token grid 16x. For an image of ``H x W`` pixels at the working
resolution (both multiples of 16) the token grid is therefore ``H/16 x W/16``, and the
encoder is fed the image resampled to ``H/16 * patch`` by ``W/16 * patch`` pixels: unchanged
for patch-16 encoders, 896 px for a 1024 px tile with patch 14. This keeps the predictions
aligned with the pixels whatever the patch size of the encoder.

``seg_mpp`` is the working resolution in um/px. ``None`` means "the tile's own resolution"
(about 0.25 um/px at 40x, which small cells need); 0.5 is four times cheaper and enough for
megakaryocytes.

Head checkpoints hold the decoder weights and everything needed to rebuild the model around
them, but never the encoder's weights.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..embeddings.base_model import SSLBaseModel, build_base_model, image_to_tensor
from ..embeddings.registry import BaseModelSpec
from ..embeddings.windows import _axis_offsets
from .decoder import DECODER_STRIDE, NUM_FEATURE_LEVELS, CellDecoder
from .labels import LabelSchema

logger = logging.getLogger(__name__)

HEAD_FORMAT = "mpn_al.segmentation_head/1"


class CellSegmenter(nn.Module):
    """Frozen base model + decoder. Call it on ``[B, 3, H, W]`` images in ``[0, 1]`` at working resolution."""

    def __init__(
        self,
        base: SSLBaseModel,
        decoder: CellDecoder,
        labels: LabelSchema,
        *,
        layers: Sequence[int] | None = None,
        seg_mpp: float | None = None,
    ):
        super().__init__()
        if decoder.embed_dim != base.embed_dim:
            raise ValueError(f"decoder expects {decoder.embed_dim}-d features, {base.name} produces {base.embed_dim}-d")
        layers = base.default_layers(NUM_FEATURE_LEVELS) if layers is None else layers
        self.layers = tuple(sorted(int(layer) % base.num_layers for layer in layers))
        if len(set(self.layers)) != NUM_FEATURE_LEVELS:
            raise ValueError(f"the decoder needs {NUM_FEATURE_LEVELS} distinct encoder layers, got {layers}")
        self.base = base
        self.decoder = decoder
        self.labels = labels
        self.seg_mpp = seg_mpp

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return logits ``np``/``hv`` (and ``nt``) at input resolution, plus ``tokens`` and ``cls`` of the encoder."""
        height, width = int(images.shape[-2]), int(images.shape[-1])
        if height % DECODER_STRIDE or width % DECODER_STRIDE:
            raise ValueError(f"image size {height}x{width} must be a multiple of {DECODER_STRIDE}")
        normalized = self.base.normalize(images)
        patch = self.base.patch_size
        encoder_input = normalized
        if patch != DECODER_STRIDE:
            size = (height // DECODER_STRIDE * patch, width // DECODER_STRIDE * patch)
            encoder_input = F.interpolate(normalized, size=size, mode="bilinear", antialias=True, align_corners=False)
        hidden = self.base(encoder_input, layers=self.layers, normalize=False)
        output = self.decoder(normalized, [hidden.hidden_states[layer] for layer in self.layers])
        output["tokens"] = hidden.patch_tokens
        if hidden.cls_token is not None:
            output["cls"] = hidden.cls_token
        return output

    def working_size(self, height: int, width: int, mpp: float | None) -> tuple[int, int]:
        """Size (multiples of 16) an image of ``height x width`` px at ``mpp`` is resampled to."""
        return working_size(height, width, mpp, self.seg_mpp)


def working_size(height: int, width: int, mpp: float | None, seg_mpp: float | None) -> tuple[int, int]:
    """Working-resolution size, as multiples of the decoder stride, of an image at ``mpp`` um/px."""
    if seg_mpp is None:
        scale = 1.0
    elif mpp is None:
        raise ValueError(f"the head works at {seg_mpp} um/px, so the tile resolution must be known (pass --mpp)")
    else:
        scale = mpp / seg_mpp
    return _to_stride(height * scale), _to_stride(width * scale)


@dataclass
class SegmentationMaps:
    """Dense predictions for one tile at working resolution."""

    prob: np.ndarray  # [H, W] float32, cell probability
    hv: np.ndarray  # [2, H, W] float32
    type_prob: np.ndarray | None  # [K + 1, H, W] float32
    tokens: np.ndarray  # [D, H/16, W/16] float32, encoder patch tokens
    scale: tuple[float, float]  # level-0 pixels per working pixel (x, y)


@torch.no_grad()
def predict_maps(
    segmenter: CellSegmenter,
    image: np.ndarray,
    *,
    mpp: float | None = None,
    amp: bool = True,
    max_size: int = 1024,
    overlap: int = 128,
) -> SegmentationMaps:
    """Run the segmenter on an RGB uint8 tile of any size.

    Images larger than ``max_size`` at working resolution are processed as overlapping crops
    whose logits are blended with centre-weighted windows, so memory stays bounded and seams
    do not show.
    """
    height, width = image.shape[:2]
    work_h, work_w = segmenter.working_size(height, width, mpp)
    maps = predict_working_maps(
        segmenter, resize_image(image, work_h, work_w), amp=amp, max_size=max_size, overlap=overlap
    )
    maps.scale = (width / work_w, height / work_h)
    return maps


def resize_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resample an RGB uint8 image, area-averaging when it shrinks."""
    if image.shape[:2] == (height, width):
        return image
    shrinking = height * width < image.shape[0] * image.shape[1]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)


@torch.no_grad()
def predict_working_maps(
    segmenter: CellSegmenter,
    image: np.ndarray,
    *,
    amp: bool = True,
    max_size: int = 1024,
    overlap: int = 128,
) -> SegmentationMaps:
    """:func:`predict_maps` for an image already at working resolution (sides multiples of 16)."""
    if max_size % DECODER_STRIDE or overlap % DECODER_STRIDE or overlap >= max_size:
        raise ValueError(f"max_size and overlap must be multiples of {DECODER_STRIDE}, with overlap < max_size")
    device = segmenter.base.device
    work_h, work_w = image.shape[:2]
    tensor = image_to_tensor(image, device)

    crop_h, crop_w = min(max_size, work_h), min(max_size, work_w)
    tops = _stride_offsets(work_h, crop_h, overlap)
    lefts = _stride_offsets(work_w, crop_w, overlap)
    sums: dict[str, torch.Tensor] = {}
    weight_sum = torch.zeros(1, 1, work_h, work_w, device=device)
    token_sum: torch.Tensor | None = None
    token_count = torch.zeros(1, 1, work_h // DECODER_STRIDE, work_w // DECODER_STRIDE, device=device)
    window = _blend_window(crop_h, crop_w, device)
    segmenter.eval()
    for top in tops:
        for left in lefts:
            crop = tensor[..., top : top + crop_h, left : left + crop_w]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                output = segmenter(crop)
            rows, cols = slice(top, top + crop_h), slice(left, left + crop_w)
            for name in ("np", "hv", "nt"):
                if name in output:
                    if name not in sums:
                        sums[name] = torch.zeros(1, output[name].shape[1], work_h, work_w, device=device)
                    sums[name][..., rows, cols] += output[name].float() * window
            weight_sum[..., rows, cols] += window
            tokens = output["tokens"].float()
            if token_sum is None:
                token_sum = torch.zeros(1, tokens.shape[1], *token_count.shape[-2:], device=device)
            token_rows = slice(top // DECODER_STRIDE, (top + crop_h) // DECODER_STRIDE)
            token_cols = slice(left // DECODER_STRIDE, (left + crop_w) // DECODER_STRIDE)
            token_sum[..., token_rows, token_cols] += tokens
            token_count[..., token_rows, token_cols] += 1

    logits = {name: total / weight_sum for name, total in sums.items()}
    return SegmentationMaps(
        prob=logits["np"].softmax(dim=1)[0, 1].cpu().numpy(),
        hv=logits["hv"][0].cpu().numpy(),
        type_prob=logits["nt"].softmax(dim=1)[0].cpu().numpy() if "nt" in logits else None,
        tokens=(token_sum / token_count)[0].cpu().numpy(),
        scale=(1.0, 1.0),
    )


# ---- checkpoints ----------------------------------------------------------------------------


def save_head(path: str | Path, segmenter: CellSegmenter, **extra: Any) -> Path:
    """Write the decoder and the recipe for the model around it. ``extra`` (metrics, training setup) is stored as given."""
    spec = dataclasses.asdict(segmenter.base.spec)
    spec["create_kwargs"] = dict(spec["create_kwargs"])
    for key in ("mean", "std"):
        spec[key] = list(spec[key]) if spec[key] is not None else None
    payload = {
        "format": HEAD_FORMAT,
        "base_model": spec,
        "encoder": {
            "embed_dim": segmenter.base.embed_dim,
            "patch_size": segmenter.base.patch_size,
            "num_layers": segmenter.base.num_layers,
        },
        "layers": list(segmenter.layers),
        "seg_mpp": segmenter.seg_mpp,
        "decoder": dict(segmenter.decoder.config),
        "labels": segmenter.labels.to_dict(),
        "state_dict": {key: value.detach().cpu() for key, value in segmenter.decoder.state_dict().items()},
        "extra": extra,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def read_head(path: str | Path) -> dict[str, Any]:
    """Load a head checkpoint as plain data (no code is executed) and check its format."""
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != HEAD_FORMAT:
        raise ValueError(f"{path} is not a segmentation head checkpoint ({HEAD_FORMAT})")
    return payload


def load_segmenter(
    path: str | Path,
    *,
    base: SSLBaseModel | None = None,
    encoder_checkpoint: str | Path | None = None,
    pretrained: bool = True,
    device: str | torch.device | None = None,
) -> CellSegmenter:
    """Rebuild the segmenter of a head checkpoint.

    The encoder is loaded from its usual source (hub or ``encoder_checkpoint``) unless ``base``
    is passed in; either way it must match the encoder the head was trained on.
    ``pretrained=False`` builds the encoder with random weights (tests only).
    """
    payload = read_head(path)
    if base is None:
        known = {f.name for f in dataclasses.fields(BaseModelSpec)}
        spec = {key: value for key, value in payload["base_model"].items() if key in known}
        for key in ("mean", "std"):
            spec[key] = tuple(spec[key]) if spec[key] is not None else None
        base = build_base_model(BaseModelSpec(**spec), pretrained=pretrained, checkpoint_path=encoder_checkpoint)
    expected = payload["encoder"]
    actual = {"embed_dim": base.embed_dim, "patch_size": base.patch_size, "num_layers": base.num_layers}
    if actual != expected:
        raise ValueError(
            f"{path} was trained on an encoder with {expected}, but {base.name} has {actual}; "
            "a head only works with the base model it was trained on"
        )
    decoder = CellDecoder(base.embed_dim, **payload["decoder"])
    decoder.load_state_dict(payload["state_dict"], strict=True)
    segmenter = CellSegmenter(
        base,
        decoder,
        LabelSchema.from_dict(payload["labels"]),
        layers=payload["layers"],
        seg_mpp=payload["seg_mpp"],
    )
    segmenter.eval()
    if device is not None:
        segmenter.to(device)
    return segmenter


def _to_stride(value: float) -> int:
    return max(DECODER_STRIDE, int(round(value / DECODER_STRIDE)) * DECODER_STRIDE)


def _stride_offsets(length: int, crop: int, overlap: int) -> list[int]:
    """Crop offsets covering ``length``, snapped to the token grid; the last crop ends at ``length``."""
    offsets = _axis_offsets(length, crop, overlap / crop)
    snapped = [offset // DECODER_STRIDE * DECODER_STRIDE for offset in offsets[:-1]] + [length - crop]
    return sorted(set(snapped))


def _blend_window(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Separable weights that fall off linearly towards the crop border (never reaching zero)."""

    def ramp(length: int) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32, device=device) + 0.5
        return torch.minimum(position, length - position) / (length / 2)

    return (ramp(height)[:, None] * ramp(width)[None, :])[None, None]
