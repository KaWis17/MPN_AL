"""Segmentation decoder that turns frozen ViT features into cell maps.

The design is modelled on CellViT (Hoerst et al.): a UNETR-style decoder with HoVer-Net
style outputs, a published way of getting instance segmentation out of pathology
foundation models:

* four intermediate ViT feature maps at stride 16 enter through skip paths, the raw image
  through a fifth, so cell borders can be placed at pixel precision;
* every output has its own upsampling branch:

  - ``np``  cell vs background (2 logits),
  - ``hv``  horizontal and vertical distance to the cell centre in [-1, 1] (separates touching cells),
  - ``nt``  cell type (``num_types + 1`` logits, 0 = background), only when ``num_types > 0``.

The decoder is the only trained part. It never contains base-model weights, so a head
checkpoint can be shared even when the encoder's licence forbids redistributing weights.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

DECODER_STRIDE = 16  # output pixels per input token, fixed by the four 2x upsampling stages
NUM_FEATURE_LEVELS = 4


def _norm(kind: str, channels: int) -> nn.Module:
    if kind == "group":
        return nn.GroupNorm(math.gcd(8, channels), channels)
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    raise ValueError(f"norm must be 'group' or 'batch', got {kind!r}")


def _conv(cin: int, cout: int, norm: str, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        _norm(norm, cout),
        nn.ReLU(inplace=True),
        nn.Dropout2d(dropout),
    )


def _deconv(cin: int, cout: int, norm: str, dropout: float) -> nn.Sequential:
    """Upsample 2x, then refine."""
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2), _conv(cout, cout, norm, dropout))


class _Branch(nn.Module):
    """One upsampling path from the deepest feature map to full resolution."""

    def __init__(self, embed_dim: int, width: int, out_channels: int, norm: str, dropout: float):
        super().__init__()
        w = width
        self.bottleneck = nn.ConvTranspose2d(embed_dim, 8 * w, kernel_size=2, stride=2)
        self.up3 = nn.Sequential(
            _conv(16 * w, 8 * w, norm, dropout),
            _conv(8 * w, 8 * w, norm, dropout),
            nn.ConvTranspose2d(8 * w, 4 * w, kernel_size=2, stride=2),
        )
        self.up2 = nn.Sequential(
            _conv(8 * w, 4 * w, norm, dropout),
            _conv(4 * w, 4 * w, norm, dropout),
            nn.ConvTranspose2d(4 * w, 2 * w, kernel_size=2, stride=2),
        )
        self.up1 = nn.Sequential(
            _conv(4 * w, 2 * w, norm, dropout),
            _conv(2 * w, 2 * w, norm, dropout),
            nn.ConvTranspose2d(2 * w, w, kernel_size=2, stride=2),
        )
        self.head = nn.Sequential(
            _conv(2 * w, w, norm, dropout),
            _conv(w, w, norm, dropout),
            nn.Conv2d(w, out_channels, kernel_size=1),
        )

    def forward(self, deepest: torch.Tensor, skips: Sequence[torch.Tensor]) -> torch.Tensor:
        skip0, skip1, skip2, skip3 = skips
        x = self.bottleneck(deepest)  # stride 8
        x = self.up3(torch.cat([skip3, x], dim=1))  # stride 4
        x = self.up2(torch.cat([skip2, x], dim=1))  # stride 2
        x = self.up1(torch.cat([skip1, x], dim=1))  # stride 1
        return self.head(torch.cat([skip0, x], dim=1))


class CellDecoder(nn.Module):
    """CellViT-style decoder; ``base_width`` scales every layer (64 = full size, 16 for tests)."""

    def __init__(
        self,
        embed_dim: int,
        num_types: int = 0,
        *,
        base_width: int = 64,
        norm: str = "group",
        dropout: float = 0.0,
    ):
        super().__init__()
        if base_width % 2:
            raise ValueError("base_width must be even")
        self.embed_dim = embed_dim
        self.config = {"num_types": num_types, "base_width": base_width, "norm": norm, "dropout": dropout}
        w = base_width
        # Skip paths are shared by all branches.
        self.skip0 = nn.Sequential(_conv(3, w // 2, norm, dropout), _conv(w // 2, w, norm, dropout))
        self.skip1 = nn.Sequential(
            _deconv(embed_dim, 8 * w, norm, dropout), _deconv(8 * w, 4 * w, norm, dropout), _deconv(4 * w, 2 * w, norm, dropout)
        )
        self.skip2 = nn.Sequential(_deconv(embed_dim, 8 * w, norm, dropout), _deconv(8 * w, 4 * w, norm, dropout))
        self.skip3 = _deconv(embed_dim, 8 * w, norm, dropout)

        outputs = {"np": 2, "hv": 2}
        if num_types > 0:
            outputs["nt"] = num_types + 1
        self.branches = nn.ModuleDict(
            {name: _Branch(embed_dim, w, channels, norm, dropout) for name, channels in outputs.items()}
        )

    def forward(self, image: torch.Tensor, features: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
        """``image`` ``[B, 3, H, W]`` (normalised) and four feature maps ``[B, D, H/16, W/16]``, shallow to deep."""
        if len(features) != NUM_FEATURE_LEVELS:
            raise ValueError(f"expected {NUM_FEATURE_LEVELS} feature maps, got {len(features)}")
        expected = (image.shape[-2] // DECODER_STRIDE, image.shape[-1] // DECODER_STRIDE)
        if tuple(features[-1].shape[-2:]) != expected:
            raise ValueError(
                f"feature maps are {tuple(features[-1].shape[-2:])} but a {tuple(image.shape[-2:])} image "
                f"needs {expected} (image side / {DECODER_STRIDE})"
            )
        z1, z2, z3, z4 = features
        skips = (self.skip0(image), self.skip1(z1), self.skip2(z2), self.skip3(z3))
        return {name: branch(z4, skips) for name, branch in self.branches.items()}
