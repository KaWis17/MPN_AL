"""Loss for the segmentation head (HoVer-Net / CellViT formulation, with an ignore mask).

Per branch:

* ``np``  cross-entropy + soft Dice on cell vs background;
* ``hv``  mean squared error, plus the squared error of the maps' spatial gradients inside
  cells (this is what sharpens the borders between touching cells);
* ``nt``  cross-entropy + soft Dice per class, when the head predicts types.

Pixels with ``valid == 0`` (ignore regions) contribute to none of them.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_WEIGHTS = {"np_ce": 1.0, "np_dice": 1.0, "hv_mse": 2.0, "hv_grad": 1.0, "nt_ce": 1.0, "nt_dice": 1.0}


class HeadLoss(nn.Module):
    def __init__(self, weights: Mapping[str, float] | None = None, gradient_kernel: int = 5):
        super().__init__()
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        offsets = torch.arange(-(gradient_kernel // 2), gradient_kernel // 2 + 1, dtype=torch.float32)
        x, y = torch.meshgrid(offsets, offsets, indexing="xy")
        radius = x * x + y * y + 1e-15
        # Sobel-like derivative kernels: d/dx for the horizontal map, d/dy for the vertical one.
        self.register_buffer("kernel", torch.stack([x / radius, y / radius])[:, None], persistent=False)

    def forward(
        self, outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """``targets``: ``np`` long ``[B,H,W]``, ``hv`` float ``[B,2,H,W]``, ``valid`` float ``[B,H,W]``, optional ``nt`` long."""
        valid = targets["valid"].float()
        parts = {
            "np_ce": _masked_ce(outputs["np"].float(), targets["np"], valid),
            "np_dice": _dice(outputs["np"].float().softmax(dim=1)[:, 1], targets["np"].float(), valid),
        }
        hv_pred, hv_true = outputs["hv"].float(), targets["hv"]
        parts["hv_mse"] = (((hv_pred - hv_true) ** 2).mean(dim=1) * valid).sum() / valid.sum().clamp_min(1.0)
        # The gradient at a pixel looks at its neighbours, so only pixels whose whole kernel
        # footprint is valid may count; otherwise ignored regions would leak into the loss.
        reach = self.kernel.shape[-1] // 2
        valid_footprint = -F.max_pool2d(-valid[:, None], 2 * reach + 1, stride=1, padding=reach)[:, 0]
        inside = valid_footprint * targets["np"].float()
        difference = self._gradient(hv_pred) - self._gradient(hv_true)
        parts["hv_grad"] = ((difference**2).sum(dim=1) * inside).sum() / inside.sum().clamp_min(1.0)

        if "nt" in outputs and "nt" in targets:
            logits = outputs["nt"].float()
            parts["nt_ce"] = _masked_ce(logits, targets["nt"], valid)
            probabilities = logits.softmax(dim=1)
            one_hot = F.one_hot(targets["nt"], logits.shape[1]).permute(0, 3, 1, 2).float()
            dice = [_dice(probabilities[:, c], one_hot[:, c], valid) for c in range(1, logits.shape[1])]
            parts["nt_dice"] = torch.stack(dice).mean()

        total = sum(self.weights[name] * value for name, value in parts.items())
        return total, {name: float(value.detach()) for name, value in parts.items()}

    def _gradient(self, hv: torch.Tensor) -> torch.Tensor:
        return F.conv2d(hv, self.kernel.to(hv.dtype), padding=self.kernel.shape[-1] // 2, groups=2)


def _masked_ce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(logits, target, reduction="none")
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def _dice(probability: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probability, target = probability * valid, target * valid
    intersection = (probability * target).sum()
    return 1.0 - (2.0 * intersection + smooth) / (probability.sum() + target.sum() + smooth)
