"""Small dependency-light helpers used by model adapters."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .base import ModelResult, TileRequest


def rgb_array(tile: Any) -> np.ndarray:
    """Convert a PIL image or array into contiguous uint8 RGB."""
    array = np.asarray(tile.convert("RGB") if isinstance(tile, Image.Image) else tile)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"Expected an RGB tile, got shape {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array[:, :, :3])


def checked_checkpoint(path: Path | None, model: str) -> Path:
    if path is None:
        raise FileNotFoundError(f"{model} requires an explicit checkpoint path; weights are never downloaded automatically")
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{model} checkpoint does not exist: {path}")
    return path.resolve()


def result(model: str, version: str, raw: Any, request: TileRequest,
           checkpoint: str, device: str, started: float, metadata: dict[str, Any]) -> ModelResult:
    """Normalize an integer instance map and center crop contextual predictions."""
    labels = np.asarray(raw)
    if labels.ndim == 3:
        # Common (instances, H, W) mask stacks: assign one id per plane.
        if labels.shape[0] <= 64 and labels.shape[-1] != 1:
            planes = labels > 0
            merged = np.zeros(planes.shape[1:], dtype=np.int32)
            for idx, plane in enumerate(planes, 1):
                merged[plane & (merged == 0)] = idx
            labels = merged
        elif labels.shape[-1] == 1:
            labels = labels[..., 0]
        else:
            raise ValueError(f"{model} returned unsupported mask shape {labels.shape}")
    if labels.ndim != 2:
        raise ValueError(f"{model} must return a 2D instance label map; got {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        # Binary backend output is not sufficient to claim instance segmentation.
        if np.unique(labels).size > 2:
            labels = labels.astype(np.int32)
        else:
            raise RuntimeError(f"{model} returned only a semantic/binary mask; cell-instance output is required")
    border = int(round(request.output_size * request.border_ratio))
    expected = request.output_size + 2 * border
    if labels.shape != (expected, expected):
        labels = np.asarray(Image.fromarray(labels.astype(np.int32), mode="I").resize((expected, expected), Image.Resampling.NEAREST))
    if border:
        labels = labels[border:border + request.output_size, border:border + request.output_size]
    labels = labels.astype(np.int32, copy=False)
    binary = (labels > 0).astype(np.uint8)
    return ModelResult(instance_mask=labels, binary_mask=binary, model_name=model,
        adapter_version=version, tile_id=request.tile_id, image_id=request.image_id,
        level=request.level, x=request.x, y=request.y, output_size=request.output_size,
        border_ratio=request.border_ratio, padding_policy=request.padding_policy,
        input_shape=(expected, expected, 3), output_shape=(request.output_size, request.output_size),
        device=device, checkpoint_id=checkpoint, elapsed_seconds=time.perf_counter() - started,
        metadata=metadata)
