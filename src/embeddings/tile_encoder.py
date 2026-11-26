"""Tile-level embeddings from a base model.

A tile is cut into windows (see ``windows.py``), each window is embedded, and the tile
embedding is the mean of its window embeddings. These are the "appearance vectors" the
selector uses for diversity and de-duplication
(``02_active_learning/rounds/<round_id>/embeddings.parquet``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..image_backend.tiles import TileRef, TileSource
from .base_model import SSLBaseModel
from .tables import VECTOR
from .windows import Window, WindowPlanner, crop_window

TILE_COLUMNS = {
    "tile_id": "string",
    "prep_id": "string",
    "slide_id": "string",
    "patient_id": "string",
    "base_model": "string",
    "pooling": "string",
    "window_mode": "string",
    "mpp_model": "float32",
    "n_windows": "int64",
    "embedding": VECTOR,
}
WINDOW_COLUMNS = {
    "window_id": "string",
    "tile_id": "string",
    "prep_id": "string",
    "base_model": "string",
    "pooling": "string",
    "embedding": VECTOR,
}


@dataclass
class TileEmbedding:
    """Embeddings of one tile: one row per window, and their mean."""

    tile: TileRef
    windows: list[Window]
    window_embeddings: np.ndarray  # [n_windows, D] float32
    embedding: np.ndarray  # [D] float32

    def tile_row(self, base_model: str, pooling: str, window_mode: str) -> dict:
        return {
            "tile_id": self.tile.tile_id,
            "prep_id": self.tile.prep_id,
            "slide_id": self.tile.slide_id,
            "patient_id": self.tile.patient_id,
            "base_model": base_model,
            "pooling": pooling,
            "window_mode": window_mode,
            "mpp_model": self.windows[0].mpp_model,
            "n_windows": len(self.windows),
            "embedding": self.embedding,
        }

    def window_rows(self, base_model: str, pooling: str) -> list[dict]:
        return [
            {
                "window_id": window.window_id,
                "tile_id": self.tile.tile_id,
                "prep_id": self.tile.prep_id,
                "base_model": base_model,
                "pooling": pooling,
                "embedding": vector,
            }
            for window, vector in zip(self.windows, self.window_embeddings)
        ]


class TileEncoder:
    """Embeds tiles with a base model, batching windows across tiles."""

    def __init__(
        self,
        model: SSLBaseModel,
        *,
        window_mode: str = "grid",
        mpp_model: float | None = None,
        overlap: float = 0.0,
        default_mpp: float | None = None,
        pooling: str | None = None,
        batch_size: int = 32,
        amp: bool = True,
        num_workers: int = 0,
    ):
        self.model = model
        self.pooling = pooling or model.spec.tile_pooling
        self.planner = WindowPlanner(
            mode=window_mode,
            patch_size=model.patch_size,
            native_size=model.fit_size(model.spec.native_size),
            mpp_model=mpp_model if mpp_model is not None else model.spec.native_mpp,
            overlap=overlap,
            default_mpp=default_mpp,
            role="embed",
        )
        self.batch_size = batch_size
        self.amp = amp
        self.num_workers = num_workers

    @torch.no_grad()
    def embed_windows(self, crops: torch.Tensor) -> np.ndarray:
        """``[B, H, W, 3]`` uint8 window crops -> ``[B, D]`` float32 embeddings."""
        device = self.model.device
        images = crops.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=self.amp and device.type == "cuda"):
            vectors = self.model.embed(images, self.pooling)
        return vectors.float().cpu().numpy()

    def encode_tile(self, image: np.ndarray, tile: TileRef) -> TileEmbedding:
        """Embed a single tile image."""
        windows = self.planner.plan(tile)
        crops = torch.from_numpy(np.stack([crop_window(image, tile, window) for window in windows]))
        vectors = np.concatenate(
            [self.embed_windows(crops[i : i + self.batch_size]) for i in range(0, len(crops), self.batch_size)]
        )
        return TileEmbedding(tile, windows, vectors, vectors.mean(axis=0))

    def encode(self, source: TileSource, tiles: Sequence[TileRef] | None = None) -> Iterator[TileEmbedding]:
        """Embed many tiles, in order. Reading and cropping can run in ``num_workers`` processes."""
        tiles = list(source.tiles if tiles is None else tiles)
        loader = DataLoader(
            _WindowDataset(source, tiles, self.planner),
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
        )
        pending: deque[dict] = deque()  # tiles whose windows are not all embedded yet, in order
        batch: list[torch.Tensor] = []
        owners: list[dict] = []

        def flush() -> None:
            if not batch:
                return
            for owner, vector in zip(owners, self.embed_windows(torch.stack(batch))):
                owner["vectors"].append(vector)
            batch.clear()
            owners.clear()

        def drain() -> Iterator[TileEmbedding]:
            while pending and len(pending[0]["vectors"]) == len(pending[0]["windows"]):
                entry = pending.popleft()
                vectors = np.stack(entry["vectors"])
                yield TileEmbedding(entry["tile"], entry["windows"], vectors, vectors.mean(axis=0))

        for index, crops in loader:
            entry = {"tile": tiles[index], "windows": self.planner.plan(tiles[index]), "vectors": []}
            pending.append(entry)
            for crop in crops:
                if batch and crop.shape != batch[0].shape:  # another tile size: cannot share a batch
                    flush()
                batch.append(crop)
                owners.append(entry)
                if len(batch) >= self.batch_size:
                    flush()
            yield from drain()
        flush()
        yield from drain()


class _WindowDataset(Dataset):
    """Item ``i`` is ``(i, crops)`` with all window crops of tile ``i`` as ``[n, H, W, 3]`` uint8."""

    def __init__(self, source: TileSource, tiles: Sequence[TileRef], planner: WindowPlanner):
        self.source = source
        self.tiles = tiles
        self.planner = planner

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        tile = self.tiles[index]
        image = self.source.read(tile)
        crops = np.stack([crop_window(image, tile, window) for window in self.planner.plan(tile)])
        return index, torch.from_numpy(crops)
