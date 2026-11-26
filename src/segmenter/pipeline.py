"""Tile in, cells out: the end-to-end segmentation pipeline and its on-disk outputs.

Two ways of finding cells share everything downstream:

``head``            the base-model segmenter (frozen SSL encoder + trained decoder);
``stain-baseline``  the label-free nucleus fallback, with the base model used only to embed
                    the cells it finds.

Outputs per run directory (``--out``, e.g. ``data/03_predictions/<round_id>``)::

    by_tile/<tile_id>.geojson    cell polygons in level-0 slide coordinates, for QuPath
    cells/<tile_id>.parquet      one row per cell: geometry, class, confidence, embedding
    tile_stats.jsonl             one line per tile: counts and uncertainty, for the selector
    run.json                     what produced all of the above

``pd.read_parquet("<out>/cells")`` loads the cells of every tile as one table.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ..embeddings.base_model import SSLBaseModel, image_to_tensor
from ..embeddings.tables import VECTOR, ParquetStreamWriter
from ..image_backend.tiles import TileRef
from .cell_features import pool_cell_embeddings
from .geojson import cells_to_feature_collection, write_geojson
from .labels import LabelSchema
from .model import CellSegmenter, predict_maps
from .postprocess import (
    Cell,
    PostprocessConfig,
    drop_instances_without_cells,
    hv_to_instances,
    instances_to_cells,
    tile_statistics,
)
from .stain_baseline import StainBaselineConfig, segment_nuclei_stain

logger = logging.getLogger(__name__)

CELL_COLUMNS = {
    "cell_id": "string",
    "tile_id": "string",
    "prep_id": "string",
    "slide_id": "string",
    "x": "float32",
    "y": "float32",
    "area_px": "float32",
    "area_um2": "float32",
    "class_id": "int64",
    "class_name": "string",
    "prob": "float32",
    "class_prob": "float32",
    "touches_border": "bool",
    "method": "string",
    "base_model": "string",
}


@dataclass(frozen=True)
class PipelineConfig:
    default_mpp: float | None = None  # slide resolution for tiles that do not carry one
    amp: bool = True
    max_size: int = 1024  # largest side (working px) the segmenter sees in one pass
    overlap: int = 128
    embed_mpp: float = 0.5  # stain baseline: resolution of the base-model pass that embeds the cells
    cell_embeddings: bool = True
    context_margin: int = 0  # level-0 px of surroundings read around each tile; cells are kept by their centroid
    simplify: float = 0.75  # polygon tolerance in working pixels
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    baseline: StainBaselineConfig = field(default_factory=StainBaselineConfig)


@dataclass
class TileResult:
    tile: TileRef
    cells: list[Cell]
    cell_embeddings: np.ndarray | None  # [n_cells, D] float32
    stats: dict[str, float]
    instances: np.ndarray  # instance map at working resolution
    scale: tuple[float, float]  # level-0 pixels per working pixel (x, y)
    mpp: float | None
    method: str
    prob: np.ndarray | None = None  # cell probability at working resolution (head only)


class SegmentationPipeline:
    """Segments tiles with a head checkpoint's segmenter, or with the stain baseline when there is none."""

    def __init__(
        self,
        *,
        segmenter: CellSegmenter | None = None,
        base_model: SSLBaseModel | None = None,
        labels: LabelSchema | None = None,
        config: PipelineConfig = PipelineConfig(),
    ):
        self.segmenter = segmenter
        self.base_model = segmenter.base if segmenter is not None else base_model
        self.method = "head" if segmenter is not None else "stain-baseline"
        self.labels = labels or (segmenter.labels if segmenter is not None else LabelSchema.single_class("nucleus"))
        self.config = config
        self._warned_mpp = False

    def run_tile(self, image: np.ndarray, tile: TileRef) -> TileResult:
        """Segment one RGB uint8 tile image.

        With ``config.context_margin`` = M the image is the tile plus M px on every side
        (``TileSource.read_with_margin``). The whole padded image is segmented, then only the cells
        whose centroid lies inside the tile are kept, so tiles that abut never report a cell twice
        and cells crossing the tile edge are whole. Their polygons are not clipped to the tile, and
        ``touches_border`` then means "crosses the tile edge".
        """
        margin = self.config.context_margin
        if margin <= 0:
            return self._run(image, tile)
        expected = (tile.height + 2 * margin, tile.width + 2 * margin)
        if image.shape[:2] != expected:
            raise ValueError(
                f"tile {tile.tile_id}: expected the tile plus a {margin} px margin ({expected[1]}x{expected[0]} px), "
                f"got {image.shape[1]}x{image.shape[0]} px"
            )
        padded = dataclasses.replace(tile, x=tile.x - margin, y=tile.y - margin, width=expected[1], height=expected[0])
        return self._crop_to_tile(self._run(image, padded), tile, margin)

    def _crop_to_tile(self, result: TileResult, tile: TileRef, margin: int) -> TileResult:
        x0, y0, x1, y1 = tile.x, tile.y, tile.x + tile.width, tile.y + tile.height
        keep = [
            i for i, cell in enumerate(result.cells)
            if x0 <= cell.centroid[0] < x1 and y0 <= cell.centroid[1] < y1
        ]
        cells = [result.cells[i] for i in keep]
        for cell in cells:
            lo, hi = cell.polygon.min(axis=0), cell.polygon.max(axis=0)
            cell.touches_border = bool(lo[0] < x0 or lo[1] < y0 or hi[0] > x1 or hi[1] > y1)
        embeddings = result.cell_embeddings[keep] if result.cell_embeddings is not None else None

        sx, sy = result.scale
        top, left = int(round(margin / sy)), int(round(margin / sx))
        rows = slice(top, top + int(round(tile.height / sy)))
        cols = slice(left, left + int(round(tile.width / sx)))
        kept_labels = np.zeros(int(result.instances.max()) + 1, dtype=bool)
        kept_labels[[cell.label for cell in cells]] = True
        instances = np.where(kept_labels[result.instances], result.instances, 0)[rows, cols]
        prob = result.prob[rows, cols] if result.prob is not None else None
        return dataclasses.replace(
            result,
            tile=tile,
            cells=cells,
            cell_embeddings=embeddings,
            instances=instances,
            prob=prob,
            stats=tile_statistics(cells, instances, prob),
        )

    def _run(self, image: np.ndarray, tile: TileRef) -> TileResult:
        config = self.config
        mpp = tile.mpp if tile.mpp is not None else config.default_mpp
        embeddings = None
        if self.segmenter is not None:
            maps = predict_maps(
                self.segmenter, image, mpp=mpp, amp=config.amp, max_size=config.max_size, overlap=config.overlap
            )
            scale, prob = maps.scale, maps.prob
            work_mpp = mpp * scale[0] if mpp is not None else None
            instances = hv_to_instances(prob, maps.hv, mpp=work_mpp, config=config.postprocess)
            cells = self._cells(instances, scale, tile, prob, maps.type_prob)
            instances = drop_instances_without_cells(instances, cells)
            if config.cell_embeddings:
                embeddings = pool_cell_embeddings(instances, maps.tokens, [cell.label for cell in cells])
        else:
            if mpp is None and not self._warned_mpp:
                logger.warning(
                    "no slide resolution given; the stain baseline assumes %.2f um/px (pass --mpp)",
                    config.baseline.reference_mpp,
                )
                self._warned_mpp = True
            scale, prob = (1.0, 1.0), None
            instances = segment_nuclei_stain(image, mpp=mpp, config=config.baseline)
            cells = self._cells(instances, scale, tile, None, None)
            instances = drop_instances_without_cells(instances, cells)
            if config.cell_embeddings and self.base_model is not None:
                tokens = dense_tokens(self.base_model, image, mpp=mpp, target_mpp=config.embed_mpp, amp=config.amp)
                embeddings = pool_cell_embeddings(instances, tokens, [cell.label for cell in cells])
        return TileResult(
            tile=tile,
            cells=cells,
            cell_embeddings=embeddings,
            stats=tile_statistics(cells, instances, prob),
            instances=instances,
            scale=scale,
            mpp=mpp,
            method=self.method,
            prob=prob,
        )

    def _cells(self, instances, scale, tile, prob, type_prob) -> list[Cell]:
        return instances_to_cells(
            instances,
            scale=scale,
            origin=(float(tile.x), float(tile.y)),
            prob=prob,
            type_prob=type_prob,
            labels=self.labels,
            simplify=self.config.simplify,
        )


@torch.no_grad()
def dense_tokens(
    model: SSLBaseModel,
    image: np.ndarray,
    *,
    mpp: float | None,
    target_mpp: float = 0.5,
    amp: bool = True,
    max_side: int = 1568,
) -> np.ndarray:
    """Patch tokens ``[D, gh, gw]`` of a whole tile, read in one pass at about ``target_mpp``.

    Without a known resolution the image is used at its own size. ``max_side`` caps the
    encoder input so that a large image cannot exhaust memory.
    """
    height, width = image.shape[:2]
    scale = 1.0 if mpp is None else mpp / target_mpp
    scale = min(scale, max_side / max(height, width))
    size = (model.fit_size(width * scale), model.fit_size(height * scale))
    if size != (width, height):
        shrinking = size[0] * size[1] < width * height
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC)
    device = model.device
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
        tokens = model(image_to_tensor(image, device)).patch_tokens
    return tokens[0].float().cpu().numpy()


# ---- outputs --------------------------------------------------------------------------------


class RunWriter:
    """Writes the per-tile files of a run and appends to its ``tile_stats.jsonl``."""

    def __init__(
        self,
        out_dir: str | Path,
        *,
        labels: LabelSchema | None,
        base_model: str | None,
        object_type: str = "annotation",
        save_masks: bool = False,
        stats_name: str = "tile_stats.jsonl",
    ):
        self.out_dir = Path(out_dir)
        self.labels = labels
        self.base_model = base_model
        self.object_type = object_type
        self.save_masks = save_masks
        self.stats_path = self.out_dir / stats_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def geojson_path(self, tile_id: str) -> Path:
        return self.out_dir / "by_tile" / f"{tile_id}.geojson"

    def cells_path(self, tile_id: str) -> Path:
        return self.out_dir / "cells" / f"{tile_id}.parquet"

    def is_done(self, tile_id: str) -> bool:
        """True once both per-tile files exist (each is moved into place only when complete)."""
        return self.geojson_path(tile_id).exists() and self.cells_path(tile_id).exists()

    def write(self, result: TileResult, seconds: float | None = None) -> None:
        tile = result.tile
        collection = cells_to_feature_collection(
            result.cells, self.labels, tile_id=tile.tile_id, object_type=self.object_type, mpp=result.mpp
        )
        columns = dict(CELL_COLUMNS)
        vector_sizes = {}
        if result.cell_embeddings is not None:
            columns["embedding"] = VECTOR
            vector_sizes["embedding"] = int(result.cell_embeddings.shape[1])
        with ParquetStreamWriter(self.cells_path(tile.tile_id), columns, vector_sizes=vector_sizes) as writer:
            for index, cell in enumerate(result.cells):
                row: dict[str, Any] = {
                    "cell_id": f"{tile.tile_id}_c{index:05d}",
                    "tile_id": tile.tile_id,
                    "prep_id": tile.prep_id,
                    "slide_id": tile.slide_id,
                    "x": cell.centroid[0],
                    "y": cell.centroid[1],
                    "area_px": cell.area,
                    "area_um2": cell.area * result.mpp**2 if result.mpp is not None else None,
                    "class_id": cell.class_id,
                    "class_name": cell.class_name,
                    "prob": cell.prob,
                    "class_prob": cell.class_prob,
                    "touches_border": cell.touches_border,
                    "method": result.method,
                    "base_model": self.base_model,
                }
                if result.cell_embeddings is not None:
                    row["embedding"] = result.cell_embeddings[index]
                writer.add(row)
        if self.save_masks:
            mask_path = self.out_dir / "masks" / f"{tile.tile_id}.npz"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mask_path, instances=result.instances, scale=np.asarray(result.scale))
        # The GeoJSON goes last: its presence marks the tile as finished.
        write_geojson(self.geojson_path(tile.tile_id), collection)

        record = {"tile_id": tile.tile_id, "prep_id": tile.prep_id, "method": result.method, **result.stats}
        if seconds is not None:
            record["seconds"] = round(seconds, 3)
        with open(self.stats_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def read_tile_stats(path: str | Path):
    """Tile statistics as a DataFrame with one row per tile (the latest, if a tile was redone).

    ``path`` is a ``tile_stats.jsonl`` file, or the run directory: then the file and all the
    ``tile_stats.shard-*.jsonl`` files of a job array are read together.
    """
    import pandas as pd

    path = Path(path)
    files = sorted(path.glob("tile_stats*.jsonl")) if path.is_dir() else [path]
    if not path.is_dir() and not path.exists():  # a merged file that was never written: use the shards beside it
        files = sorted(path.parent.glob(f"{path.stem}.shard-*{path.suffix}"))
    tables = [pd.read_json(file, lines=True) for file in files if file.stat().st_size > 0]
    if not tables:
        raise FileNotFoundError(f"no tile statistics found at {path}")
    table = pd.concat(tables, ignore_index=True)
    return table.drop_duplicates("tile_id", keep="last").reset_index(drop=True)


class Stopwatch:
    """``with Stopwatch() as watch: ...`` then ``watch.seconds``."""

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc_info) -> None:
        self.seconds = time.perf_counter() - self._start
