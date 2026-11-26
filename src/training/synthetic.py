"""Synthetic H&E-like tiles with known cells: for tests, smoke tests and trying the pipeline without data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def synthetic_cells(
    size: int = 256, count: int = 14, seed: int = 0, radius: tuple[float, float] = (9, 20)
) -> tuple[np.ndarray, np.ndarray]:
    """A toy H&E-like tile: dark purple discs ("nuclei") on a pink background.

    Returns ``(image uint8 [size, size, 3], instances int32 [size, size])``. Discs may touch
    but never overlap, so the instance map is unambiguous.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    instances = np.zeros((size, size), dtype=np.int32)
    centers: list[tuple[float, float, float]] = []
    attempts = 0
    while len(centers) < count and attempts < max(4000, 40 * count):
        attempts += 1
        r = rng.uniform(*radius)
        cx, cy = rng.uniform(r + 1, size - r - 1, size=2)
        if all((cx - ox) ** 2 + (cy - oy) ** 2 >= (r + orr) ** 2 for ox, oy, orr in centers):
            centers.append((cx, cy, r))
            top, bottom = max(0, int(np.floor(cy - r))), min(size, int(np.ceil(cy + r)) + 1)
            left, right = max(0, int(np.floor(cx - r))), min(size, int(np.ceil(cx + r)) + 1)
            inside = (xx[top:bottom, left:right] - cx) ** 2 + (yy[top:bottom, left:right] - cy) ** 2 < r**2
            instances[top:bottom, left:right][inside] = len(centers)
    image = np.empty((size, size, 3), dtype=np.float32)
    image[:] = (236, 190, 214)  # eosin-like background
    image[instances > 0] = (88, 52, 138)  # haematoxylin-like nuclei
    image += rng.normal(0, 6, size=image.shape)
    return np.clip(image, 0, 255).astype(np.uint8), instances


def write_demo_dataset(
    root: str | Path,
    *,
    tiles: int = 6,
    size: int = 256,
    count: int = 14,
    mpp: float = 0.25,
    seed: int = 0,
    radius: tuple[float, float] = (9, 20),
    slide_id: str = "DEMO",
    class_name: str = "meg_norm",
) -> list[str]:
    """Write ``tiles`` synthetic tiles with labels in every format the training script reads.

    ``root/images/<tile_id>.png`` the tiles, named like real tile ids (``tl_DEMO_x4096_y8192_s256``)
    ``root/annotations/<tile_id>.geojson`` QuPath-style polygons in slide coordinates (``--annotations``)
    ``root/masks/<tile_id>.npz`` instance masks (``--masks``)
    ``root/dataset.json`` resolution and tile list

    Returns the tile ids.
    """
    from PIL import Image

    from ..segmenter import LabelSchema, cells_to_feature_collection, instances_to_cells, write_geojson

    root = Path(root)
    for sub in ("images", "annotations", "masks"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    schema = LabelSchema.single_class(class_name)
    tile_ids = []
    for index in range(tiles):
        image, instances = synthetic_cells(size, count, seed + index, radius)
        x, y = 4096 + size * index, 8192
        tile_id = f"tl_{slide_id}_x{x}_y{y}_s{size}"
        Image.fromarray(image).save(root / "images" / f"{tile_id}.png")
        np.savez_compressed(root / "masks" / f"{tile_id}.npz", instances=instances)
        cells = instances_to_cells(instances, origin=(x, y), labels=schema, simplify=0.5)
        write_geojson(
            root / "annotations" / f"{tile_id}.geojson", cells_to_feature_collection(cells, schema, tile_id=tile_id)
        )
        tile_ids.append(tile_id)
    info = {"mpp": mpp, "tile_size": size, "class_name": class_name, "tiles": tile_ids}
    (root / "dataset.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return tile_ids
