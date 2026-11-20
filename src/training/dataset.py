"""Training data for the segmentation head: annotated tiles and random crops of them.

Labels of a tile come from one of two places:

* ``--annotations``: ``<tile_id>.geojson`` as exported from QuPath (the layout of
  ``04_annotations/<round>/by_tile``), rasterised on load;
* ``--masks``: ready-made arrays, one per tile, in any of these forms
  (the layout of ``05_targets/<dataset_id>/by_tile``)::

      <tile_id>/instances.npy [+ types.npy, ignore.npy]
      <tile_id>.npz            with arrays instances [, types, ignore]
      <tile_id>.npy            [H, W] instance ids, or [H, W, 2] = (instance ids, class ids)
      <tile_id>.png / .tif     integer instance-id image

Every tile is resampled once to the head's working resolution and kept in memory; crops,
their HV maps and augmentations are produced on the fly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from ..image_backend.tiles import ImageFolderSource, TileRef
from ..segmenter.geojson import read_features
from ..segmenter.labels import LabelSchema
from ..segmenter.model import resize_image, working_size
from ..segmenter.postprocess import remove_small_instances
from .targets import hv_maps, rasterize_annotations

logger = logging.getLogger(__name__)


@dataclass
class TrainingTile:
    """An annotated tile at working resolution."""

    tile_id: str
    image: np.ndarray  # [H, W, 3] uint8
    instances: np.ndarray  # [H, W] int32, 0 = background
    types: np.ndarray  # [H, W] uint8 class ids, 0 = background
    ignore: np.ndarray  # [H, W] bool
    objects: list  # scipy.ndimage.find_objects(instances)

    @property
    def num_instances(self) -> int:
        return len(self.objects)


def load_training_tiles(
    images: str | Path,
    *,
    schema: LabelSchema,
    masks: str | Path | None = None,
    annotations: str | Path | None = None,
    mpp: float | None = None,
    seg_mpp: float | None = None,
    unlabeled: str = "background",
    tile_ids: Sequence[str] | None = None,
) -> list[TrainingTile]:
    """Load every image under ``images`` that has labels, resampled to the working resolution."""
    if (masks is None) == (annotations is None):
        raise ValueError("give exactly one of masks and annotations")
    source = ImageFolderSource(images, mpp=mpp)
    wanted = set(tile_ids) if tile_ids is not None else None
    tiles, unlabelled = [], []
    for ref in source.tiles:
        if wanted is not None and ref.tile_id not in wanted:
            continue
        if annotations is not None:
            path = Path(annotations) / f"{ref.tile_id}.geojson"
            if not path.exists():
                unlabelled.append(ref.tile_id)
                continue
            instances, types, ignore = rasterize_annotations(read_features(path), ref, schema, unlabeled=unlabeled)
        else:
            loaded = _read_masks(Path(masks), ref)
            if loaded is None:
                unlabelled.append(ref.tile_id)
                continue
            instances, types, ignore = loaded
        tiles.append(_to_working_resolution(ref, source.read(ref), instances, types, ignore, seg_mpp))
    if unlabelled:
        logger.info("%d images have no labels and are not used (e.g. %s)", len(unlabelled), unlabelled[:3])
    if not tiles:
        raise FileNotFoundError(f"no labelled tiles found for the images under {images}")
    return tiles


def make_training_tile(
    tile_id: str,
    image: np.ndarray,
    instances: np.ndarray,
    types: np.ndarray | None = None,
    ignore: np.ndarray | None = None,
) -> TrainingTile:
    """Wrap arrays that already are at working resolution (sides multiples of 16).

    Instance ids are renumbered 1..n, so masks with gaps in their numbering (crops of a larger
    label image, for example) are fine.
    """
    instances = remove_small_instances(np.ascontiguousarray(instances, dtype=np.int32), 1)
    types = (instances > 0).astype(np.uint8) if types is None else np.asarray(types, dtype=np.uint8)
    ignore = np.zeros(instances.shape, dtype=bool) if ignore is None else np.asarray(ignore, dtype=bool)
    return TrainingTile(tile_id, image, instances, types, ignore, ndimage.find_objects(instances))


def _to_working_resolution(ref, image, instances, types, ignore, seg_mpp) -> TrainingTile:
    if instances.shape != image.shape[:2]:
        raise ValueError(f"{ref.tile_id}: labels are {instances.shape} but the image is {image.shape[:2]}")
    height, width = working_size(ref.height, ref.width, ref.mpp, seg_mpp)
    if (height, width) != image.shape[:2]:
        image = resize_image(image, height, width)
        rows = np.minimum((np.arange(height) + 0.5) * instances.shape[0] / height, instances.shape[0] - 1).astype(int)
        cols = np.minimum((np.arange(width) + 0.5) * instances.shape[1] / width, instances.shape[1] - 1).astype(int)
        instances, types, ignore = (array[rows][:, cols] for array in (instances, types, ignore))
    return make_training_tile(ref.tile_id, image, instances, types, ignore)


def _read_masks(root: Path, ref: TileRef) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    arrays: dict[str, np.ndarray] = {}
    folder = root / ref.tile_id
    if (folder / "instances.npy").exists():
        for name in ("instances", "types", "ignore"):
            if (folder / f"{name}.npy").exists():
                arrays[name] = np.load(folder / f"{name}.npy")
    elif (root / f"{ref.tile_id}.npz").exists():
        with np.load(root / f"{ref.tile_id}.npz") as archive:
            arrays = {name: archive[name] for name in ("instances", "types", "ignore") if name in archive}
    elif (root / f"{ref.tile_id}.npy").exists():
        array = np.load(root / f"{ref.tile_id}.npy")
        arrays["instances"] = array[..., 0] if array.ndim == 3 else array
        if array.ndim == 3 and array.shape[2] > 1:
            arrays["types"] = array[..., 1]
    else:
        from PIL import Image

        for extension in (".png", ".tif", ".tiff"):
            if (root / f"{ref.tile_id}{extension}").exists():
                arrays["instances"] = np.asarray(Image.open(root / f"{ref.tile_id}{extension}"))
                break
    if "instances" not in arrays:
        return None
    instances = arrays["instances"].astype(np.int32)
    types = arrays["types"].astype(np.uint8) if "types" in arrays else (instances > 0).astype(np.uint8)
    ignore = arrays["ignore"].astype(bool) if "ignore" in arrays else np.zeros(instances.shape, dtype=bool)
    return instances, types, ignore


class CropDataset(Dataset):
    """Random crops with targets. Each index draws a fresh crop, so the length only sets the epoch size."""

    def __init__(
        self,
        tiles: Sequence[TrainingTile],
        *,
        crop_size: int = 256,
        length: int = 800,
        foreground_bias: float = 0.5,
        augment: bool = True,
        seed: int = 0,
    ):
        if crop_size % 16:
            raise ValueError("crop_size must be a multiple of 16")
        self.tiles = [_pad_to(tile, crop_size) for tile in tiles]
        self.crop_size = crop_size
        self.length = length
        self.foreground_bias = foreground_bias
        self.augment = augment
        self.seed = seed
        self.epoch = 0  # bump between epochs to draw new crops with a fixed seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng([self.seed, self.epoch, index])
        tile = self.tiles[int(rng.integers(len(self.tiles)))]
        size = self.crop_size
        height, width = tile.instances.shape
        if tile.num_instances and rng.random() < self.foreground_bias:
            # Centre the crop on a random cell so that sparse classes are seen often enough.
            box = tile.objects[int(rng.integers(tile.num_instances))]
            centre_y = (box[0].start + box[0].stop) // 2 + int(rng.integers(-size // 4, size // 4 + 1))
            centre_x = (box[1].start + box[1].stop) // 2 + int(rng.integers(-size // 4, size // 4 + 1))
            top = int(np.clip(centre_y - size // 2, 0, height - size))
            left = int(np.clip(centre_x - size // 2, 0, width - size))
        else:
            top = int(rng.integers(0, height - size + 1))
            left = int(rng.integers(0, width - size + 1))
        region = (slice(top, top + size), slice(left, left + size))

        image = tile.image[region].astype(np.float32) / 255.0
        instances, types, ignore = tile.instances[region], tile.types[region], tile.ignore[region]
        hv = hv_maps(tile.instances, region, tile.objects)
        if self.augment:
            image, instances, types, ignore, hv = _random_symmetry(rng, image, instances, types, ignore, hv)
            image = _colour_jitter(rng, image)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "np": torch.from_numpy(np.ascontiguousarray(instances > 0)).long(),
            "hv": torch.from_numpy(np.ascontiguousarray(hv)),
            "nt": torch.from_numpy(np.ascontiguousarray(types)).long(),
            "valid": torch.from_numpy(np.ascontiguousarray(~ignore)).float(),
        }


def _pad_to(tile: TrainingTile, size: int) -> TrainingTile:
    """Tiles smaller than a crop are padded with white, ignored pixels."""
    height, width = tile.instances.shape
    pad_h, pad_w = max(0, size - height), max(0, size - width)
    if not (pad_h or pad_w):
        return tile
    pad = ((0, pad_h), (0, pad_w))
    return make_training_tile(
        tile.tile_id,
        np.pad(tile.image, (*pad, (0, 0)), constant_values=255),
        np.pad(tile.instances, pad),
        np.pad(tile.types, pad),
        np.pad(tile.ignore, pad, constant_values=True),
    )


def _random_symmetry(rng, image, instances, types, ignore, hv):
    """One of the eight symmetries of the square; the HV maps are vector-valued and change sign/order with it."""
    if rng.random() < 0.5:  # transpose: x and y swap
        image = image.transpose(1, 0, 2)
        instances, types, ignore = instances.T, types.T, ignore.T
        hv = hv[::-1].transpose(0, 2, 1)
    if rng.random() < 0.5:  # mirror left-right
        image, instances, types, ignore = image[:, ::-1], instances[:, ::-1], types[:, ::-1], ignore[:, ::-1]
        hv = hv[:, :, ::-1] * np.array([-1.0, 1.0], dtype=np.float32)[:, None, None]
    if rng.random() < 0.5:  # mirror top-bottom
        image, instances, types, ignore = image[::-1], instances[::-1], types[::-1], ignore[::-1]
        hv = hv[:, ::-1] * np.array([1.0, -1.0], dtype=np.float32)[:, None, None]
    return image, instances, types, ignore, hv


def _colour_jitter(rng, image: np.ndarray) -> np.ndarray:
    """Mild brightness, contrast, saturation and per-channel gain changes (stain and scanner variation)."""
    image = image * rng.uniform(0.9, 1.1, size=3).astype(np.float32)
    image = image * rng.uniform(0.85, 1.15)
    mean = image.mean()
    image = (image - mean) * rng.uniform(0.85, 1.15) + mean
    grey = image.mean(axis=2, keepdims=True)
    image = (image - grey) * rng.uniform(0.8, 1.2) + grey
    return np.clip(image, 0.0, 1.0).astype(np.float32)
