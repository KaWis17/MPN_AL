"""Tile identity and tile readers.

A *tile* is the annotation field of view defined in ``docs/architecture/data-layout.md``:
a box in level-0 slide coordinates, keyed by ``tile_id``. Everything downstream
(embeddings, predictions, GeoJSON) reports coordinates in that level-0 frame.

Two readers cover the cases the base models need today:

* :class:`ImageFolderSource` - tiles exported as image files (``<tile_id>.png``), e.g. the
  public annotated sets used before the cohort WSIs arrive.
* :class:`WsiTileSource` - tiles listed in ``tiles.parquet`` and read from the WSI on demand
  ("pixels stay on the WSI").

Both yield RGB ``uint8`` arrays at level-0 resolution, shaped ``[height, width, 3]``.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# `{slide_id}_x{x}_y{y}_s{size}` with the optional `tl_` prefix used in the data-layout examples.
_TILE_ID_RE = re.compile(r"^(?:tl_)?(?P<slide_id>.+)_x(?P<x>\d+)_y(?P<y>\d+)_s(?P<size>\d+)$")


@dataclass(frozen=True)
class TileRef:
    """One annotation field of view. ``x``/``y`` are the level-0 top-left corner in the slide."""

    tile_id: str
    x: int = 0
    y: int = 0
    width: int = 1024
    height: int = 1024
    mpp: float | None = None
    slide_id: str | None = None
    patient_id: str | None = None
    prep_id: str | None = None
    path: str | None = None


def parse_tile_id(tile_id: str) -> dict[str, Any] | None:
    """Split ``[tl_]{slide_id}_x{x}_y{y}_s{size}`` into its parts, or return ``None``."""
    match = _TILE_ID_RE.match(tile_id)
    if match is None:
        return None
    return {
        "slide_id": match["slide_id"],
        "x": int(match["x"]),
        "y": int(match["y"]),
        "size": int(match["size"]),
    }


def select_tiles(
    tiles: Sequence[TileRef],
    *,
    tile_ids: Iterable[str] | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    limit: int | None = None,
) -> list[TileRef]:
    """Filter by id, keep one shard (for Slurm job arrays) and cap the count, in that order."""
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    selected = list(tiles)
    if tile_ids is not None:
        wanted = set(tile_ids)
        selected = [t for t in selected if t.tile_id in wanted]
    if num_shards > 1:
        selected = selected[shard_index::num_shards]
    if limit is not None:
        selected = selected[:limit]
    return selected


class TileSource(ABC):
    """A catalogue of tiles plus the means to read their pixels."""

    @property
    @abstractmethod
    def tiles(self) -> list[TileRef]:
        """All tiles of this source, in a stable order."""

    @abstractmethod
    def read(self, tile: TileRef) -> np.ndarray:
        """Return the tile as RGB ``uint8`` ``[height, width, 3]`` at level-0 resolution."""

    supports_margin = False  # True when pixels around a tile can be read too (``read_with_margin``)

    def read_with_margin(self, tile: TileRef, margin: int) -> np.ndarray:
        """The tile plus ``margin`` level-0 px of surroundings on every side, ``[h + 2m, w + 2m, 3]``.

        Only sources that read from a whole slide can do this; the tile's own pixels sit at ``[margin:-margin]``.
        """
        if margin == 0:
            return self.read(tile)
        raise NotImplementedError(
            f"{type(self).__name__} cannot read around a tile; use --tiles-parquet (WSI) for a context margin"
        )

    def __len__(self) -> int:
        return len(self.tiles)

    def __iter__(self) -> Iterator[TileRef]:
        return iter(self.tiles)


class ArrayTileSource(TileSource):
    """In-memory tiles; used by tests and notebooks."""

    def __init__(self, images: dict[str, np.ndarray], *, mpp: float | None = None, prep_id: str | None = None):
        self._images = {tile_id: _as_rgb_uint8(image) for tile_id, image in images.items()}
        self._tiles = [
            _ref_from_name(tile_id, image.shape[1], image.shape[0], mpp=mpp, prep_id=prep_id, path=None)
            for tile_id, image in self._images.items()
        ]

    @property
    def tiles(self) -> list[TileRef]:
        return self._tiles

    def read(self, tile: TileRef) -> np.ndarray:
        return self._images[tile.tile_id]


class ImageFolderSource(TileSource):
    """Tiles stored as image files; ``tile_id`` is the file stem.

    If the stem follows ``[tl_]{slide_id}_x{x}_y{y}_s{size}`` the level-0 origin is taken from
    it, so predictions land in slide coordinates. Otherwise the origin is ``(0, 0)`` and
    coordinates are tile-local.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        mpp: float | None = None,
        prep_id: str | None = None,
        recursive: bool = False,
        extensions: Sequence[str] = IMAGE_EXTENSIONS,
    ):
        from PIL import Image

        root = Path(root)
        if root.is_file():
            files = [root]
        elif root.is_dir():
            candidates = root.rglob("*") if recursive else root.iterdir()
            files = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in extensions)
        else:
            raise FileNotFoundError(f"no such image file or directory: {root}")
        if not files:
            raise FileNotFoundError(f"no images with extensions {tuple(extensions)} under {root}")

        stems = [p.stem for p in files]
        duplicates = sorted({s for s in stems if stems.count(s) > 1})
        if duplicates:
            raise ValueError(f"tile ids must be unique, but these file stems repeat: {duplicates[:5]}")

        self.root = root
        self._tiles = []
        for path in files:
            with Image.open(path) as image:
                width, height = image.size
            self._tiles.append(_ref_from_name(path.stem, width, height, mpp=mpp, prep_id=prep_id, path=str(path)))

    @property
    def tiles(self) -> list[TileRef]:
        return self._tiles

    def read(self, tile: TileRef) -> np.ndarray:
        from PIL import Image

        if tile.path is None:
            raise ValueError(f"tile {tile.tile_id} has no file path")
        with Image.open(tile.path) as image:
            return np.asarray(image.convert("RGB"))


class WsiTileSource(TileSource):
    """Tiles from ``tiles.parquet``, read from the WSI at level 0.

    Slide files are resolved through ``slides.parquet`` (``slide_id`` -> ``path``; relative
    paths are taken relative to that file) or, failing that, by searching ``wsi_root`` for
    ``<slide_id>.*``. Slides are opened lazily, so the source can be pickled into DataLoader
    workers.

    ``opener`` maps a file path to an object with ``read_region((x, y), level, (w, h))``
    returning an RGBA/RGB PIL image (the OpenSlide API). It defaults to OpenSlide, with
    TiffSlide as a fallback.
    """

    def __init__(
        self,
        tiles_parquet: str | Path,
        *,
        slides_parquet: str | Path | None = None,
        wsi_root: str | Path | None = None,
        qc_pass_only: bool = True,
        split: str | None = None,
        default_mpp: float | None = None,
        opener: Callable[[str], Any] | None = None,
        max_open_slides: int = 4,
    ):
        import pandas as pd

        table = pd.read_parquet(tiles_parquet)
        missing = {"tile_id", "slide_id", "x", "y"} - set(table.columns)
        if missing:
            raise ValueError(f"{tiles_parquet} lacks required columns: {sorted(missing)}")
        if qc_pass_only and "qc_pass" in table.columns:
            table = table[table["qc_pass"].astype(bool)]
        if split is not None:
            if "split" not in table.columns:
                raise ValueError(f"{tiles_parquet} has no 'split' column to filter on")
            table = table[table["split"] == split]

        self._slide_paths = _slide_paths(slides_parquet, wsi_root, table["slide_id"].unique())
        self._opener = opener
        self._max_open = max_open_slides
        self._open: OrderedDict[str, Any] = OrderedDict()

        def optional(row: Any, column: str) -> Any:
            value = getattr(row, column, None)
            return None if value is None or pd.isna(value) else value

        self._tiles = []
        for row in table.itertuples(index=False):
            mpp = optional(row, "mpp")
            self._tiles.append(
                TileRef(
                    tile_id=str(row.tile_id),
                    x=int(row.x),
                    y=int(row.y),
                    width=int(optional(row, "width") or 1024),
                    height=int(optional(row, "height") or 1024),
                    mpp=float(mpp) if mpp is not None else default_mpp,
                    slide_id=str(row.slide_id),
                    patient_id=optional(row, "patient_id"),
                    prep_id=optional(row, "prep_id"),
                    path=self._slide_paths.get(str(row.slide_id)),
                )
            )

    @property
    def tiles(self) -> list[TileRef]:
        return self._tiles

    def read(self, tile: TileRef) -> np.ndarray:
        if tile.path is None:
            raise FileNotFoundError(
                f"no WSI file known for slide {tile.slide_id!r}; pass slides_parquet or wsi_root"
            )
        slide = self._slide(tile.path)
        region = slide.read_region((tile.x, tile.y), 0, (tile.width, tile.height))
        return _composite_on_white(region)

    supports_margin = True

    def read_with_margin(self, tile: TileRef, margin: int) -> np.ndarray:
        """Surroundings outside the scanned area (or the slide) come back white, as in :meth:`read`."""
        if margin < 0:
            raise ValueError(f"margin must be >= 0, got {margin}")
        if tile.path is None:
            raise FileNotFoundError(f"no WSI file known for slide {tile.slide_id!r}; pass slides_parquet or wsi_root")
        slide = self._slide(tile.path)
        size = (tile.width + 2 * margin, tile.height + 2 * margin)
        return _composite_on_white(slide.read_region((tile.x - margin, tile.y - margin), 0, size))

    def _slide(self, path: str) -> Any:
        if path in self._open:
            self._open.move_to_end(path)
            return self._open[path]
        slide = (self._opener or _default_opener())(path)
        self._open[path] = slide
        while len(self._open) > self._max_open:
            _, stale = self._open.popitem(last=False)
            close = getattr(stale, "close", None)
            if close is not None:
                close()
        return slide

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_open"] = OrderedDict()  # file handles do not cross process boundaries
        return state


def _ref_from_name(
    tile_id: str, width: int, height: int, *, mpp: float | None, prep_id: str | None, path: str | None
) -> TileRef:
    parsed = parse_tile_id(tile_id) or {}
    return TileRef(
        tile_id=tile_id,
        x=parsed.get("x", 0),
        y=parsed.get("y", 0),
        width=width,
        height=height,
        mpp=mpp,
        slide_id=parsed.get("slide_id"),
        prep_id=prep_id,
        path=path,
    )


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected an RGB uint8 array [H, W, 3], got {image.dtype} {image.shape}")
    return image


def _composite_on_white(region: Any) -> np.ndarray:
    """OpenSlide returns RGBA with transparent pixels outside the scanned area; paint those white."""
    array = np.asarray(region)
    if array.ndim == 3 and array.shape[2] == 4:
        alpha = array[..., 3:4].astype(np.float32) / 255.0
        rgb = array[..., :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
        return rgb.round().astype(np.uint8)
    return _as_rgb_uint8(array[..., :3])


def _slide_paths(
    slides_parquet: str | Path | None, wsi_root: str | Path | None, slide_ids: Iterable[str]
) -> dict[str, str]:
    paths: dict[str, str] = {}
    if slides_parquet is not None:
        import pandas as pd

        slides = pd.read_parquet(slides_parquet)
        if not {"slide_id", "path"} <= set(slides.columns):
            raise ValueError(f"{slides_parquet} must have 'slide_id' and 'path' columns")
        base = Path(slides_parquet).parent
        for slide_id, path in zip(slides["slide_id"], slides["path"]):
            resolved = Path(path)
            paths[str(slide_id)] = str(resolved if resolved.is_absolute() else base / resolved)
    if wsi_root is not None:
        root = Path(wsi_root)
        for slide_id in map(str, slide_ids):
            if slide_id not in paths:
                match = next(iter(sorted(root.rglob(f"{slide_id}.*"))), None)
                if match is not None:
                    paths[slide_id] = str(match)
    return paths


def _default_opener() -> Callable[[str], Any]:
    try:
        import openslide

        return openslide.OpenSlide
    except ImportError:
        pass
    try:
        from tiffslide import TiffSlide

        return TiffSlide
    except ImportError as exc:
        raise ImportError(
            "reading tiles from WSIs needs OpenSlide: pip install 'mpn-al[wsi]' "
            "(openslide-python + openslide-bin), or install tiffslide"
        ) from exc
