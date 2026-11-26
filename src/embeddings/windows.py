"""Input adapter: from a 1024 px annotation tile to what a base model reads.

The pathologist's tile is always 1024x1024 px at level 0 (~0.25 um/px at 40x), while SSL
encoders were trained on 224 px inputs at ~0.5 um/px with patch size 14 or 16. A *window*
is one model read: a box inside the tile plus the size it is resampled to. This follows
``docs/architecture/data-layout.md`` ("Annotation vs model size") and produces rows for
``model_windows.parquet``.

Window modes
------------
``grid``       native-size windows at the model's resolution, on the smallest grid covering
               the tile. Level-0 side: ``S0 = size_model * mpp_model / mpp_slide``
               (224 px @ 0.5 um/px on a 0.25 um/px slide -> 448 px, 3x3 windows per tile).
``center``     one such window in the middle of the tile.
``full``       the whole tile in one window, resampled to ``mpp_model`` and rounded to a
               multiple of the patch size (1024 px -> 518 px for patch 14). Needs an encoder
               that accepts other input sizes (all registered ones do).
``thumbnail``  the whole tile squeezed into one native-size window (cheapest, coarsest).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from ..image_backend.tiles import TileRef

if TYPE_CHECKING:
    import pandas as pd

WINDOW_MODES = ("grid", "center", "full", "thumbnail")


@dataclass(frozen=True)
class Window:
    """One model read. ``x``/``y`` are level-0 slide coordinates, like the tile's."""

    window_id: str
    tile_id: str
    x: int
    y: int
    width_level0: int
    height_level0: int
    width_model: int
    height_model: int
    mpp_model: float | None  # resolution the network actually sees; None if the slide mpp is unknown
    role: str  # "embed" | "segment"


@dataclass(frozen=True)
class WindowPlanner:
    """Window layout for one base model; picklable, so DataLoader workers can carry it."""

    mode: str = "grid"
    patch_size: int = 14
    native_size: int = 224
    mpp_model: float = 0.5
    overlap: float = 0.0  # grid mode: minimum fractional overlap between neighbouring windows
    default_mpp: float | None = None  # slide resolution to assume when a tile does not carry one
    role: str = "embed"

    def __post_init__(self) -> None:
        if self.mode not in WINDOW_MODES:
            raise ValueError(f"window mode must be one of {WINDOW_MODES}, got {self.mode!r}")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {self.overlap}")
        if self.native_size % self.patch_size:
            raise ValueError(f"native size {self.native_size} is not a multiple of the patch size {self.patch_size}")

    def plan(self, tile: TileRef) -> list[Window]:
        """Windows for ``tile``, all inside its box."""
        mpp_slide = tile.mpp if tile.mpp is not None else self.default_mpp
        if self.mode == "thumbnail":
            size = (self.native_size, self.native_size)
            return [self._window(tile, 0, 0, tile.width, tile.height, *size, mpp_slide)]
        if mpp_slide is None:
            raise ValueError(
                f"tile {tile.tile_id} has no mpp; window mode {self.mode!r} needs the slide resolution "
                "(pass --mpp, e.g. 0.25 for 40x)"
            )
        scale = mpp_slide / self.mpp_model  # model pixels per level-0 pixel
        if self.mode == "full":
            width = _round_to_multiple(tile.width * scale, self.patch_size)
            height = _round_to_multiple(tile.height * scale, self.patch_size)
            return [self._window(tile, 0, 0, tile.width, tile.height, width, height, mpp_slide)]

        side = min(int(round(self.native_size / scale)), tile.width, tile.height)
        if self.mode == "center":
            offsets_x = [(tile.width - side) // 2]
            offsets_y = [(tile.height - side) // 2]
        else:
            offsets_x = _axis_offsets(tile.width, side, self.overlap)
            offsets_y = _axis_offsets(tile.height, side, self.overlap)
        native = (self.native_size, self.native_size)
        return [self._window(tile, dx, dy, side, side, *native, mpp_slide) for dy in offsets_y for dx in offsets_x]

    def _window(
        self,
        tile: TileRef,
        dx: int,
        dy: int,
        width0: int,
        height0: int,
        width_model: int,
        height_model: int,
        mpp_slide: float | None,
    ) -> Window:
        mpp = None if mpp_slide is None else mpp_slide * width0 / width_model
        core = tile.tile_id[3:] if tile.tile_id.startswith("tl_") else tile.tile_id
        shape0 = f"s{width0}" if width0 == height0 else f"s{width0}x{height0}"
        shape_model = f"{width_model}" if width_model == height_model else f"{width_model}x{height_model}"
        resolution = "" if mpp is None else f"_mpp{_format_mpp(mpp)}"
        return Window(
            window_id=f"win_{core}_dx{dx}_dy{dy}_{shape0}_{shape_model}{resolution}",
            tile_id=tile.tile_id,
            x=tile.x + dx,
            y=tile.y + dy,
            width_level0=width0,
            height_level0=height0,
            width_model=width_model,
            height_model=height_model,
            mpp_model=mpp,
            role=self.role,
        )


def crop_window(tile_image: np.ndarray, tile: TileRef, window: Window) -> np.ndarray:
    """Cut ``window`` out of the tile image and resample it to the model size (RGB uint8)."""
    import cv2

    if tile_image.shape[:2] != (tile.height, tile.width):
        raise ValueError(
            f"tile {tile.tile_id}: image is {tile_image.shape[1]}x{tile_image.shape[0]} px "
            f"but the tile is {tile.width}x{tile.height} px at level 0"
        )
    dx, dy = window.x - tile.x, window.y - tile.y
    crop = tile_image[dy : dy + window.height_level0, dx : dx + window.width_level0]
    if crop.shape[:2] == (window.height_model, window.width_model):
        return np.ascontiguousarray(crop)
    shrinking = window.width_model < window.width_level0
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(crop, (window.width_model, window.height_model), interpolation=interpolation)


def windows_table(windows: Sequence[Window], prep_id: str | None = None) -> "pd.DataFrame":
    """Rows for ``model_windows.parquet`` (see ``schemas/tiles.parquet.md``).

    ``size_level0``/``size_model`` hold the window width; the two ``height_*`` columns extend
    the schema for the rare non-square window and equal the sizes otherwise.
    """
    import pandas as pd

    return pd.DataFrame(
        {
            "window_id": [w.window_id for w in windows],
            "tile_id": [w.tile_id for w in windows],
            "prep_id": [prep_id] * len(windows),
            "role": [w.role for w in windows],
            "x": [w.x for w in windows],
            "y": [w.y for w in windows],
            "size_level0": [w.width_level0 for w in windows],
            "size_model": [w.width_model for w in windows],
            "mpp_model": [w.mpp_model for w in windows],
            "height_level0": [w.height_level0 for w in windows],
            "height_model": [w.height_model for w in windows],
        }
    )


def _axis_offsets(length: int, side: int, overlap: float) -> list[int]:
    """Evenly spread window offsets covering ``[0, length)`` with at least ``overlap`` between neighbours."""
    if side >= length:
        return [0]
    stride = max(1.0, side * (1.0 - overlap))
    count = math.ceil((length - side) / stride) + 1
    return [int(round(i * (length - side) / (count - 1))) for i in range(count)]


def _round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _format_mpp(mpp: float) -> str:
    """0.5 -> ``0p5``, 0.2471 -> ``0p247``."""
    return f"{mpp:.3f}".rstrip("0").rstrip(".").replace(".", "p")
