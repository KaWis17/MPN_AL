"""Training targets for the segmentation head.

Two steps of the data layout meet here:

* ``04_annotations`` -> masks: :func:`rasterize_annotations` turns the QuPath GeoJSON of a
  tile into an instance map, a type map and an ignore mask;
* masks -> ``05_targets``: :func:`hv_maps` derives the horizontal/vertical distance maps the
  decoder regresses (HoVer-Net's target for separating touching cells).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import cv2
import numpy as np
from scipy import ndimage

from ..image_backend.tiles import TileRef
from ..segmenter.geojson import feature_class_name, feature_polygons
from ..segmenter.labels import LabelSchema

logger = logging.getLogger(__name__)


def hv_maps(
    instances: np.ndarray,
    region: tuple[slice, slice] | None = None,
    objects: Sequence[tuple[slice, slice] | None] | None = None,
) -> np.ndarray:
    """Horizontal and vertical maps ``[2, h, w]`` for ``region`` of an instance map.

    Inside each instance the horizontal map runs from -1 at its left edge through 0 at its
    centre of mass to +1 at its right edge; the vertical map does the same top to bottom.
    Background is 0. An instance cut by the region border still gets the values of the whole
    instance, so a crop looks like the corresponding part of the full tile.

    ``objects`` is ``scipy.ndimage.find_objects(instances)``; pass it to avoid recomputing it
    for every crop of the same tile.
    """
    height, width = instances.shape
    if region is None:
        region = (slice(0, height), slice(0, width))
    top, left = region[0].start, region[1].start
    out_h, out_w = region[0].stop - top, region[1].stop - left
    maps = np.zeros((2, out_h, out_w), dtype=np.float32)
    if objects is None:
        objects = ndimage.find_objects(instances)
    for label in np.unique(instances[region]):
        if label == 0:
            continue
        box = objects[label - 1]
        ys, xs = np.nonzero(instances[box] == label)
        dx = _signed_unit(xs - np.round(xs.mean()))
        dy = _signed_unit(ys - np.round(ys.mean()))
        rows, cols = ys + box[0].start - top, xs + box[1].start - left
        inside = (rows >= 0) & (rows < out_h) & (cols >= 0) & (cols < out_w)
        maps[0, rows[inside], cols[inside]] = dx[inside]
        maps[1, rows[inside], cols[inside]] = dy[inside]
    return maps


def _signed_unit(offsets: np.ndarray) -> np.ndarray:
    """Scale negative offsets to [-1, 0) and positive ones to (0, 1], each side on its own."""
    values = offsets.astype(np.float32)
    low, high = values.min(), values.max()
    if low < 0:
        values[values < 0] /= -low
    if high > 0:
        values[values > 0] /= high
    return values


def rasterize_annotations(
    features: Sequence[dict[str, Any]],
    tile: TileRef,
    schema: LabelSchema,
    *,
    unlabeled: str = "background",
    unknown: str = "ignore",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterise a tile's GeoJSON annotations into ``(instances, types, ignore)``.

    * a polygon whose class is in the schema becomes one instance of that class;
    * a polygon with an ``ignore`` name masks its area out of the loss;
    * polygons with a ``roi`` name delimit what was annotated: if a tile has any, everything
      outside them is ignored;
    * a polygon with a class the schema does not know is ignored (``unknown="ignore"``) or
      skipped (``unknown="skip"``).

    ``unlabeled`` says what pixels without any annotation are: ``"background"`` (the tile was
    annotated exhaustively) or ``"ignore"`` (only some cells were drawn; note that a head
    cannot learn what background looks like from such tiles alone).

    Coordinates are level-0 slide coordinates; the tile's origin is subtracted.
    """
    if unlabeled not in ("background", "ignore") or unknown not in ("ignore", "skip"):
        raise ValueError("unlabeled must be 'background' or 'ignore'; unknown must be 'ignore' or 'skip'")
    shape = (tile.height, tile.width)
    instances = np.zeros(shape, dtype=np.int32)
    ignore = np.zeros(shape, dtype=np.uint8)
    roi = np.zeros(shape, dtype=np.uint8)
    class_of_instance = [0]
    has_roi = False
    unknown_names: set[str] = set()
    origin = np.array([tile.x + 0.5, tile.y + 0.5])  # pixel i covers [i, i + 1): its centre is at i + 0.5

    for feature in features:
        name = feature_class_name(feature)
        role, label = schema.role_of(name)
        if role == "unknown":
            unknown_names.add(str(name))
            if unknown == "skip":
                continue
            role = "ignore"
        for rings in feature_polygons(feature):
            points = [np.round(ring - origin).astype(np.int32) for ring in rings]
            if role == "class":
                cv2.fillPoly(instances, [points[0]], len(class_of_instance))
                if len(points) > 1:
                    cv2.fillPoly(instances, points[1:], 0)
                class_of_instance.append(label.id)
            elif role == "ignore":
                cv2.fillPoly(ignore, [points[0]], 1)
            else:
                cv2.fillPoly(roi, [points[0]], 1)
                has_roi = True

    if unknown_names:
        logger.warning("tile %s: annotation classes not in the label schema: %s", tile.tile_id, sorted(unknown_names))
    types = np.asarray(class_of_instance, dtype=np.uint8)[instances]
    ignore_mask = ignore.astype(bool)
    if has_roi:
        ignore_mask |= ~roi.astype(bool)
    if unlabeled == "ignore":
        ignore_mask |= instances == 0
    # Instances drawn entirely outside the tile, or painted over, leave gaps in the numbering.
    present = np.unique(instances)
    renumber = np.zeros(len(class_of_instance), dtype=np.int32)
    renumber[present[present > 0]] = np.arange(1, int((present > 0).sum()) + 1)
    return renumber[instances], types, ignore_mask
