"""From network outputs to cell instances, polygons and per-tile statistics.

Instance decoding is HoVer-Net's: the horizontal/vertical maps change sign abruptly between
two touching cells, so their gradient gives an edge energy; cell interiors (foreground with
low edge energy) become markers, and markers are grown back over the edge band by a
watershed. The watershed here is a level-set flood built on ``cv2.dilate`` so that the
pipeline needs no scikit-image.

Size parameters are given in microns and converted with the working resolution, so one
configuration serves 40x and 20x.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from scipy import ndimage

from .labels import LabelSchema

_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
_SQUARE = np.ones((3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class PostprocessConfig:
    prob_threshold: float = 0.5  # cell probability above which a pixel is foreground
    marker_threshold: float = 0.4  # edge energy below which a foreground pixel is a cell interior
    sobel_um: float = 5.25  # Sobel aperture: 21 px at 0.25 um/px, HoVer-Net's 40x setting
    opening_um: float = 1.25  # marker clean-up: 5 px at 0.25 um/px
    min_marker_um2: float = 0.6  # 10 px at 0.25 um/px
    min_area_um2: float = 3.0  # smaller instances are dropped
    levels: int = 12  # flood levels between the marker threshold and the top of the edge energy
    reference_mpp: float = 0.25  # resolution assumed when the tile does not carry one


@dataclass
class Cell:
    """One segmented cell; geometry is in level-0 slide coordinates."""

    label: int  # id in the instance map
    polygon: np.ndarray  # [n, 2] float32 (x, y), open ring
    centroid: tuple[float, float]
    area: float  # level-0 px^2
    class_id: int | None = None
    class_name: str | None = None
    prob: float | None = None  # mean cell probability inside the instance
    class_prob: float | None = None  # share of the type vote won by class_id
    touches_border: bool = False


def flood_labels(markers: np.ndarray, elevation: np.ndarray, mask: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    """Marker-controlled watershed by level-set flooding.

    For each level in increasing order, labels spread through neighbouring pixels of ``mask``
    whose ``elevation`` is at most that level, until nothing changes; a last pass ignores the
    elevation. Where two labels meet they stop, so basins split along ridges of the elevation.
    Cross- and square-shaped steps alternate to keep the growth close to isotropic.
    """
    labels = markers.astype(np.float32)  # cv2.dilate has no int32 path; float32 is exact for label ids
    step = 0
    for level in [*levels, np.inf]:
        allowed = mask & (elevation <= level)
        misses = 0
        while misses < 2:  # stop once neither step shape reaches a new pixel
            grown = cv2.dilate(labels, _CROSS if step % 2 == 0 else _SQUARE)
            step += 1
            update = allowed & (labels == 0) & (grown > 0)
            if update.any():
                labels[update] = grown[update]
                misses = 0
            else:
                misses += 1
    return labels.astype(np.int32)


def hv_to_instances(
    prob: np.ndarray,
    hv: np.ndarray,
    *,
    mpp: float | None = None,
    config: PostprocessConfig = PostprocessConfig(),
) -> np.ndarray:
    """Decode a cell-probability map ``[H, W]`` and HV maps ``[2, H, W]`` into an instance map."""
    mpp = mpp or config.reference_mpp
    pixel_area = mpp * mpp
    min_marker = max(1, int(round(config.min_marker_um2 / pixel_area)))
    foreground = _remove_small(prob >= config.prob_threshold, min_marker)
    if not foreground.any():
        return np.zeros(prob.shape, dtype=np.int32)

    aperture = int(np.clip(_odd(config.sobel_um / mpp), 3, 31))
    grad_h = cv2.Sobel(_minmax(hv[0]), cv2.CV_32F, 1, 0, ksize=aperture)
    grad_v = cv2.Sobel(_minmax(hv[1]), cv2.CV_32F, 0, 1, ksize=aperture)
    # Between touching cells the maps jump from +1 to -1: the most negative gradient, i.e. energy ~1.
    edge = np.maximum(1.0 - _minmax(grad_h), 1.0 - _minmax(grad_v))
    edge[~foreground] = 0.0

    interior = ndimage.binary_fill_holes(foreground & (edge < config.marker_threshold))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(config.opening_um / mpp),) * 2)
    interior = cv2.morphologyEx(interior.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    markers = _label(_remove_small(interior, min_marker))

    elevation = cv2.GaussianBlur(edge, (3, 3), 0)
    levels = np.linspace(config.marker_threshold, 1.0, config.levels)
    instances = flood_labels(markers, elevation, foreground, levels)

    # Foreground blobs that produced no marker are kept as single cells instead of being lost.
    orphans = _label(foreground & (instances == 0))
    instances = np.where(orphans > 0, orphans + instances.max(), instances)
    return remove_small_instances(instances, max(1, int(round(config.min_area_um2 / pixel_area))))


def remove_small_instances(instances: np.ndarray, min_pixels: int) -> np.ndarray:
    """Drop instances smaller than ``min_pixels`` and renumber the rest 1..n."""
    counts = np.bincount(instances.ravel())
    keep = counts >= min_pixels
    keep[0] = False
    renumber = np.zeros(len(counts), dtype=np.int32)
    renumber[keep] = np.arange(1, int(keep.sum()) + 1)
    return renumber[instances]


def instances_to_cells(
    instances: np.ndarray,
    *,
    scale: tuple[float, float] = (1.0, 1.0),
    origin: tuple[float, float] = (0.0, 0.0),
    prob: np.ndarray | None = None,
    type_prob: np.ndarray | None = None,
    labels: LabelSchema | None = None,
    simplify: float = 0.75,
) -> list[Cell]:
    """Outline every instance and attach class and confidence.

    ``scale`` is level-0 pixels per map pixel ``(x, y)`` and ``origin`` the level-0 position of
    the map's top-left corner. ``type_prob`` ``[K + 1, H, W]`` votes for the class (channel 0 is
    background and does not vote); without it every cell gets the schema's first class.
    ``simplify`` is the polygon tolerance in map pixels.
    """
    count = int(instances.max())
    if count == 0:
        return []
    ids = np.arange(1, count + 1)
    areas = np.bincount(instances.ravel(), minlength=count + 1)[1:]
    centroids = ndimage.center_of_mass(instances > 0, instances, ids)
    mean_prob = ndimage.mean(prob, instances, ids) if prob is not None else None
    if type_prob is not None and type_prob.shape[0] > 1:
        votes = np.stack([ndimage.sum_labels(channel, instances, ids) for channel in type_prob[1:]], axis=1)
        class_ids = votes.argmax(axis=1) + 1
        class_probs = votes.max(axis=1) / np.maximum(votes.sum(axis=1), 1e-9)
    else:
        class_ids = np.ones(count, dtype=np.int64) if labels is not None else None
        class_probs = None

    height, width = instances.shape
    sx, sy = scale
    cells = []
    for index, box in enumerate(ndimage.find_objects(instances)):
        if box is None or areas[index] == 0:
            continue
        label = index + 1
        mask = np.pad((instances[box] == label).astype(np.uint8), 1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        if simplify > 0:
            simplified = cv2.approxPolyDP(contour, simplify, True)
            contour = simplified if len(simplified) >= 3 else contour
        if len(contour) < 3:  # a one-pixel-wide sliver has no outline; see drop_instances_without_cells
            continue
        # Contour points are pixel indices in the padded crop; +0.5 moves them to pixel centres.
        points = contour[:, 0, :].astype(np.float32) - 1.0 + (box[1].start + 0.5, box[0].start + 0.5)
        polygon = points * (sx, sy) + origin
        class_id = int(class_ids[index]) if class_ids is not None else None
        cy, cx = centroids[index]
        cells.append(
            Cell(
                label=label,
                polygon=polygon.astype(np.float32),
                centroid=((cx + 0.5) * sx + origin[0], (cy + 0.5) * sy + origin[1]),
                area=float(areas[index] * sx * sy),
                class_id=class_id,
                class_name=labels.by_id(class_id).name if labels is not None and class_id is not None else None,
                prob=float(mean_prob[index]) if mean_prob is not None else None,
                class_prob=float(class_probs[index]) if class_probs is not None else None,
                touches_border=bool(
                    box[0].start == 0 or box[1].start == 0 or box[0].stop == height or box[1].stop == width
                ),
            )
        )
    return cells


def drop_instances_without_cells(instances: np.ndarray, cells: Sequence[Cell]) -> np.ndarray:
    """Erase instances that produced no cell, so the instance map and the cell list describe the same objects.

    Ids are kept as they are: ``cell.label`` still points into the returned map.
    """
    keep = np.zeros(int(instances.max()) + 1, dtype=bool)
    keep[np.fromiter((cell.label for cell in cells), dtype=np.intp)] = True
    present = np.bincount(instances.ravel(), minlength=len(keep)) > 0
    present[0] = False
    if not (present & ~keep).any():
        return instances
    return np.where(keep[instances], instances, 0)


def tile_statistics(
    cells: Sequence[Cell], instances: np.ndarray, prob: np.ndarray | None = None
) -> dict[str, float | None]:
    """Per-tile numbers for the selector: cell counts and, with a probability map, uncertainty.

    ``mean_entropy`` is the mean binary entropy (in bits, 0..1) of the cell probability over
    the tile; ``uncertain_fraction`` the share of pixels with probability in [0.3, 0.7].
    """
    stats: dict[str, float | None] = {
        "n_cells": float(len(cells)),
        "foreground_fraction": float((instances > 0).mean()),
        "n_border_cells": float(sum(cell.touches_border for cell in cells)),
    }
    for cell in cells:
        if cell.class_name is not None:
            key = f"n_{cell.class_name}"
            stats[key] = stats.get(key, 0.0) + 1.0
    if prob is not None:
        p = np.clip(prob.astype(np.float64), 1e-7, 1 - 1e-7)
        entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        stats["mean_entropy"] = float(entropy.mean())
        stats["uncertain_fraction"] = float(((prob > 0.3) & (prob < 0.7)).mean())
        confidences = [cell.prob for cell in cells if cell.prob is not None]
        stats["mean_cell_prob"] = float(np.mean(confidences)) if confidences else None
    return stats


def _label(mask: np.ndarray) -> np.ndarray:
    """4-connected components, numbered from 1."""
    return cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)[1]


def _remove_small(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_pixels
    keep[0] = False
    return keep[labels]


def _minmax(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    low, high = float(array.min()), float(array.max())
    return (array - low) / (high - low) if high > low else np.zeros_like(array)


def _odd(value: float) -> int:
    """Nearest odd integer >= 3."""
    return max(3, int(round(value)) | 1)
