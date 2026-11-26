"""Model-independent descriptors of an H&E bone-marrow tile, used to judge how diverse a selection of tiles is.

None of the base models is involved, so scoring a model's selection with these numbers is not circular. They are crude
proxies computed from pixels, not diagnoses:

``white``          share of unstained pixels (fat cells, glass, lumina);
``fat``            the part of it lying in rounded white blobs of adipocyte size;
``erythrocyte``    saturated orange-red pixels (red cells, haemorrhage);
``pigment``        dark brown pixels (haemosiderin and similar);
``compact_eosin``  large smooth eosin-rich areas (bone trabeculae, dense matrix);
``nuclear_area``, ``nuclear_density``, ``nuclear_size``, ``large_nuclei``
                   from the haematoxylin nucleus candidates: share of area, count per mm2, median area and the share of
                   area in nuclei above 80 um2;
``h_median``, ``e_median``, ``h_high``  haematoxylin/eosin concentration over stained pixels (staining intensity);
``texture``        mean gradient magnitude of the grey image.

Every tile is analysed at half resolution. ``white`` is the background colour of the whole slide (estimate it once with
:func:`slide_background` and pass it to every tile, so that tiles stay comparable).
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import pandas as pd

from ..segmenter.stain_baseline import _HED_FROM_RGB, segment_nuclei_stain

FEATURES = (
    "white", "fat", "erythrocyte", "pigment", "compact_eosin", "nuclear_area", "nuclear_density", "nuclear_size",
    "large_nuclei", "h_median", "e_median", "h_high", "texture",
)
RARE_FEATURES = ("compact_eosin", "erythrocyte", "pigment", "nuclear_density", "large_nuclei", "white")
SCALE = 0.5  # tiles are analysed at this fraction of their resolution


def slide_background(images: Sequence[np.ndarray]) -> np.ndarray:
    """RGB colour of unstained glass: the median over sample tiles of each tile's 99th percentile per channel."""
    highs = [np.percentile(image[::8, ::8].reshape(-1, 3), 99.0, axis=0) for image in images]
    return np.clip(np.median(np.stack(highs), axis=0), 180.0, 255.0).astype(np.float32)


def describe_tile(image: np.ndarray, *, mpp: float, background: np.ndarray | None = None) -> dict[str, float]:
    """The :data:`FEATURES` of one RGB uint8 tile of resolution ``mpp`` um/px."""
    small = cv2.resize(image, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    um = mpp / SCALE  # um per analysed pixel
    pixel_area = um * um
    white_rgb = np.asarray((255.0, 255.0, 255.0) if background is None else background, dtype=np.float32)
    density = np.clip(-np.log((small.astype(np.float32) + 1.0) / (white_rgb + 1.0)), 0.0, None)
    mean_density = density.mean(axis=2)
    stains = density @ _HED_FROM_RGB
    h = np.clip(stains[..., 0], 0.0, None)
    e = np.clip(stains[..., 1], 0.0, None)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hue, saturation, value = (hsv[..., i].astype(np.int16) for i in range(3))
    total = mean_density.size

    white = mean_density < 0.08
    opened = cv2.morphologyEx(white.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=4)
    fat_pixels = 0
    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA]
        if not 200.0 <= area * pixel_area <= 20000.0:
            continue
        x, y, w, box_h = stats[index, :4]
        blob = (labels[y : y + box_h, x : x + w] == index).astype(np.uint8)
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.contourArea(cv2.convexHull(contours[0]))
        if hull > 0 and cv2.contourArea(contours[0]) / hull > 0.8:  # rounded, as fat cells are
            fat_pixels += int(area)

    erythrocyte = ((hue <= 8) | (hue >= 175)) & (saturation > 110) & (value > 90) & (value < 235) & ~white
    pigment = (hue >= 6) & (hue <= 30) & (saturation > 70) & (value <= 150)

    grey = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gradient = cv2.magnitude(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1))
    side = max(3, int(round(12.0 / um)) | 1)  # 12 um neighbourhood
    smooth = cv2.boxFilter(gradient, -1, (side, side)) < 60.0
    eosinophilic = smooth & (e > 0.25) & (h < 0.35) & ~white
    eosinophilic = cv2.morphologyEx(eosinophilic.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side)))
    count_e, _, stats_e, _ = cv2.connectedComponentsWithStats(eosinophilic, connectivity=4)
    compact = sum(int(a) for a in stats_e[1:count_e, cv2.CC_STAT_AREA] if a * pixel_area >= 1000.0)

    nuclei = segment_nuclei_stain(small, mpp=um)
    areas = np.bincount(nuclei.ravel())[1:] * pixel_area
    areas = areas[(areas > 0) & (areas < 1500.0)]  # anything larger is a merged region, not a nucleus
    tile_mm2 = total * pixel_area / 1e6

    stained = mean_density >= 0.12
    return {
        "white": float(white.sum() / total),
        "fat": float(fat_pixels / total),
        "erythrocyte": float(erythrocyte.sum() / total),
        "pigment": float(pigment.sum() / total),
        "compact_eosin": float(compact / total),
        "nuclear_area": float(areas.sum() / (total * pixel_area)),
        "nuclear_density": float(len(areas) / tile_mm2),
        "nuclear_size": float(np.median(areas)) if len(areas) else 0.0,
        "large_nuclei": float(areas[areas > 80.0].sum() / (total * pixel_area)),
        "h_median": float(np.median(h[stained])) if stained.any() else 0.0,
        "e_median": float(np.median(e[stained])) if stained.any() else 0.0,
        "h_high": float(np.percentile(h[stained], 95)) if stained.any() else 0.0,
        "texture": float(gradient.mean()),
    }


def standardize(table: pd.DataFrame, features: Sequence[str] = FEATURES, clip: float = 5.0) -> np.ndarray:
    """Z-scored feature matrix (constant features become zero), clipped so that single outliers do not dominate distances."""
    values = table[list(features)].to_numpy(dtype=np.float64)
    spread = values.std(axis=0)
    scaled = (values - values.mean(axis=0)) / np.where(spread > 0, spread, 1.0)
    return np.clip(scaled, -clip, clip)
