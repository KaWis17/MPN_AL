"""Label-free nucleus candidates from the haematoxylin stain.

This is **not** a base-model segmentation. It is the classical fallback for the time before
any head has been trained: colour deconvolution isolates haematoxylin, a threshold finds
nuclear pixels and a distance-transform watershed splits touching nuclei. It finds nuclei,
not whole cells, and it is not meant to be accurate on megakaryocytes.

Its use in this project is to exercise the whole chain (tiles -> instances -> GeoJSON ->
per-cell embeddings from the base model) from day one, and to serve as the baseline a
trained head has to beat.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from .postprocess import _label, _odd, _remove_small, flood_labels, remove_small_instances

# Ruifrok & Johnston stain vectors (rows: haematoxylin, eosin, residual), as optical densities.
_RGB_FROM_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], dtype=np.float64)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED).astype(np.float32)


@dataclass(frozen=True)
class StainBaselineConfig:
    smooth_um: float = 0.4  # Gaussian sigma applied to the haematoxylin channel
    min_diameter_um: float = 3.0  # smaller blobs are debris
    peak_distance_um: float = 4.0  # two nucleus centres are at least this far apart
    tissue_od: float = 0.12  # mean optical density below which a pixel is glass or fat
    reference_mpp: float = 0.25


def haematoxylin(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Haematoxylin concentration ``[H, W]`` and mean optical density ``[H, W]`` of an RGB uint8 image."""
    density = -np.log((image.astype(np.float32) + 1.0) / 256.0)
    concentration = density @ _HED_FROM_RGB[:, 0]
    return np.clip(concentration, 0.0, None), density.mean(axis=2)


def segment_nuclei_stain(
    image: np.ndarray, *, mpp: float | None = None, config: StainBaselineConfig = StainBaselineConfig()
) -> np.ndarray:
    """Instance map ``[H, W]`` (int32, 0 = background) of nucleus candidates in an H&E tile."""
    mpp = mpp or config.reference_mpp
    empty = np.zeros(image.shape[:2], dtype=np.int32)
    stain, density = haematoxylin(image)
    tissue = density > config.tissue_od
    if tissue.sum() < 50:
        return empty
    stain = cv2.GaussianBlur(stain, (0, 0), max(0.5, config.smooth_um / mpp))

    # Otsu between "nucleus" and "everything else in the tissue".
    top = float(np.percentile(stain[tissue], 99.5))
    if top <= 0:
        return empty
    scaled = np.clip(stain / top * 255.0, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(scaled[tissue].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (scaled > threshold) & tissue

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(1.0 / mpp),) * 2)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    mask = ndimage.binary_fill_holes(mask)
    min_pixels = max(4, int(round(np.pi * (config.min_diameter_um / 2) ** 2 / (mpp * mpp))))
    mask = _remove_small(mask, min_pixels)
    if not mask.any():
        return empty

    # One marker per local maximum of the distance to the background, then flood downhill.
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    peak_pixels = config.peak_distance_um / mpp
    distance = cv2.GaussianBlur(distance, (0, 0), max(1.0, 0.2 * peak_pixels))
    window = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(peak_pixels),) * 2)
    peaks = (distance >= cv2.dilate(distance, window) - 1e-6) & mask
    markers = cv2.connectedComponents(peaks.astype(np.uint8), connectivity=8)[1]
    elevation = 1.0 - distance / max(float(distance.max()), 1e-6)
    instances = flood_labels(markers, elevation, mask, np.linspace(0.0, 1.0, 24))

    orphans = _label(mask & (instances == 0))
    instances = np.where(orphans > 0, orphans + instances.max(), instances)
    return remove_small_instances(instances, min_pixels)
