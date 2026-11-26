"""See what each trained head does with a tile, side by side, and what neighbouring context changes.

Two questions are answered with pictures and numbers:

1. *Same tile, different base models.* :func:`run_model` turns a tile into a :class:`ModelView`
   (cell probability, HV maps, instances, encoder-token PCA, all in the tile's own pixel frame);
   :func:`render_comparison` lays the views of several models out in one PNG, one row per model,
   and :func:`agreement_table` gives the instance-level agreement (PQ) between each pair of models.
2. *Does the tile need its neighbours?* :func:`context_effect` segments the same central
   region twice, alone and cut out of a larger image that includes a margin of surroundings, and
   compares both results as a function of the distance to the region's border.
   :func:`render_context` draws it.

Only OpenCV and NumPy draw; nothing here needs matplotlib, so it runs on a compute node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import cv2
import numpy as np

from ..segmenter.decoder import DECODER_STRIDE
from ..segmenter.model import CellSegmenter, predict_maps
from ..segmenter.postprocess import PostprocessConfig, hv_to_instances
from .instance_metrics import instance_counts, aggregate

DISTANCE_BINS = (0, 16, 32, 64, 128, 256, 10**9)  # px from the region border, in input pixels
_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class ModelView:
    """One model's reading of one image, everything at the image's own pixel grid (``[H, W]``)."""

    name: str
    prob: np.ndarray  # [H, W] float32
    hv: np.ndarray  # [2, H, W] float32
    instances: np.ndarray  # [H, W] int32
    token_rgb: np.ndarray  # [H, W, 3] uint8, PCA of the encoder's patch tokens
    geometry: dict = field(default_factory=dict)

    @property
    def cell_count(self) -> int:
        return int(len(np.unique(self.instances)) - (1 if (self.instances == 0).any() else 0))


# ---- running models ---------------------------------------------------------------------------


def model_geometry(segmenter: CellSegmenter, height: int, width: int, mpp: float | None, max_size: int = 1024) -> dict:
    """What the encoder actually receives for an ``height x width`` image (see ``segmenter.model``)."""
    work_h, work_w = segmenter.working_size(height, width, mpp)
    patch = segmenter.base.patch_size
    passes_h = 1 if work_h <= max_size else int(np.ceil((work_h - max_size) / (max_size - 128))) + 1
    passes_w = 1 if work_w <= max_size else int(np.ceil((work_w - max_size) / (max_size - 128))) + 1
    crop_h, crop_w = min(work_h, max_size), min(work_w, max_size)
    return {
        "model": segmenter.base.name,
        "input_px": f"{width}x{height}",
        "working_px": f"{work_w}x{work_h}",
        "seg_mpp": segmenter.seg_mpp,
        "encoder_input_px": f"{crop_w // DECODER_STRIDE * patch}x{crop_h // DECODER_STRIDE * patch}",
        "patch": patch,
        "token_grid": f"{crop_w // DECODER_STRIDE}x{crop_h // DECODER_STRIDE}",
        "passes": passes_h * passes_w,
        "tile_covers_um": None if mpp is None else f"{width * mpp:.0f}x{height * mpp:.0f}",
    }


def token_pca_grid(tokens: np.ndarray) -> np.ndarray:
    """First three principal components of ``tokens`` ``[D, gh, gw]`` as a ``[gh, gw, 3]`` uint8 image.

    Fitted on this image only (eigenvectors of the D x D covariance, so cost does not grow with the number of
    tokens). Colours are comparable inside one panel, not between models or images; the order of components is
    fixed by their variance and their sign by making the largest loading of each positive.
    """
    dim, grid_h, grid_w = tokens.shape
    flat = tokens.reshape(dim, -1).T.astype(np.float64)
    flat -= flat.mean(axis=0, keepdims=True)
    values, vectors = np.linalg.eigh(flat.T @ flat)
    top = vectors[:, ::-1][:, : min(3, dim)]
    top = top * np.sign(top[np.abs(top).argmax(axis=0), np.arange(top.shape[1])])
    comps = flat @ top
    if comps.shape[1] < 3:
        comps = np.pad(comps, ((0, 0), (0, 3 - comps.shape[1])))
    low, high = np.percentile(comps, 2, axis=0), np.percentile(comps, 98, axis=0)
    scaled = np.clip((comps - low) / np.maximum(high - low, 1e-9), 0, 1)
    return (scaled.reshape(grid_h, grid_w, 3) * 255).astype(np.uint8)


def token_pca_rgb(tokens: np.ndarray, height: int, width: int) -> np.ndarray:
    """:func:`token_pca_grid` resampled to an RGB image of ``height x width``."""
    return cv2.resize(token_pca_grid(tokens), (width, height), interpolation=cv2.INTER_CUBIC)


def run_model(
    name: str,
    segmenter: CellSegmenter,
    image: np.ndarray,
    *,
    mpp: float | None = None,
    postprocess: PostprocessConfig = PostprocessConfig(),
    amp: bool = True,
    max_size: int = 1024,
) -> ModelView:
    """Segment ``image`` and express every output on the image's own pixel grid."""
    height, width = image.shape[:2]
    maps = predict_maps(segmenter, image, mpp=mpp, amp=amp, max_size=max_size)
    work_h, work_w = maps.prob.shape
    work_mpp = mpp * maps.scale[0] if mpp is not None else None
    instances_work = hv_to_instances(maps.prob, maps.hv, mpp=work_mpp, config=postprocess)

    def to_image(array: np.ndarray, nearest: bool = False) -> np.ndarray:
        if array.shape[-2:] == (height, width):
            return array
        mode = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        if array.ndim == 2:
            return cv2.resize(array, (width, height), interpolation=mode)
        return np.stack([cv2.resize(channel, (width, height), interpolation=mode) for channel in array])

    return ModelView(
        name=name,
        prob=to_image(maps.prob).astype(np.float32),
        hv=to_image(maps.hv).astype(np.float32),
        instances=to_image(instances_work.astype(np.int32), nearest=True),
        token_rgb=token_pca_rgb(maps.tokens, height, width),
        geometry=model_geometry(segmenter, height, width, mpp, max_size),
    )


# ---- drawing ----------------------------------------------------------------------------------


def instance_boundaries(instances: np.ndarray) -> np.ndarray:
    """Boolean ``[H, W]`` map of pixels on the outline of an instance (or of the foreground edge)."""
    labels = instances.astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(labels, kernel) != cv2.erode(labels, kernel)


def overlay_instances(image: np.ndarray, instances: np.ndarray, color=(0, 255, 0), alpha: float = 1.0) -> np.ndarray:
    """``image`` with instance outlines drawn in ``color`` (RGB)."""
    out = image.copy()
    edge = instance_boundaries(instances)
    out[edge] = (alpha * np.array(color) + (1 - alpha) * out[edge]).astype(np.uint8)
    return out


def heatmap(values: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    """RGB viridis-like colouring of a scalar map (OpenCV's VIRIDIS, converted from BGR)."""
    scaled = np.clip((values - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    return cv2.cvtColor(cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)


def hv_rgb(hv: np.ndarray) -> np.ndarray:
    """Horizontal/vertical distance maps (``[-1, 1]``) as red/green; blue is their magnitude."""
    h, v = np.clip(hv[0], -1, 1), np.clip(hv[1], -1, 1)
    magnitude = np.clip(np.sqrt(h * h + v * v), 0, 1)
    return (np.stack([(h + 1) / 2, (v + 1) / 2, magnitude], axis=-1) * 255).astype(np.uint8)


def _label(panel: np.ndarray, text: str) -> np.ndarray:
    panel = panel.copy()
    (w, h), base = cv2.getTextSize(text, _FONT, 0.5, 1)
    cv2.rectangle(panel, (0, 0), (w + 8, h + base + 6), (0, 0, 0), -1)
    cv2.putText(panel, text, (4, h + 3), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def _fit(panel: np.ndarray, size: int) -> np.ndarray:
    """Scale so the longer side is ``size`` px, keeping the aspect ratio."""
    height, width = panel.shape[:2]
    scale = size / max(height, width)
    return cv2.resize(panel, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def _stack(items: Sequence[np.ndarray], axis: int) -> np.ndarray:
    """Concatenate images along ``axis`` (0 = rows, 1 = columns), padding the other side with white."""
    other = 1 - axis
    size = max(item.shape[other] for item in items)
    padded = []
    for item in items:
        pad = [(0, 0)] * 3
        pad[other] = (0, size - item.shape[other])
        padded.append(np.pad(item, pad, constant_values=255))
    return np.concatenate(padded, axis=axis)


def render_comparison(
    image: np.ndarray,
    views: Sequence[ModelView],
    *,
    reference: np.ndarray | None = None,
    reference_name: str = "reference",
    panel: int = 360,
) -> np.ndarray:
    """One row per model: outlines on the image | cell probability | HV maps | encoder-token PCA.

    The first row shows the plain image and, when ``reference`` (an instance map: annotation or pseudo-label)
    is given, its outlines. Returns an RGB image.
    """
    shown = _fit(image, panel)
    blank = np.full_like(shown, 255)
    first = [_label(shown, "input")]
    first.append(_label(_fit(overlay_instances(image, reference, (255, 255, 0)), panel), reference_name) if reference is not None else blank)
    first += [blank, blank]
    rows = [_stack(first, 1)]
    for view in views:
        cells = [
            _label(_fit(overlay_instances(image, view.instances, (0, 255, 0)), panel), f"{view.name}: {view.cell_count} cells"),
            _label(_fit(heatmap(view.prob), panel), "cell probability"),
            _label(_fit(hv_rgb(view.hv), panel), "HV maps"),
            _label(_fit(view.token_rgb, panel), f"tokens PCA {view.geometry.get('token_grid', '')}"),
        ]
        rows.append(_stack(cells, 1))
    return _stack(rows, 0)


# ---- zoomed side-by-side and scores against the reference -------------------------------------


def densest_box(instances: np.ndarray, size: int) -> tuple[slice, slice]:
    """The ``size`` x ``size`` window (clipped to the image) holding the most instance pixels, on a coarse grid.

    Deterministic, so every model and every rerun is shown on the same region.
    """
    height, width = instances.shape
    size = min(size, height, width)
    step = max(size // 4, 1)
    foreground = (instances > 0).astype(np.float32)
    integral = cv2.integral(foreground)
    best, best_at = -1.0, (0, 0)
    for top in list(range(0, height - size + 1, step)) + [height - size]:
        for left in list(range(0, width - size + 1, step)) + [width - size]:
            total = integral[top + size, left + size] - integral[top, left + size] - integral[top + size, left] + integral[top, left]
            if total > best:
                best, best_at = total, (top, left)
    return slice(best_at[0], best_at[0] + size), slice(best_at[1], best_at[1] + size)


def reference_scores(views: Sequence[ModelView], reference: np.ndarray) -> "pd.DataFrame":  # noqa: F821
    """PQ/DQ/SQ and cell counts of every model against ``reference`` (an annotation or pseudo-label) on one image."""
    import pandas as pd

    rows = []
    for view in views:
        counts = instance_counts(reference, view.instances)
        scores = aggregate([counts])
        rows.append({"model": view.name, "pq": scores["pq"], "dq": scores["dq"], "sq": scores["sq"],
                     "cells": view.cell_count, "reference_cells": int(len(np.unique(reference)) - (1 if (reference == 0).any() else 0))})
    return pd.DataFrame(rows)


def render_zoom(
    image: np.ndarray,
    views: Sequence[ModelView],
    box: tuple[slice, slice],
    *,
    reference: np.ndarray | None = None,
    scores: "pd.DataFrame | None" = None,  # noqa: F821
    panel: int = 480,
) -> np.ndarray:
    """One row of equal-sized crops of the same region: plain image | reference outlines | one panel per model.

    A crop is enlarged with nearest-neighbour interpolation, so outlines stay one source pixel wide and cells can be
    counted by eye. The model panels carry the model's PQ against ``reference`` when ``scores`` is given.
    """
    rows, cols = box

    def crop(array):
        return array[rows, cols]

    def enlarge(rgb):
        height, width = rgb.shape[:2]
        scale = panel / max(height, width)
        return cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_NEAREST)

    plain = crop(image)
    panels = [_label(enlarge(plain), f"input {plain.shape[1]}x{plain.shape[0]} px")]
    if reference is not None:
        panels.append(_label(enlarge(overlay_instances(plain, crop(reference), (255, 255, 0))), "reference"))
    by_model = {} if scores is None else scores.set_index("model")["pq"].to_dict()
    for view in views:
        text = view.name + (f"  PQ {by_model[view.name]:.2f}" if view.name in by_model else "")
        panels.append(_label(enlarge(overlay_instances(plain, crop(view.instances), (0, 255, 0))), text))
    return _stack(panels, 1)


# ---- agreement between models -----------------------------------------------------------------


def agreement_table(views: Sequence[ModelView]) -> "pd.DataFrame":  # noqa: F821
    """Pairwise PQ between the instance maps of every two models (rows predict, columns are the reference)."""
    import pandas as pd

    names = [view.name for view in views]
    table = pd.DataFrame(np.nan, index=names, columns=names)
    for a in views:
        for b in views:
            counts = aggregate([instance_counts(b.instances, a.instances)])
            table.loc[a.name, b.name] = counts["pq"]
    return table


# ---- the effect of neighbouring context -------------------------------------------------------


def distance_to_border(height: int, width: int) -> np.ndarray:
    """Distance in pixels of every pixel of an ``height x width`` region to its nearest edge."""
    rows = np.minimum(np.arange(height), np.arange(height)[::-1])
    cols = np.minimum(np.arange(width), np.arange(width)[::-1])
    return np.minimum(rows[:, None], cols[None, :])


def crop_box(height: int, width: int, margin: int) -> tuple[slice, slice]:
    """The central region left after removing ``margin`` px on every side."""
    if margin * 2 >= min(height, width):
        raise ValueError(f"margin {margin} px leaves nothing of a {width}x{height} px image")
    return slice(margin, height - margin), slice(margin, width - margin)


def context_effect(
    name: str,
    segmenter: CellSegmenter,
    image: np.ndarray,
    margin: int,
    *,
    mpp: float | None = None,
    band: int = 48,
    postprocess: PostprocessConfig = PostprocessConfig(),
    amp: bool = True,
) -> tuple[dict, ModelView, ModelView]:
    """Segment the central region of ``image`` alone and with ``margin`` px of surroundings, and compare.

    Returns ``(row, alone, with_context)``; both views are cropped to the central region. ``row`` holds:

    * ``pq_all``, ``pq_band``, ``pq_interior``: PQ of "alone" against "with context" over the whole region,
      only within ``band`` px of its border, and only deeper inside (the second is the *context effect*);
    * ``cells_alone`` / ``cells_context`` and their counts of cells that touch the region border;
    * ``prob_diff_<lo>_<hi>``: mean absolute difference of the cell probability at that distance (px) from
      the border, which shows how far the missing context reaches.
    """
    rows, cols = crop_box(*image.shape[:2], margin)
    core = np.ascontiguousarray(image[rows, cols])
    alone = run_model(name, segmenter, core, mpp=mpp, postprocess=postprocess, amp=amp)
    wide = run_model(name, segmenter, image, mpp=mpp, postprocess=postprocess, amp=amp)
    with_context = ModelView(
        name=name,
        prob=wide.prob[rows, cols],
        hv=wide.hv[:, rows, cols],
        instances=wide.instances[rows, cols],
        token_rgb=wide.token_rgb[rows, cols],
        geometry=wide.geometry,
    )

    distance = distance_to_border(*core.shape[:2])
    near = distance < band

    def pq(ignore: np.ndarray | None) -> float:
        # Cells of the context run that were cut by the region border are kept: that is what a tile has to live with.
        return aggregate([instance_counts(with_context.instances, alone.instances, ignore=ignore)])["pq"]

    def border_cells(instances: np.ndarray) -> int:
        edge = np.zeros(instances.shape, bool)
        edge[0], edge[-1], edge[:, 0], edge[:, -1] = True, True, True, True
        return int(len(np.unique(instances[edge & (instances > 0)])))

    row = {
        "model": name,
        "margin_px": margin,
        "band_px": band,
        "pq_all": pq(None),
        "pq_band": pq(~near),
        "pq_interior": pq(near),
        "cells_alone": alone.cell_count,
        "cells_context": with_context.cell_count,
        "border_cells_alone": border_cells(alone.instances),
        "border_cells_context": border_cells(with_context.instances),
    }
    difference = np.abs(alone.prob - with_context.prob)
    for low, high in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]):
        selected = (distance >= low) & (distance < high)
        key = f"prob_diff_{low}_{high if high < 10**9 else 'inf'}"
        row[key] = float(difference[selected].mean()) if selected.any() else float("nan")
    return row, alone, with_context


def render_context(
    image: np.ndarray,
    margin: int,
    results: Sequence[tuple[dict, ModelView, ModelView]],
    *,
    panel: int = 360,
) -> np.ndarray:
    """Per model: the full image with the central region framed | alone | with context | where they disagree."""
    rows_slice, cols_slice = crop_box(*image.shape[:2], margin)
    framed = image.copy()
    cv2.rectangle(framed, (cols_slice.start, rows_slice.start), (cols_slice.stop - 1, rows_slice.stop - 1), (255, 0, 0), max(2, image.shape[0] // 256))
    core = image[rows_slice, cols_slice]
    lines = []
    for row, alone, with_context in results:
        difference = np.abs(alone.prob - with_context.prob)
        cells = [
            _label(_fit(framed, panel), f"{row['model']}: region + {margin}px context"),
            _label(_fit(overlay_instances(core, alone.instances, (255, 128, 0)), panel), f"alone: {row['cells_alone']} cells"),
            _label(_fit(overlay_instances(core, with_context.instances, (0, 255, 0)), panel), f"with context: {row['cells_context']} cells"),
            _label(_fit(heatmap(difference, 0, 0.5), panel), f"|dP|, PQ band {row['pq_band']:.2f} / inside {row['pq_interior']:.2f}"),
        ]
        lines.append(_stack(cells, 1))
    return _stack(lines, 0)


def summarize_context(table: "pd.DataFrame") -> "pd.DataFrame":  # noqa: F821
    """Mean over images of the numeric columns, per model."""
    numeric = [c for c in table.columns if c not in ("model", "tile_id")]
    return table.groupby("model")[numeric].mean().reset_index()
