"""Figures of the tile-selection experiment: curves over K, 2-D maps, boards of selected tiles and their place on the slide."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd

from .cluster_figures import NOISE_COLOUR, OKABE, _plt, _save, palette

BASELINE_STYLE = {
    "noise-control": {"color": NOISE_COLOUR, "linestyle": "--", "marker": "x"},
    "random": {"color": "#000000", "linestyle": ":", "marker": ""},
    "spatial": {"color": "#CC79A7", "linestyle": "-.", "marker": ""},
    "descriptor-kmeans": {"color": "#56B4E9", "linestyle": "--", "marker": ""},
}


def _style(name: str, index: int) -> dict:
    return BASELINE_STYLE.get(name, {"color": OKABE[index % 4], "linestyle": "-", "marker": "o"})


def plot_metric_curves(table: pd.DataFrame, panels: Sequence[tuple[str, str]], path: str | Path, *, title: str = "") -> Path:
    """``table`` has ``method, k`` and, for every panel column ``c``, ``c`` (mean) and ``c_sd``. One panel per metric."""
    plt = _plt()
    columns = min(3, len(panels))
    rows = -(-len(panels) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 3.6 * rows), squeeze=False)
    methods = list(dict.fromkeys(table["method"]))
    for ax, (column, label) in zip(axes.ravel(), panels):
        for index, method in enumerate(methods):
            part = table[(table["method"] == method)].dropna(subset=[column]).sort_values("k")
            if part.empty:
                continue
            style = _style(method, index)
            ax.plot(part["k"], part[column], label=method, **style)
            if f"{column}_sd" in part:
                ax.fill_between(part["k"], part[column] - part[f"{column}_sd"], part[column] + part[f"{column}_sd"], color=style["color"], alpha=0.12)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("K")
        ax.set_xticks(sorted(table["k"].unique()))
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    handles, labels = max((ax.get_legend_handles_labels() for ax in axes.ravel()), key=lambda pair: len(pair[0]))
    fig.legend(handles, labels, loc="lower center", ncol=min(7, len(labels)), fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return _save(fig, path)


def plot_space_map(coords: np.ndarray, clusters: np.ndarray, values: Mapping[str, np.ndarray], selected: Sequence[int], path: str | Path, *, title: str = "") -> Path:
    """The 2-D map of one model's space: coloured by cluster, then by continuous descriptors; selected tiles ringed."""
    plt = _plt()
    names = ["cluster", *values]
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.5), squeeze=False)
    colours = palette(int(clusters.max()) + 1)
    for ax, name in zip(axes[0], names):
        if name == "cluster":
            ax.scatter(coords[:, 0], coords[:, 1], s=5, c=[colours[c] for c in clusters], linewidths=0)
            ax.scatter(coords[selected, 0], coords[selected, 1], s=34, facecolors="none", edgecolors="black", linewidths=0.8)
            ax.set_title(f"clusters (circles = {len(selected)} selected)", fontsize=10)
        else:
            points = ax.scatter(coords[:, 0], coords[:, 1], s=5, c=values[name], cmap="viridis", linewidths=0)
            ax.set_title(name, fontsize=10)
            fig.colorbar(points, ax=ax, shrink=0.8)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def board(
    indices: Sequence[int],
    image_of: Callable[[int], np.ndarray],
    *,
    title: str,
    marks: Mapping[int, tuple[int, int, int]] | None = None,
    captions: Mapping[int, str] | None = None,
    columns: int = 10,
    thumb: int = 256,
) -> np.ndarray:
    """RGB plate of tiles in the given order; ``marks`` frames a tile in an RGB colour, ``captions`` writes under it."""
    marks, captions = marks or {}, captions or {}
    gap, line = 8, 18
    rows = -(-len(indices) // columns)
    header = 40
    canvas = np.full((header + rows * (thumb + line + gap) + gap, gap + columns * (thumb + gap), 3), 255, np.uint8)
    cv2.putText(canvas, title, (gap, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    for position, index in enumerate(indices):
        row, column = divmod(position, columns)
        y, x = header + row * (thumb + line + gap), gap + column * (thumb + gap)
        canvas[y : y + thumb, x : x + thumb] = cv2.resize(image_of(index), (thumb, thumb), interpolation=cv2.INTER_AREA)
        if index in marks:
            cv2.rectangle(canvas, (x + 2, y + 2), (x + thumb - 3, y + thumb - 3), marks[index], 5)
        text = f"{position + 1}. {captions.get(index, '')}"
        cv2.putText(canvas, text, (x, y + thumb + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
    return canvas


def slide_mosaic(positions: np.ndarray, size: int, image_of: Callable[[int], np.ndarray], *, scale: float = 1 / 32, margin: int = 512) -> tuple[np.ndarray, tuple[int, int]]:
    """A low-resolution picture of the slide assembled from the tiles: ``(canvas, level-0 origin)``."""
    origin = (int(positions[:, 0].min()) - margin, int(positions[:, 1].min()) - margin)
    width = int((positions[:, 0].max() + size + margin - origin[0]) * scale)
    height = int((positions[:, 1].max() + size + margin - origin[1]) * scale)
    canvas = np.full((height, width, 3), 255, np.uint8)
    side = max(1, int(round(size * scale)))
    for index, (x, y) in enumerate(positions):
        px, py = int((x - origin[0]) * scale), int((y - origin[1]) * scale)
        small = cv2.resize(image_of(index), (side, side), interpolation=cv2.INTER_AREA)
        canvas[py : py + side, px : px + side] = small[: height - py, : width - px]
    return canvas, origin


def slide_map(
    mosaic: np.ndarray,
    origin: tuple[int, int],
    positions: np.ndarray,
    size: int,
    selections: Mapping[str, Sequence[int]],
    *,
    scale: float = 1 / 32,
    colour: tuple[int, int, int] = (0, 90, 255),
) -> np.ndarray:
    """One copy of the slide per selection, stacked vertically, with the selected tiles outlined."""
    panels = []
    for name, indices in selections.items():
        panel = (mosaic.astype(np.float32) * 0.75 + 255 * 0.25).astype(np.uint8)
        for index in indices:
            x, y = positions[index]
            px, py = int((x - origin[0]) * scale), int((y - origin[1]) * scale)
            side = int(round(size * scale))
            cv2.rectangle(panel, (px - 4, py - 4), (px + side + 4, py + side + 4), colour, 6)
        cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), (120, 120, 120), 3)
        cv2.putText(panel, name, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4, cv2.LINE_AA)
        panels.append(panel)
    return np.concatenate(panels, axis=0)
