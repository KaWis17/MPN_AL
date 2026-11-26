"""Figures for the clustering of embeddings (output of ``cluster_embeddings.py``).

``plot_curves``         metrics against ``k`` for every model, with the noise control dashed;
``plot_heatmap``        model x k heatmap of one metric;
``plot_matrix``         model x model matrix (ARI of the clusterings, CKA or kNN overlap of the spaces);
``project_2d`` / ``plot_scatter``   2-D map of a space (UMAP, else t-SNE) coloured by cluster, label and nuisance variables;
``plot_composition``    what each cluster is made of (label, image size, patient group);
``render_gallery``      tiles nearest to each cluster's centre and random members, to look at what a cluster is;
``clusters_to_geojson`` tile rectangles with their cluster as a QuPath classification, for tiles whose id carries the position.

matplotlib (and optionally umap-learn) come with ``pip install -e ".[viz]"``. Colours: Okabe-Ito for up to eight
groups, ``tab20`` beyond; the noise control is always grey and dashed, so it cannot be mistaken for a model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..image_backend.tiles import parse_tile_id

logger = logging.getLogger("mpn_al.cluster_figures")

OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
NOISE_COLOUR = "#888888"
CONTROL = "noise-control"


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise RuntimeError('matplotlib is required for the figures: pip install -e ".[viz]"') from exc
    return plt


def palette(count: int) -> list[str]:
    if count <= len(OKABE):
        return OKABE[:count]
    plt = _plt()
    return [plt.get_cmap("tab20")(i % 20) for i in range(count)]


def _style(model: str, index: int) -> dict:
    if model == CONTROL:
        return {"color": NOISE_COLOUR, "linestyle": "--", "marker": "x"}
    return {"color": OKABE[index % len(OKABE)], "linestyle": "-", "marker": "o"}


# ---- metrics ----------------------------------------------------------------------------------


def plot_curves(metrics: pd.DataFrame, variant: str, path: str | Path, *, extra: Sequence[str] = ()) -> Path:
    """One panel per metric, ``k`` on the x axis. ``extra`` adds ``nmi_<column>`` curves (nuisance variables)."""
    plt = _plt()
    table = metrics[metrics["variant"] == variant]
    panels = [("silhouette", "silhouette (cosine)", None), ("ari_seed_mean", "ARI between seeds", "ari_seed_sd")]
    if "ari_heldout_mean" in table.columns:
        panels.append(("ari_heldout_mean", "ARI of held-out patients", None))
    # Chance-corrected measures only: NMI of a random clustering with a many-valued variable (patient) grows with k.
    panels.append(("ami_nuisance", "AMI with patient (0 = chance, lower = less batch)", None) if "ami_nuisance" in table.columns else None)
    panels += [(f"ami_{column}", f"AMI with {column} (0 = chance)", None) for column in extra if f"ami_{column}" in table.columns]
    panels += [("ari_label", "ARI with labels", None)] if "ari_label" in table.columns else []
    panels = [p for p in panels if p is not None and p[0] in table.columns]
    columns = min(3, len(panels))
    rows = -(-len(panels) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 3.4 * rows), squeeze=False)
    models = list(dict.fromkeys(table["model"]))
    for ax, (column, title, spread) in zip(axes.ravel(), panels):
        for index, model in enumerate(models):
            part = table[table["model"] == model].sort_values("k")
            ax.plot(part["k"], part[column], label=model, **_style(model, index))
            if spread and spread in part:
                ax.fill_between(part["k"], part[column] - part[spread], part[column] + part[spread], color=_style(model, index)["color"], alpha=0.15)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("k")
        ax.set_xticks(sorted(table["k"].unique()))
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(f"variant: {variant}", fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def plot_heatmap(metrics: pd.DataFrame, variant: str, column: str, path: str | Path) -> Path:
    """Model x k grid of ``column`` with the values written in the cells."""
    plt = _plt()
    table = metrics[metrics["variant"] == variant].pivot(index="model", columns="k", values=column)
    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * table.shape[1], 0.9 + 0.55 * table.shape[0]))
    image = ax.imshow(table.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.iloc[i, j]
            ax.text(j, i, "" if pd.isna(value) else f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if value < np.nanmean(table.to_numpy(dtype=float)) else "black")
    ax.set_xlabel("k")
    ax.set_title(f"{column}, {variant}", fontsize=10)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    return _save(fig, path)


def pivot_pairs(table: pd.DataFrame, value: str, **where) -> pd.DataFrame:
    """Symmetric model x model matrix from rows ``model_a, model_b, value`` filtered by ``where`` (diagonal = 1)."""
    for column, wanted in where.items():
        table = table[table[column] == wanted]
    names = sorted(set(table["model_a"]) | set(table["model_b"]))
    matrix = pd.DataFrame(np.nan, index=names, columns=names)
    for row in table.itertuples():
        matrix.loc[row.model_a, row.model_b] = matrix.loc[row.model_b, row.model_a] = getattr(row, value)
    np.fill_diagonal(matrix.values, 1.0)
    return matrix


def plot_matrix(matrix: pd.DataFrame, title: str, path: str | Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(1.6 + 0.9 * len(matrix), 1.3 + 0.8 * len(matrix)))
    image = ax.imshow(matrix.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(matrix)), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix)), matrix.index)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            value = matrix.iloc[i, j]
            ax.text(j, i, "" if pd.isna(value) else f"{value:.2f}", ha="center", va="center", fontsize=9, color="white" if value < 0.6 else "black")
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    return _save(fig, path)


# ---- maps of a space --------------------------------------------------------------------------


def project_2d(vectors: np.ndarray, *, seed: int = 0, method: str = "auto") -> tuple[np.ndarray, str]:
    """2-D coordinates of ``vectors`` (cosine geometry) and the method used: UMAP when it works, else t-SNE (``method="tsne"`` forces it)."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if method in ("auto", "umap"):
        try:
            import umap

            neighbours = int(min(15, max(2, len(vectors) - 1)))
            return umap.UMAP(n_neighbors=neighbours, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(vectors), "umap"
        except Exception as exc:  # noqa: BLE001 - not installed, or built against another scikit-learn: fall back unless UMAP was asked for
            if method == "umap":
                raise
            logger.warning("UMAP is not usable (%s: %s); using t-SNE", type(exc).__name__, str(exc)[:80])
    from sklearn.manifold import TSNE

    perplexity = float(min(30.0, max(2.0, (len(vectors) - 1) / 3.0)))
    return TSNE(n_components=2, perplexity=perplexity, init="pca", metric="cosine", random_state=seed).fit_transform(vectors), "t-SNE"


def plot_scatter(coords: np.ndarray, panels: Mapping[str, Sequence], path: str | Path, *, method: str, title: str = "") -> Path:
    """One scatter per entry of ``panels`` (name -> categorical value per point). Legends only for at most 12 categories."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.4), squeeze=False)
    for ax, (name, values) in zip(axes[0], panels.items()):
        values = np.asarray(values).astype(str)
        categories = sorted(set(values), key=lambda v: (len(v), v))
        colours = dict(zip(categories, palette(len(categories))))
        for category in categories:
            mask = values == category
            ax.scatter(coords[mask, 0], coords[mask, 1], s=14, color=colours[category], label=category, alpha=0.85, linewidths=0)
        ax.set_title(f"{name} ({len(categories)})", fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
        if len(categories) <= 12:
            ax.legend(fontsize=7, markerscale=1.4, loc="best", frameon=False)
    fig.suptitle(f"{title}  [{method}]", fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def plot_composition(clusters: Sequence, categories: Mapping[str, Sequence], path: str | Path, *, title: str = "") -> Path:
    """Stacked bars: for every cluster the share of each category, one panel per categorical variable."""
    plt = _plt()
    clusters = np.asarray(clusters)
    order = sorted(set(clusters))
    fig, axes = plt.subplots(1, len(categories), figsize=(5.2 * len(categories), 3.8), squeeze=False)
    for ax, (name, values) in zip(axes[0], categories.items()):
        values = np.asarray(values).astype(str)
        kinds = sorted(set(values), key=lambda v: (len(v), v))
        counts = np.array([[np.sum((clusters == c) & (values == v)) for v in kinds] for c in order], dtype=float)
        shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        bottom = np.zeros(len(order))
        for index, (kind, colour) in enumerate(zip(kinds, palette(len(kinds)))):
            ax.bar(range(len(order)), shares[:, index], bottom=bottom, color=colour, label=kind, width=0.8)
            bottom += shares[:, index]
        ax.set_title(f"cluster composition by {name}", fontsize=10)
        ax.set_xticks(range(len(order)), [str(c) for c in order])
        ax.set_xlabel("cluster")
        ax.set_ylim(0, 1)
        if len(kinds) <= 10:
            ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    _plt().close(fig)
    return path


# ---- what a cluster looks like ----------------------------------------------------------------


def cluster_members(vectors: np.ndarray, clusters: np.ndarray, *, near: int = 6, random: int = 6, seed: int = 0) -> dict[int, dict[str, list[int]]]:
    """Per cluster: row indices of the ``near`` samples closest to its centre (cosine), of ``random`` other members, and its ``size``."""
    rng = np.random.default_rng(seed)
    out = {}
    for cluster in sorted(set(clusters.tolist())):
        members = np.flatnonzero(clusters == cluster)
        centre = vectors[members].mean(axis=0)
        centre = centre / max(np.linalg.norm(centre), 1e-12)
        unit = vectors[members] / np.maximum(np.linalg.norm(vectors[members], axis=1, keepdims=True), 1e-12)
        nearest = members[np.argsort(-(unit @ centre))[:near]]
        rest = np.setdiff1d(members, nearest)
        sampled = rng.choice(rest, size=min(random, len(rest)), replace=False) if len(rest) else rest
        out[int(cluster)] = {"nearest": nearest.tolist(), "random": sorted(sampled.tolist()), "size": int(len(members))}
    return out


def render_gallery(
    members: Mapping[int, Mapping[str, Sequence[int] | int]], image_of, *, thumb: int = 128, title: str = ""
) -> np.ndarray:
    """RGB montage: a row per cluster, its central tiles then random ones. ``image_of(row_index)`` returns an RGB uint8 image."""
    import cv2

    def fit(image: np.ndarray) -> np.ndarray:
        scale = thumb / max(image.shape[:2])
        small = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        canvas = np.full((thumb, thumb, 3), 255, np.uint8)
        canvas[: small.shape[0], : small.shape[1]] = small
        return canvas

    width = max((len(m["nearest"]) + len(m["random"]) for m in members.values()), default=0)
    label_width = 70
    rows = []
    for cluster, group in members.items():
        tiles = [fit(image_of(i)) for i in group["nearest"]] + [fit(image_of(i)) for i in group["random"]]
        tiles += [np.full((thumb, thumb, 3), 255, np.uint8)] * (width - len(tiles))
        strip = np.concatenate(tiles, axis=1)
        label = np.full((thumb, label_width, 3), 255, np.uint8)
        cv2.putText(label, f"cluster {cluster}", (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(label, f"n={group.get('size', len(group['nearest']) + len(group['random']))}", (4, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
        rows.append(np.concatenate([label, strip], axis=1))
    header = np.full((26, rows[0].shape[1], 3), 255, np.uint8)
    cv2.putText(header, f"{title}  |  left: nearest the cluster centre, right: random members", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([header, *rows], axis=0)


# ---- QuPath -----------------------------------------------------------------------------------


def clusters_to_geojson(table: pd.DataFrame, out_dir: str | Path, *, prefix: str = "cluster") -> list[Path]:
    """One GeoJSON per slide: each tile as a rectangle in level-0 coordinates classified ``<prefix>_<cluster>``.

    ``table`` has ``tile_id`` and ``cluster``; the position comes from ``x, y`` and ``width`` (or ``size``) columns when present,
    else from an id like ``[tl_]slide_x123_y456_s1024``. Tiles without a position are skipped (the caller can count them).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_slide: dict[str, list[dict]] = {}
    for row in table.itertuples():
        parsed = parse_tile_id(row.tile_id)
        if {"x", "y"} <= set(table.columns) and not pd.isna(getattr(row, "x")):
            x, y = float(row.x), float(row.y)
            width = float(getattr(row, "width", None) or getattr(row, "size", None) or (parsed or {}).get("size") or 0)
            slide = str(getattr(row, "slide_id", None) or (parsed or {}).get("slide_id") or "slide")
        elif parsed is not None:
            x, y, width, slide = float(parsed["x"]), float(parsed["y"]), float(parsed["size"]), parsed["slide_id"]
        else:
            continue
        if width <= 0:
            continue
        ring = [[x, y], [x + width, y], [x + width, y + width], [x, y + width], [x, y]]
        per_slide.setdefault(slide, []).append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "objectType": "annotation",
                    "name": row.tile_id,
                    "classification": {"name": f"{prefix}_{int(row.cluster)}"},
                },
            }
        )
    written = []
    for slide, features in per_slide.items():
        path = out_dir / f"{slide}.geojson"
        path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
        written.append(path)
    return written
