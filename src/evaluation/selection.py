"""Choosing tiles for annotation from clustered embeddings, the baselines to beat, and how a selection is scored.

Geometry
--------
Tiles are squares of side ``size`` at level-0 positions ``(x, y)``. With a stride of half a tile, neighbours share
half their pixels, so two tiles *overlap* when their boxes share any area. A selection made with
``constrained=True`` never contains two overlapping tiles.

Selections
----------
``representatives``    one tile per cluster, the nearest to the centroid (cosine); clusters served largest first;
``random_selection``   uniformly random tiles;
``spatial_selection``  farthest-point sampling of tile positions: spread evenly over the slide, no embeddings;
(the reference upper bound, KMeans on the descriptors, is ``representatives`` applied to the descriptor matrix.)

Scores
------
All scores use model-independent descriptors (``tile_descriptors``) or the tiles' positions, never an embedding,
so a model is not judged in its own space.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# ---- tiles and geometry -----------------------------------------------------------------------


def tile_manifest(table: pd.DataFrame) -> pd.DataFrame:
    """``tiles.parquet`` rows plus ``lattice`` (member of the non-overlapping grid) and ``region`` (connected tissue piece).

    Rows are sorted by ``tile_id``, the order every array of the experiment follows.
    """
    out = table.sort_values("tile_id").reset_index(drop=True).copy()
    size = int(out["width"].iloc[0])
    if not ((out["width"] == size) & (out["height"] == size)).all():
        raise ValueError("all tiles must be squares of one size")
    out["lattice"] = ((out["x"] - out["x"].min()) % size == 0) & ((out["y"] - out["y"].min()) % size == 0)
    out["region"] = connected_regions(out[["x", "y"]].to_numpy(), size)
    return out


def overlaps(positions: np.ndarray, index: int, others: Sequence[int], size: int) -> np.ndarray:
    """For each tile in ``others``: does it share any area with tile ``index``?"""
    delta = np.abs(positions[np.asarray(others, dtype=int)] - positions[index])
    return (delta[:, 0] < size) & (delta[:, 1] < size)


def overlap_matrix(positions: np.ndarray, size: int) -> np.ndarray:
    """Boolean ``[n, n]``: True where two tiles share area (the diagonal is True)."""
    dx = np.abs(positions[:, None, 0] - positions[None, :, 0])
    dy = np.abs(positions[:, None, 1] - positions[None, :, 1])
    return (dx < size) & (dy < size)


def connected_regions(positions: np.ndarray, size: int) -> np.ndarray:
    """Label tiles by the piece of tissue they belong to: tiles that overlap or touch are connected. 0 = the largest."""
    n = len(positions)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = np.argsort(positions[:, 0], kind="stable")
    xs = positions[order, 0]
    for a, i in enumerate(order):
        stop = np.searchsorted(xs, positions[i, 0] + size, side="right")
        near = order[a + 1 : stop]
        for j in near[np.abs(positions[near, 1] - positions[i, 1]) <= size]:
            parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(n)])
    values, counts = np.unique(roots, return_counts=True)
    rank = {root: position for position, root in enumerate(values[np.argsort(-counts, kind="stable")])}
    return np.array([rank[r] for r in roots])


# ---- selections -------------------------------------------------------------------------------


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def representatives(
    vectors: np.ndarray,
    labels: np.ndarray,
    centres: np.ndarray,
    *,
    conflicts: np.ndarray | None = None,
    metric: str = "cosine",
) -> tuple[np.ndarray, int]:
    """One tile per non-empty cluster: the member nearest to its centre. Returns ``(indices, fallbacks)``.

    ``conflicts`` (an ``[n, n]`` boolean overlap matrix) forbids choosing tiles that overlap one already chosen: clusters are
    served from the largest, each taking its nearest free member; a cluster whose members all conflict takes the nearest free
    tile of the whole slide instead (counted in ``fallbacks``). ``metric`` is ``cosine`` or ``euclidean``.
    """
    if metric == "cosine":
        similarity = _unit(vectors) @ _unit(centres).T
    elif metric == "euclidean":
        similarity = -(((vectors[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2))
    else:
        raise ValueError(f"unknown metric {metric!r}")
    clusters, sizes = np.unique(labels, return_counts=True)
    blocked = np.zeros(len(vectors), dtype=bool)
    chosen: list[int] = []
    fallbacks = 0
    for cluster in clusters[np.argsort(-sizes, kind="stable")]:
        members = np.flatnonzero((labels == cluster) & ~blocked)
        if len(members) == 0:
            members = np.flatnonzero(~blocked)
            if len(members) == 0:
                break  # no free tile left anywhere
            fallbacks += 1
        best = int(members[np.argmax(similarity[members, cluster])])
        chosen.append(best)
        blocked[best] = True
        if conflicts is not None:
            blocked |= conflicts[best]
    return np.array(chosen, dtype=int), fallbacks


def random_selection(count: int, k: int, rng: np.random.Generator, conflicts: np.ndarray | None = None) -> np.ndarray:
    """``k`` random tiles; with ``conflicts`` no two of them overlap (fewer only if the slide runs out)."""
    blocked = np.zeros(count, dtype=bool)
    chosen: list[int] = []
    for index in rng.permutation(count):
        if len(chosen) == k:
            break
        if not blocked[index]:
            chosen.append(int(index))
            blocked[index] = True
            if conflicts is not None:
                blocked |= conflicts[index]
    return np.array(chosen, dtype=int)


def spatial_selection(positions: np.ndarray, k: int, rng: np.random.Generator, conflicts: np.ndarray | None = None) -> np.ndarray:
    """Farthest-point sampling over tile positions from a random start: an even spread over the slide."""
    blocked = np.zeros(len(positions), dtype=bool)
    nearest = np.full(len(positions), np.inf)
    chosen: list[int] = []
    index = int(rng.integers(len(positions)))
    points = positions.astype(np.float64)
    while len(chosen) < k:
        chosen.append(index)
        blocked[index] = True
        if conflicts is not None:
            blocked |= conflicts[index]
        nearest = np.minimum(nearest, np.sqrt(((points - points[index]) ** 2).sum(axis=1)))
        if blocked.all():
            break
        index = int(np.argmax(np.where(blocked, -np.inf, nearest)))
    return np.array(chosen, dtype=int)


# ---- scores -----------------------------------------------------------------------------------


def coverage(descriptors: np.ndarray, selected: np.ndarray) -> tuple[float, float]:
    """Mean and 95th percentile, over all tiles, of the distance to the nearest selected tile (lower = better)."""
    chosen = descriptors[selected]
    squared = (descriptors**2).sum(axis=1)[:, None] + (chosen**2).sum(axis=1)[None, :] - 2.0 * descriptors @ chosen.T
    nearest = np.sqrt(np.clip(squared.min(axis=1), 0.0, None))
    return float(nearest.mean()), float(np.percentile(nearest, 95))


def rare_flags(table: pd.DataFrame, features: Sequence[str], share: float = 0.05) -> np.ndarray:
    """``[n, len(features)]``: is the tile in the top ``share`` of that descriptor (ties at zero do not count)?"""
    flags = []
    for name in features:
        values = table[name].to_numpy(dtype=float)
        threshold = np.quantile(values, 1.0 - share)
        flags.append((values >= threshold) & (values > 0))
    return np.stack(flags, axis=1)


def jensen_shannon(a: np.ndarray, b: np.ndarray) -> float:
    """Jensen-Shannon divergence (bits, 0 to 1) between two histograms."""
    p, q = a / a.sum(), b / b.sum()
    m = 0.5 * (p + q)

    def kl(x, y):
        keep = x > 0
        return float((x[keep] * np.log2(x[keep] / y[keep])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


class Scorer:
    """Scores selections of one slide. Build once; call :meth:`score` for every selection."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        descriptors: np.ndarray,
        flags: np.ndarray,
        groups: np.ndarray,
        *,
        mpp: float,
        cell: int = 4096,
    ):
        self.descriptors = descriptors
        self.flags = flags
        self.groups = groups
        self.group_counts = np.bincount(groups).astype(float)
        self.positions = manifest[["x", "y"]].to_numpy(dtype=np.float64)
        self.size = int(manifest["width"].iloc[0])
        self.region = manifest["region"].to_numpy()
        self.region_counts = np.bincount(self.region)
        cells = (self.positions // cell).astype(int)
        _, self.cell = np.unique(cells, axis=0, return_inverse=True)
        self.mm_per_px = mpp / 1000.0

    def score(self, selected: np.ndarray) -> dict[str, float]:
        selected = np.asarray(selected, dtype=int)
        mean_distance, p95 = coverage(self.descriptors, selected)
        chosen = self.descriptors[selected]
        between = np.sqrt(((chosen[:, None, :] - chosen[None, :, :]) ** 2).sum(axis=2))
        upper = between[np.triu_indices(len(selected), 1)]
        points = self.positions[selected]
        apart = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(apart, np.inf)
        shared = overlap_matrix(points, self.size)
        covered = np.unique(self.region[selected])
        found = self.flags[selected].any(axis=0)
        return {
            "n_selected": float(len(selected)),
            "coverage_mean": mean_distance,
            "coverage_p95": p95,
            "diversity_mean": float(upper.mean()) if len(upper) else np.nan,
            "diversity_min": float(upper.min()) if len(upper) else np.nan,
            "rare_categories": float(found.sum()),
            "rare_tiles": float(self.flags[selected].any(axis=1).sum()),
            "jsd_groups": jensen_shannon(np.bincount(self.groups[selected], minlength=len(self.group_counts)).astype(float), self.group_counts),
            "regions": float(len(covered)),
            "tissue_in_covered_regions": float(self.region_counts[covered].sum() / self.region_counts.sum()),
            "grid_cells": float(len(np.unique(self.cell[selected]))),
            "nearest_neighbour_mm": float(apart.min(axis=1).mean() * self.mm_per_px) if len(selected) > 1 else np.nan,
            "overlapping_pairs": float((shared.sum() - len(selected)) / 2),
        }


def agreement(a: Sequence[int], b: Sequence[int], positions: np.ndarray, size: int) -> dict[str, float]:
    """How many tiles two selections share: identical ones, and a one-to-one matching of identical or overlapping tiles."""
    a, b = [int(i) for i in a], [int(i) for i in b]
    identical = set(a) & set(b)
    free_b = [j for j in b if j not in identical]
    matched = len(identical)
    for i in a:
        if i in identical or not free_b:
            continue
        hit = np.flatnonzero(overlaps(positions, i, free_b, size))
        if len(hit):
            distances = ((positions[[free_b[h] for h in hit]] - positions[i]) ** 2).sum(axis=1)
            free_b.pop(int(hit[np.argmin(distances)]))
            matched += 1
    smallest = max(min(len(a), len(b)), 1)
    return {"identical": float(len(identical)), "identical_or_overlapping": float(matched),
            "identical_share": len(identical) / smallest, "matched_share": matched / smallest}


def neighbour_overlap(a: np.ndarray, b: np.ndarray, conflicts: np.ndarray, k: int = 10) -> float:
    """Mean Jaccard overlap of the ``k`` nearest cosine neighbours in two spaces, ignoring spatially overlapping tiles."""
    na, nb = cosine_neighbours(a, conflicts, k), cosine_neighbours(b, conflicts, k)
    scores = []
    for x, y in zip(na, nb):
        shared = len(np.intersect1d(x, y))
        scores.append(shared / (2 * k - shared))
    return float(np.mean(scores))


def cosine_neighbours(vectors: np.ndarray, conflicts: np.ndarray, k: int) -> np.ndarray:
    """Indices ``[n, k]`` of the ``k`` most cosine-similar tiles that do not overlap the tile (nor are the tile)."""
    unit = _unit(vectors)
    similarity = unit @ unit.T
    similarity[conflicts] = -np.inf
    np.fill_diagonal(similarity, -np.inf)
    if (np.isfinite(similarity).sum(axis=1) < k).any():
        raise ValueError(f"some tile has fewer than {k} non-overlapping neighbours")
    return np.argsort(-similarity, axis=1, kind="stable")[:, :k]


def neighbour_consistency(vectors: np.ndarray, descriptors: np.ndarray, conflicts: np.ndarray, k: int = 10) -> float:
    """Mean descriptor distance between a tile and its ``k`` nearest non-overlapping embedding neighbours (lower = the space
    puts similar-looking tiles together). Compare with :func:`chance_consistency`."""
    neighbours = cosine_neighbours(vectors, conflicts, k)
    difference = descriptors[:, None, :] - descriptors[neighbours]
    return float(np.sqrt((difference**2).sum(axis=2)).mean())


def chance_consistency(descriptors: np.ndarray, samples: int = 200000, seed: int = 0) -> float:
    """Mean descriptor distance of random pairs of tiles."""
    rng = np.random.default_rng(seed)
    i, j = rng.integers(len(descriptors), size=samples), rng.integers(len(descriptors), size=samples)
    keep = i != j
    return float(np.sqrt(((descriptors[i[keep]] - descriptors[j[keep]]) ** 2).sum(axis=1)).mean())
