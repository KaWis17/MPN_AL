"""Comparing the embedding spaces of different models, which may have different dimensions.

All measures here need no common dimension:

``partition_agreement``   ARI/AMI between the clusterings two models produce for the same samples and the same ``k``;
``knn_overlap``           mean Jaccard overlap of each sample's ``k`` nearest neighbours (cosine) in the two spaces;
``linear_cka``            linear centred kernel alignment: how similar the two spaces' pairwise geometry is (1 = identical up
                          to rotation and isotropic scaling).

Row ``i`` of every array must be the same sample in every model.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors


def _check_same_samples(spaces: Mapping[str, np.ndarray]) -> None:
    sizes = {name: len(array) for name, array in spaces.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"models must be embedded on the same samples, got row counts {sizes}")


def partition_agreement(labels_by_model: Mapping[str, np.ndarray]) -> dict[str, pd.DataFrame]:
    """Symmetric ``ari`` and ``ami`` matrices between the models' clusterings (names of clusters are irrelevant)."""
    _check_same_samples(labels_by_model)
    names = list(labels_by_model)
    out = {kind: pd.DataFrame(np.nan, index=names, columns=names) for kind in ("ari", "ami")}
    for a in names:
        for b in names:
            out["ari"].loc[a, b] = adjusted_rand_score(labels_by_model[a], labels_by_model[b])
            out["ami"].loc[a, b] = adjusted_mutual_info_score(labels_by_model[a], labels_by_model[b])
    return out


def _neighbours(vectors: np.ndarray, k: int) -> np.ndarray:
    index = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute").fit(vectors)
    order = index.kneighbors(vectors, return_distance=False)
    # Drop the sample itself; with duplicated vectors it may not come first, so remove it by identity.
    result = np.empty((len(vectors), k), dtype=np.int64)
    for i, row in enumerate(order):
        others = row[row != i]
        result[i] = others[:k]
    return result


def knn_overlap(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    """Mean over samples of |N_a(i) & N_b(i)| / |N_a(i) | N_b(i)|, with N the ``k`` nearest neighbours by cosine distance."""
    if len(a) != len(b):
        raise ValueError(f"row counts differ: {len(a)} and {len(b)}")
    if k >= len(a):
        raise ValueError(f"k = {k} needs more than {len(a)} samples")
    na, nb = _neighbours(np.asarray(a, dtype=np.float64), k), _neighbours(np.asarray(b, dtype=np.float64), k)
    scores = []
    for x, y in zip(na, nb):
        shared = len(np.intersect1d(x, y))
        scores.append(shared / (2 * k - shared))
    return float(np.mean(scores))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA of two ``[n, d]`` representations of the same ``n`` samples (columns are centred first)."""
    if len(x) != len(y):
        raise ValueError(f"row counts differ: {len(x)} and {len(y)}")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x, y = x - x.mean(axis=0, keepdims=True), y - y.mean(axis=0, keepdims=True)
    if x.shape[1] <= len(x) and y.shape[1] <= len(y):  # feature form: cost grows with d, not n
        numerator = np.linalg.norm(y.T @ x, "fro") ** 2
        denominator = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    else:  # Gram form
        kx, ky = x @ x.T, y @ y.T
        numerator = float((kx * ky).sum())
        denominator = np.linalg.norm(kx, "fro") * np.linalg.norm(ky, "fro")
    return float(numerator / denominator) if denominator > 0 else float("nan")


def pairwise_matrix(spaces: Mapping[str, np.ndarray], measure) -> pd.DataFrame:
    """``measure(a, b)`` for every pair of models as a symmetric DataFrame (computed once per pair)."""
    _check_same_samples(spaces)
    names = list(spaces)
    table = pd.DataFrame(np.nan, index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i:]:
            value = measure(spaces[a], spaces[b])
            table.loc[a, b] = table.loc[b, a] = value
    return table
