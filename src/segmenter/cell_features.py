"""Per-cell embeddings: the base model's patch tokens pooled under each cell.

This follows the "cell token" idea of CellViT++ (classifiers on token embeddings): once
cells are segmented, the frozen encoder's
dense features describe each of them, and a light classifier on those vectors (megakaryocyte
or not, then the megakaryocyte subtype) can be trained from a handful of labels without
touching the encoder. The same vectors are node features for cell graphs.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import sparse


def pool_cell_embeddings(instances: np.ndarray, tokens: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """Average the tokens each cell overlaps, weighted by overlap area.

    ``instances`` is an instance map ``[H, W]``, ``tokens`` a feature map ``[D, gh, gw]``
    covering the same field of view at any resolution, ``labels`` the instance ids to embed.
    Returns ``[len(labels), D]`` float32; a label absent from the map gets a zero vector.
    """
    labels = np.asarray(labels, dtype=np.int64)
    dim, grid_h, grid_w = tokens.shape
    if labels.size == 0:
        return np.zeros((0, dim), dtype=np.float32)
    height, width = instances.shape

    ys, xs = np.nonzero(instances)
    row_of_label = np.full(int(max(instances.max(), labels.max())) + 1, -1, dtype=np.int64)
    row_of_label[labels] = np.arange(labels.size)
    rows = row_of_label[instances[ys, xs]]
    keep = rows >= 0
    token_index = (ys[keep] * grid_h // height) * grid_w + (xs[keep] * grid_w // width)

    # overlap[i, j] = number of pixels of cell i that fall on token j
    overlap = sparse.coo_matrix(
        (np.ones(int(keep.sum()), dtype=np.float32), (rows[keep], token_index)),
        shape=(labels.size, grid_h * grid_w),
    ).tocsr()
    totals = np.asarray(overlap.sum(axis=1)).ravel()
    weights = sparse.diags(1.0 / np.maximum(totals, 1.0)) @ overlap
    flat = np.asarray(tokens, dtype=np.float32).reshape(dim, grid_h * grid_w).T
    return np.asarray(weights @ flat, dtype=np.float32)
