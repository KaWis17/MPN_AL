"""Instance-segmentation metrics: panoptic quality and its parts.

For one class (or class-agnostic) instance maps:

* a predicted and a true instance match when their IoU exceeds 0.5 (such matches are unique);
* ``DQ`` = TP / (TP + FP/2 + FN/2) is the detection F1, ``SQ`` the mean IoU of the matches,
  and ``PQ = DQ * SQ``.

Counts are summed over tiles before the ratios are taken (:func:`aggregate`), so a dataset
score is not dominated by tiles with few cells.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def pairwise_iou(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """IoU of every true instance (rows) with every predicted instance (columns); ids need not be contiguous."""
    true_ids, true_index = np.unique(np.concatenate([[0], true.ravel()]), return_inverse=True)
    pred_ids, pred_index = np.unique(np.concatenate([[0], pred.ravel()]), return_inverse=True)
    true_index, pred_index = true_index[1:], pred_index[1:]  # drop the sentinel that guarantees a background row
    joint = np.bincount(
        true_index * len(pred_ids) + pred_index, minlength=len(true_ids) * len(pred_ids)
    ).reshape(len(true_ids), len(pred_ids))
    intersection = joint[1:, 1:].astype(np.float64)
    union = joint.sum(axis=1)[1:, None] + joint.sum(axis=0)[None, 1:] - intersection
    return intersection / np.maximum(union, 1.0)


def instance_counts(
    true: np.ndarray, pred: np.ndarray, *, ignore: np.ndarray | None = None, iou_threshold: float = 0.5
) -> dict[str, float]:
    """Match two instance maps and return the raw counts PQ is built from.

    Pixels under ``ignore`` are removed from both maps, and predictions lying mostly inside
    ignored regions are dropped: nobody annotated there, so they are neither right nor wrong.
    """
    if ignore is not None:
        pred = _drop_mostly_ignored(pred, ignore)
        true = np.where(ignore, 0, true)
        pred = np.where(ignore, 0, pred)
    iou = pairwise_iou(true, pred)
    matched = iou > iou_threshold
    true_positives = int(matched.sum())
    foreground_true, foreground_pred = true > 0, pred > 0
    return {
        "tp": float(true_positives),
        "fp": float(iou.shape[1] - true_positives),
        "fn": float(iou.shape[0] - true_positives),
        "iou_sum": float(iou[matched].sum()),
        "intersection": float((foreground_true & foreground_pred).sum()),
        "pixels": float(foreground_true.sum() + foreground_pred.sum()),
    }


def aggregate(counts: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Sum per-tile counts and derive PQ, DQ (= detection F1), SQ, precision, recall and foreground Dice."""
    total = {key: 0.0 for key in ("tp", "fp", "fn", "iou_sum", "intersection", "pixels")}
    for item in counts:
        for key in total:
            total[key] += item[key]
    tp, fp, fn = total["tp"], total["fp"], total["fn"]
    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if tp + fp + fn else 0.0
    sq = total["iou_sum"] / tp if tp else 0.0
    return {
        "pq": dq * sq,
        "dq": dq,
        "sq": sq,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "dice": 2 * total["intersection"] / total["pixels"] if total["pixels"] else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def panoptic_quality(true: np.ndarray, pred: np.ndarray, **kwargs) -> dict[str, float]:
    """PQ and its parts for a single pair of instance maps."""
    return aggregate([instance_counts(true, pred, **kwargs)])


def _drop_mostly_ignored(instances: np.ndarray, ignore: np.ndarray) -> np.ndarray:
    size = int(instances.max()) + 1
    area = np.bincount(instances.ravel(), minlength=size)
    ignored = np.bincount(instances.ravel(), weights=ignore.ravel().astype(np.float64), minlength=size)
    drop = ignored > 0.5 * area
    drop[0] = False
    return np.where(drop[instances], 0, instances)
