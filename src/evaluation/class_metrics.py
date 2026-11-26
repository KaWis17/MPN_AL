"""Per-class panoptic quality, confidence intervals over patients, and the effort of correcting a proposal.

* ``per_class_counts`` gives the counts behind PQ separately for every class (HoVer-Net style mPQ: instances
  of one class are matched only with predictions of that class), to be summed over tiles and then turned
  into ratios by ``instance_metrics.aggregate``.
* ``bootstrap_ci`` resamples *patients*, not tiles: tiles of one patient are not independent, so a
  tile-level interval would be too narrow.
* ``correction_effort`` estimates how much a pathologist has to change in a proposal.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from .instance_metrics import aggregate, instance_counts, pairwise_iou

COUNT_KEYS = ("tp", "fp", "fn", "iou_sum", "intersection", "pixels")


def instance_class_ids(instances: np.ndarray, type_map: np.ndarray) -> np.ndarray:
    """Class of every instance (index = instance id, 0 = background): the majority of ``type_map`` under it."""
    count = int(instances.max())
    classes = np.zeros(count + 1, dtype=np.int64)
    if count == 0:
        return classes
    foreground = instances > 0
    width = int(type_map.max()) + 1
    joint = np.bincount(
        instances[foreground].astype(np.int64) * width + type_map[foreground].astype(np.int64),
        minlength=(count + 1) * width,
    ).reshape(count + 1, width)
    classes = joint.argmax(axis=1)
    classes[0] = 0
    return classes


def per_class_counts(
    true_instances: np.ndarray,
    true_classes: np.ndarray,
    pred_instances: np.ndarray,
    pred_classes: np.ndarray,
    num_classes: int,
    *,
    ignore: np.ndarray | None = None,
) -> dict[int, dict[str, float]]:
    """Counts per class id 1..K; ``*_classes`` map instance id -> class id (see ``instance_class_ids``)."""
    result = {}
    for class_id in range(1, num_classes + 1):
        truth = np.where(true_classes[true_instances] == class_id, true_instances, 0)
        pred = np.where(pred_classes[pred_instances] == class_id, pred_instances, 0)
        result[class_id] = instance_counts(truth, pred, ignore=ignore)
    return result


def mean_pq(per_class: Mapping[int, Mapping[str, float]]) -> dict[str, float]:
    """mPQ: the mean PQ over the classes that occur in the ground truth or the predictions."""
    scores = {}
    for class_id, counts in per_class.items():
        if counts["tp"] + counts["fp"] + counts["fn"] > 0:
            scores[class_id] = aggregate([counts])["pq"]
    return {"mpq": float(np.mean(list(scores.values()))) if scores else 0.0, **{f"pq_class_{k}": v for k, v in scores.items()}}


def sum_counts(counts: Iterable[Mapping[str, float]]) -> dict[str, float]:
    total = {key: 0.0 for key in COUNT_KEYS}
    for item in counts:
        for key in COUNT_KEYS:
            total[key] += item[key]
    return total


def bootstrap_ci(
    per_patient: Mapping[str, Sequence[Mapping[str, float]]],
    metric: str = "pq",
    *,
    resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Point estimate and percentile interval of ``metric`` (see ``aggregate``) over patients.

    ``per_patient`` maps a patient id to the count dictionaries of that patient's tiles. With fewer than
    three patients no interval is meaningful: the bounds are NaN.
    """
    patients = list(per_patient)
    point = aggregate([c for p in patients for c in per_patient[p]])[metric]
    if len(patients) < 3:
        return {"estimate": point, "low": float("nan"), "high": float("nan"), "patients": len(patients)}
    rng = np.random.default_rng(seed)
    summed = {p: sum_counts(per_patient[p]) for p in patients}
    values = []
    for _ in range(resamples):
        chosen = rng.integers(0, len(patients), len(patients))
        values.append(aggregate([summed[patients[i]] for i in chosen])[metric])
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return {"estimate": point, "low": float(low), "high": float(high), "patients": len(patients)}


def correction_effort(
    true_instances: np.ndarray, pred_instances: np.ndarray, *, ignore: np.ndarray | None = None, adjust_iou: float = 0.9
) -> dict[str, float]:
    """What a pathologist would have to change to turn the proposal into the ground truth.

    ``objects_to_delete`` are predictions without a match (IoU > 0.5), ``objects_to_add`` are missed cells,
    ``objects_to_adjust`` matched cells whose outline differs (IoU below ``adjust_iou``), and
    ``pixels_to_fix`` the pixels where the two foregrounds disagree.
    """
    if ignore is not None:
        true_instances = np.where(ignore, 0, true_instances)
        pred_instances = np.where(ignore, 0, pred_instances)
    iou = pairwise_iou(true_instances, pred_instances)
    matched = iou > 0.5
    best = np.where(matched, iou, 0).max(axis=1) if iou.size else np.zeros(0)
    return {
        "objects_to_delete": float(iou.shape[1] - matched.sum()),
        "objects_to_add": float(iou.shape[0] - matched.sum()),
        "objects_to_adjust": float(((best > 0) & (best < adjust_iou)).sum()),
        "pixels_to_fix": float(((true_instances > 0) != (pred_instances > 0)).sum()),
    }
