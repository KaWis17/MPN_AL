"""Linear and kNN probes on embeddings, validated by patient.

Usage on any table with an ``embedding`` column (tile embeddings, or ``cells/*.parquet`` joined with labels)::

    python scripts/probe_embeddings.py table.parquet --label class_name --group patient_id --center-by slide_id

* Folds never split a group (patient): tiles/cells of one patient are not independent.
* ``--center-by`` subtracts the mean embedding of each slide (or any column) first. Embeddings otherwise
  cluster by source image, which a probe can mistake for signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def center_by(vectors: np.ndarray, groups: Sequence) -> np.ndarray:
    """Subtract the mean vector of each group (e.g. slide)."""
    groups = np.asarray(groups)
    out = vectors.astype(np.float64).copy()
    for value in np.unique(groups):
        mask = groups == value
        out[mask] -= out[mask].mean(axis=0)
    return out


def probe(
    vectors: np.ndarray, labels: Sequence, patients: Sequence, *, folds: int = 5, neighbours: int = 5, seed: int = 0
) -> dict[str, float]:
    """Cross-validated balanced accuracy and macro F1 of logistic regression and kNN, folds grouped by patient."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels, patients = np.asarray(labels), np.asarray(patients)
    n_patients = len(np.unique(patients))
    if n_patients < 2:
        raise ValueError("need at least two patients for a grouped cross-validation")
    models = {
        "linear": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1, random_state=seed)),
        "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=neighbours)),
    }
    splitter = GroupKFold(n_splits=min(folds, n_patients))
    result: dict[str, float] = {"patients": n_patients, "samples": len(labels)}
    for name, factory in models.items():
        predicted = np.empty(len(labels), dtype=labels.dtype)
        for train, test in splitter.split(vectors, labels, patients):
            if len(np.unique(labels[train])) < 2:  # a fold whose training patients hold one class only
                predicted[test] = labels[train][0]
                continue
            predicted[test] = factory().fit(vectors[train], labels[train]).predict(vectors[test])
        result[f"{name}_balanced_accuracy"] = float(balanced_accuracy_score(labels, predicted))
        result[f"{name}_macro_f1"] = float(f1_score(labels, predicted, average="macro"))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Patient-grouped linear/kNN probe on an embedding table.")
    parser.add_argument("table", type=Path, help="Parquet table with an 'embedding' column")
    parser.add_argument("--label", required=True, help="column with the class")
    parser.add_argument("--group", required=True, help="column with the patient id")
    parser.add_argument("--center-by", help="column (e.g. slide_id) whose mean embedding is subtracted first")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(argv)
    table = pd.read_parquet(args.table).dropna(subset=[args.label, args.group])
    vectors = np.stack(table["embedding"].to_numpy()).astype(np.float64)
    if args.center_by:
        vectors = center_by(vectors, table[args.center_by])
    for name, value in probe(vectors, table[args.label], table[args.group], folds=args.folds).items():
        print(f"{name}: {value:.4g}" if isinstance(value, float) else f"{name}: {value}")
    return 0
