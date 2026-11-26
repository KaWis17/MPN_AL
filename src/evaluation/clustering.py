"""KMeans clustering of embeddings and the metrics that say how good and how stable the clusters are.

Everything works on one model's embedding table (one row per tile or cell). Comparing models is
``embedding_compare.py``. The pieces:

``preprocess``            centre by preparation (optional), L2-normalise, reduce with PCA (optional);
``run_grid``              KMeans for a grid of ``k`` and seeds, one tidy row per ``k``;
``seed_stability``        agreement (ARI/AMI) between runs that differ only in the initialisation;
``bootstrap_stability``   fit on a share of the patients, assign the others, compare with the fit on everything;
``label_agreement``       NMI/ARI/purity of clusters against known classes, and against a shuffled control;
``confounding``           how much of the clustering is just the patient or the slide (batch effect).

Choices that matter and why
---------------------------
* Embeddings are L2-normalised, so KMeans (Euclidean) behaves like cosine clustering, and cosine silhouette applies.
* Each seed is one KMeans start (``n_init=1``): running several and keeping the best would hide the sensitivity to the
  initialisation that ``seed_stability`` is there to measure.
* Stability against *data* holds out whole patients: tiles of one patient are not independent, so resampling tiles would
  flatter the result.
* Davies-Bouldin and Calinski-Harabasz depend on the dimension and the scale of the space; compare them across ``k`` of one
  model, not across models.
"""

from __future__ import annotations

import itertools
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

DEFAULT_KS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def preprocess(
    vectors: np.ndarray,
    *,
    center_groups: Sequence | None = None,
    pca_dim: int | None = None,
    l2: bool = True,
) -> np.ndarray:
    """Embeddings ready for clustering, float64.

    Order: subtract the mean vector of each ``center_groups`` value (e.g. slide), L2-normalise, PCA to ``pca_dim`` components
    (fitted on these vectors only, no labels involved) and L2-normalise again. ``pca_dim`` is capped by what the data allows.
    Centring needs at least two samples per group; with very few per group the group mean is noisy and also removes the
    group's real share of each class, so treat it as a variant, not as the truth.
    """
    out = np.asarray(vectors, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError(f"expected [n, d] embeddings, got shape {out.shape}")
    if center_groups is not None:
        groups = np.asarray(center_groups)
        values, counts = np.unique(groups, return_counts=True)
        if (counts < 2).any():  # a group of one sample would become the zero vector
            raise ValueError(f"cannot centre groups of a single sample: {values[counts < 2][:5].tolist()}")
        out = out.copy()
        for value in np.unique(groups):
            mask = groups == value
            out[mask] -= out[mask].mean(axis=0)
    if l2:
        out = l2_normalize(out)
    if pca_dim is not None:
        components = min(pca_dim, out.shape[0] - 1, out.shape[1])
        out = PCA(n_components=components, svd_solver="full").fit_transform(out)
        if l2:
            out = l2_normalize(out)
    return out


def fit_kmeans(vectors: np.ndarray, k: int, seed: int, *, n_init: int = 1) -> KMeans:
    """A fitted KMeans (k-means++ initialisation, ``n_init`` starts of which the lowest-inertia one is kept)."""
    if k > len(vectors):
        raise ValueError(f"k = {k} exceeds the {len(vectors)} samples")
    return KMeans(n_clusters=k, n_init=n_init, init="k-means++", random_state=seed).fit(vectors)


def kmeans_labels(vectors: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, float]:
    """One KMeans start (k-means++ initialisation): labels and inertia."""
    model = fit_kmeans(vectors, k, seed)
    return model.labels_, float(model.inertia_)


def internal_metrics(vectors: np.ndarray, labels: np.ndarray, *, sample_size: int = 5000, seed: int = 0) -> dict[str, float]:
    """Cosine silhouette, Davies-Bouldin and Calinski-Harabasz; NaN when there are fewer than two clusters."""
    present = len(np.unique(labels))
    if present < 2 or present >= len(vectors):
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    return {
        "silhouette": float(
            silhouette_score(vectors, labels, metric="cosine", sample_size=min(sample_size, len(vectors)), random_state=seed)
            if len(vectors) > sample_size else silhouette_score(vectors, labels, metric="cosine")
        ),
        "davies_bouldin": float(davies_bouldin_score(vectors, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(vectors, labels)),
    }


def seed_stability(label_sets: Sequence[np.ndarray]) -> dict[str, float]:
    """Mean and sd of ARI and AMI over all pairs of clusterings that differ only in the seed."""
    pairs = list(itertools.combinations(range(len(label_sets)), 2))
    if not pairs:
        return {"ari_seed_mean": np.nan, "ari_seed_sd": np.nan, "ami_seed_mean": np.nan, "ami_seed_sd": np.nan}
    ari = [adjusted_rand_score(label_sets[a], label_sets[b]) for a, b in pairs]
    ami = [adjusted_mutual_info_score(label_sets[a], label_sets[b]) for a, b in pairs]
    return {"ari_seed_mean": float(np.mean(ari)), "ari_seed_sd": float(np.std(ari)),
            "ami_seed_mean": float(np.mean(ami)), "ami_seed_sd": float(np.std(ami))}


def bootstrap_stability(
    vectors: np.ndarray,
    groups: Sequence,
    k: int,
    *,
    n_boot: int = 20,
    fraction: float = 0.8,
    seed: int = 0,
    reference: np.ndarray | None = None,
) -> dict[str, float]:
    """Agreement of held-out patients' clusters with the clustering of all data.

    Each repetition fits KMeans on a random ``fraction`` of the patients, assigns *every* sample to the nearest centroid and
    compares, on the samples of the patients that were left out, with ``reference`` (KMeans on everything, seed 0). With fewer
    than two patients, or when no patient is left out, the result is NaN: there is nothing honest to hold out.
    """
    groups = np.asarray(groups)
    unique = np.unique(groups)
    nan = {"ari_heldout_mean": np.nan, "ari_heldout_sd": np.nan, "bootstrap_runs": 0}
    if len(unique) < 2 or n_boot <= 0:
        return nan
    if reference is None:
        reference, _ = kmeans_labels(vectors, k, 0)
    rng = np.random.default_rng(seed)
    keep = int(np.clip(round(fraction * len(unique)), 1, len(unique) - 1))
    scores = []
    for _ in range(n_boot):
        fitted = rng.choice(unique, size=keep, replace=False)
        inside = np.isin(groups, fitted)
        if inside.sum() < k:
            continue
        model = KMeans(n_clusters=k, n_init=1, init="k-means++", random_state=int(rng.integers(2**31 - 1))).fit(vectors[inside])
        held_out = ~inside
        scores.append(adjusted_rand_score(reference[held_out], model.predict(vectors[held_out])))
    if not scores:
        return nan
    return {"ari_heldout_mean": float(np.mean(scores)), "ari_heldout_sd": float(np.std(scores)), "bootstrap_runs": len(scores)}


def purity(labels: np.ndarray, truth: np.ndarray) -> float:
    """Share of samples that carry the majority class of their cluster."""
    total = 0
    for cluster in np.unique(labels):
        _, counts = np.unique(truth[labels == cluster], return_counts=True)
        total += counts.max()
    return float(total / len(labels))


def label_agreement(labels: np.ndarray, truth: Sequence, *, shuffle_seed: int | None = 0) -> dict[str, float]:
    """NMI, ARI and purity of ``labels`` against ``truth`` and the NMI/ARI against a shuffled copy of ``truth`` (chance level)."""
    truth = np.asarray(truth)
    out = {
        "nmi_label": float(normalized_mutual_info_score(truth, labels)),
        "ari_label": float(adjusted_rand_score(truth, labels)),
        "purity_label": purity(labels, truth),
    }
    if shuffle_seed is not None:
        shuffled = np.random.default_rng(shuffle_seed).permutation(truth)
        out["nmi_label_shuffled"] = float(normalized_mutual_info_score(shuffled, labels))
        out["ari_label_shuffled"] = float(adjusted_rand_score(shuffled, labels))
    return out


def confounding(labels: np.ndarray, nuisance: Sequence) -> dict[str, float]:
    """NMI and AMI between the clusters and a nuisance variable (patient, slide). High = the clusters are a batch effect."""
    nuisance = np.asarray(nuisance)
    return {"nmi_nuisance": float(normalized_mutual_info_score(nuisance, labels)),
            "ami_nuisance": float(adjusted_mutual_info_score(nuisance, labels))}


def run_grid(
    vectors: np.ndarray,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    seeds: Sequence[int] = tuple(range(10)),
    groups: Sequence | None = None,
    truth: Sequence | None = None,
    nuisances: Mapping[str, Sequence] | None = None,
    n_boot: int = 20,
    boot_fraction: float = 0.8,
    silhouette_sample: int = 5000,
) -> pd.DataFrame:
    """One row per ``k``: internal metrics (mean and sd over seeds), stability, and agreement with ``truth`` and ``groups``.

    ``vectors`` are already preprocessed (see :func:`preprocess`). ``groups`` are the patients: they define the held-out
    bootstrap and the confounding score. ``truth`` are known classes (optional). ``k`` larger than the sample count is
    skipped. ``nuisances`` maps names (slide, image size, scanner) to further variables whose confounding with the clusters is
    reported as ``nmi_<name>`` and ``ami_<name>``. Deterministic for fixed arguments.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if len(seeds) < 1:
        raise ValueError("need at least one seed")
    rows = []
    for k in ks:
        if k < 2 or k > len(vectors):
            continue
        runs = [kmeans_labels(vectors, k, seed) for seed in seeds]
        label_sets = [labels for labels, _ in runs]
        per_seed = [internal_metrics(vectors, labels, sample_size=silhouette_sample, seed=seed)
                    for labels, seed in zip(label_sets, seeds)]
        row: dict[str, float] = {"k": k, "n": len(vectors), "seeds": len(seeds), "inertia": float(np.mean([i for _, i in runs]))}
        for name in ("silhouette", "davies_bouldin", "calinski_harabasz"):
            values = np.array([m[name] for m in per_seed], dtype=float)
            row[name] = float(np.nanmean(values)) if not np.isnan(values).all() else np.nan
            row[f"{name}_sd"] = float(np.nanstd(values)) if not np.isnan(values).all() else np.nan
        row.update(seed_stability(label_sets))
        if groups is not None:
            row.update(bootstrap_stability(vectors, groups, k, n_boot=n_boot, fraction=boot_fraction, reference=label_sets[0]))
            for name, values in _mean_over_seeds([confounding(labels, groups) for labels in label_sets]).items():
                row[name] = values
        for extra, values_of in (nuisances or {}).items():
            scores = _mean_over_seeds([confounding(labels, values_of) for labels in label_sets])
            row[f"nmi_{extra}"], row[f"ami_{extra}"] = scores["nmi_nuisance"], scores["ami_nuisance"]
        if truth is not None:
            for name, values in _mean_over_seeds([label_agreement(labels, truth) for labels in label_sets]).items():
                row[name] = values
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_over_seeds(items: Sequence[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([item[key] for item in items])) for key in items[0]}
