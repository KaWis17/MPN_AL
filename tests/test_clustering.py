"""Clustering of embeddings, its metrics, and the cross-model comparison, checked on planted structure."""

import numpy as np
import pytest

from mpn_al.evaluation.clustering import (
    bootstrap_stability,
    confounding,
    internal_metrics,
    kmeans_labels,
    label_agreement,
    preprocess,
    purity,
    run_grid,
)
from mpn_al.evaluation.embedding_compare import knn_overlap, linear_cka, pairwise_matrix, partition_agreement


def planted(*, patients=6, per=30, classes=3, dim=24, noise=0.3, batch=0.0, seed=0):
    """``classes`` well separated groups; every patient has all of them; an optional shift per patient (batch effect)."""
    rng = np.random.default_rng(seed)
    centres = np.eye(dim)[:classes] * 3.0
    vectors, truth, owner = [], [], []
    for patient in range(patients):
        shift = batch * rng.normal(size=dim)
        for c in range(classes):
            vectors.append(centres[c] + shift + noise * rng.normal(size=(per, dim)))
            truth += [c] * per
            owner += [f"P{patient}"] * per
    return np.concatenate(vectors), np.array(truth), np.array(owner)


def test_preprocess_centres_normalises_and_reduces():
    vectors, _, owner = planted(patients=3, batch=2.0)
    centred = preprocess(vectors, center_groups=owner, l2=False)
    for patient in np.unique(owner):
        assert np.allclose(centred[owner == patient].mean(axis=0), 0, atol=1e-9)
    unit = preprocess(vectors)
    assert np.allclose(np.linalg.norm(unit, axis=1), 1)
    reduced = preprocess(vectors, pca_dim=8)
    assert reduced.shape == (len(vectors), 8) and np.allclose(np.linalg.norm(reduced, axis=1), 1)
    assert preprocess(vectors[:5], pca_dim=50).shape == (5, 4)  # capped by n - 1
    with pytest.raises(ValueError):
        preprocess(vectors[0])
    with pytest.raises(ValueError, match="single sample"):
        preprocess(vectors[:4], center_groups=["a", "a", "b", "c"])  # groups b and c would become zero vectors


def test_the_planted_number_of_clusters_is_recovered_and_is_stable():
    vectors, truth, owner = planted()
    table = run_grid(preprocess(vectors), ks=(2, 3, 4, 6, 8), seeds=range(5), groups=owner, truth=truth, n_boot=8)
    assert list(table["k"]) == [2, 3, 4, 6, 8]
    at = table.set_index("k")
    assert at["silhouette"].idxmax() == 3
    assert at.loc[3, "ari_label"] > 0.95 and at.loc[3, "purity_label"] > 0.95
    assert at.loc[3, "ari_seed_mean"] > 0.95 and at.loc[8, "ari_seed_mean"] < at.loc[3, "ari_seed_mean"]
    assert at.loc[3, "ari_heldout_mean"] > 0.9 and at.loc[3, "bootstrap_runs"] == 8
    assert at.loc[3, "nmi_label"] > 5 * at.loc[3, "nmi_label_shuffled"]  # chance level is far below the real agreement
    assert at.loc[3, "nmi_nuisance"] < 0.1  # classes are balanced over patients: no batch effect


def test_a_batch_effect_is_detected_and_removed_by_centring():
    vectors, truth, owner = planted(patients=3, noise=0.2, batch=6.0)
    raw = run_grid(preprocess(vectors), ks=(3,), seeds=range(3), groups=owner, truth=truth, n_boot=0).iloc[0]
    centred = run_grid(preprocess(vectors, center_groups=owner), ks=(3,), seeds=range(3), groups=owner, truth=truth, n_boot=0).iloc[0]
    assert raw["nmi_nuisance"] > 0.5 > 0.2 > centred["nmi_nuisance"]  # clusters followed the patient, then stopped to
    assert centred["ari_label"] > raw["ari_label"] + 0.5


def test_noise_lowers_stability():
    clean = run_grid(preprocess(planted(noise=0.2)[0]), ks=(3,), seeds=range(5), n_boot=0).iloc[0]
    noisy = run_grid(preprocess(planted(noise=4.0)[0]), ks=(3,), seeds=range(5), n_boot=0).iloc[0]
    assert clean["ari_seed_mean"] > noisy["ari_seed_mean"] and clean["silhouette"] > noisy["silhouette"]


def test_degenerate_inputs_do_not_crash():
    vectors, truth, owner = planted(patients=2, per=5)
    one_patient = bootstrap_stability(preprocess(vectors), np.array(["P0"] * len(vectors)), 3)
    assert np.isnan(one_patient["ari_heldout_mean"]) and one_patient["bootstrap_runs"] == 0
    constant = np.ones((20, 4))
    assert np.isnan(internal_metrics(constant, np.zeros(20, int))["silhouette"])  # one cluster only
    table = run_grid(constant, ks=(2, 50), seeds=(0, 1))  # k = 50 exceeds the 20 samples: skipped
    assert list(table["k"]) == [2]
    with pytest.raises(ValueError):
        kmeans_labels(constant, 30, 0)
    with pytest.raises(ValueError):
        run_grid(vectors, seeds=())


def test_grid_is_deterministic():
    vectors, truth, owner = planted(patients=4, per=15)
    x = preprocess(vectors)
    first = run_grid(x, ks=(2, 3), seeds=range(3), groups=owner, truth=truth, n_boot=4)
    second = run_grid(x, ks=(2, 3), seeds=range(3), groups=owner, truth=truth, n_boot=4)
    assert first.equals(second)


def test_label_metrics():
    truth = np.array([0, 0, 1, 1, 2, 2])
    assert purity(np.array([0, 0, 1, 1, 2, 2]), truth) == 1.0
    assert purity(np.zeros(6, int), truth) == pytest.approx(2 / 6)
    perfect = label_agreement(np.array([5, 5, 7, 7, 9, 9]), truth)
    assert perfect["nmi_label"] == pytest.approx(1.0) and perfect["ari_label"] == pytest.approx(1.0)
    assert confounding(np.array([0, 0, 1, 1, 2, 2]), truth)["nmi_nuisance"] == pytest.approx(1.0)


# ---- comparing models -------------------------------------------------------------------------


def random_rotation(dim, seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(dim, dim)))
    return q


def test_cka_and_knn_overlap_are_invariant_to_rotation_and_scale_but_not_to_other_spaces():
    x = np.random.default_rng(0).normal(size=(80, 12))
    rotated = 3.0 * x @ random_rotation(12, 1)
    other = np.random.default_rng(5).normal(size=(80, 12))
    assert linear_cka(x, x) == pytest.approx(1.0) and linear_cka(x, rotated) == pytest.approx(1.0)
    assert linear_cka(x, other) < 0.4
    assert knn_overlap(x, rotated, k=5) == pytest.approx(1.0)
    assert knn_overlap(x, other, k=5) < 0.15
    wide = np.random.default_rng(2).normal(size=(30, 100))  # more features than samples: Gram form
    assert linear_cka(wide, wide @ random_rotation(100, 3)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        linear_cka(x, x[:10])
    with pytest.raises(ValueError):
        knn_overlap(x, x, k=200)


def test_models_with_different_dimensions_can_be_compared():
    vectors, truth, _ = planted(per=20)
    low = preprocess(vectors, pca_dim=6)
    spaces = {"a": preprocess(vectors), "b": low, "c": np.random.default_rng(9).normal(size=(len(vectors), 50))}
    cka = pairwise_matrix(spaces, linear_cka)
    assert cka.loc["a", "a"] == pytest.approx(1.0) and cka.loc["a", "b"] > cka.loc["a", "c"] and cka.equals(cka.T)
    labels = {name: kmeans_labels(x, 3, 0)[0] for name, x in spaces.items()}
    agreement = partition_agreement(labels)
    assert agreement["ari"].loc["a", "b"] > 0.9 and agreement["ari"].loc["a", "c"] < 0.3
    assert agreement["ari"].loc["a", "b"] == pytest.approx(agreement["ari"].loc["b", "a"])
    with pytest.raises(ValueError, match="same samples"):
        partition_agreement({"a": labels["a"], "b": labels["b"][:10]})
