import numpy as np
import pandas as pd
import pytest

from mpn_al.evaluation.probe import center_by, main, probe


def make_data(seed=0):
    """Two classes separable along one axis; every slide adds its own large offset, patients own 2 slides."""
    rng = np.random.default_rng(seed)
    rows, labels, slides, patients = [], [], [], []
    for patient in range(6):
        for slide in range(2):
            offset = rng.normal(0, 4.0, 16)
            for _ in range(30):
                label = int(rng.integers(0, 2))
                vector = rng.normal(0, 1, 16) + offset
                vector[0] += 2.5 * label
                rows.append(vector), labels.append(label), slides.append(f"s{patient}{slide}"), patients.append(f"P{patient}")
    return np.array(rows), np.array(labels), np.array(slides), np.array(patients)


def test_centering_removes_the_slide_offset_and_improves_the_probe():
    vectors, labels, slides, patients = make_data()
    raw = probe(vectors, labels, patients)
    centred = probe(center_by(vectors, slides), labels, patients)
    assert np.allclose(center_by(vectors, slides)[slides == "s00"].mean(axis=0), 0)
    assert centred["linear_balanced_accuracy"] > 0.8
    assert centred["linear_balanced_accuracy"] > raw["linear_balanced_accuracy"] + 0.15
    assert centred["patients"] == 6 and centred["samples"] == len(labels)


def test_folds_never_split_a_patient_and_one_patient_is_refused():
    vectors, labels, _, patients = make_data()
    with pytest.raises(ValueError, match="two patients"):
        probe(vectors, labels, np.zeros(len(labels)))
    assert 0 <= probe(vectors, labels, patients, folds=10)["knn_macro_f1"] <= 1  # folds capped at the patient count


def test_cli(tmp_path, capsys):
    vectors, labels, slides, patients = make_data()
    path = tmp_path / "t.parquet"
    pd.DataFrame({"embedding": list(vectors.astype(np.float32)), "class_name": labels.astype(str), "patient_id": patients, "slide_id": slides}).to_parquet(path)
    assert main([str(path), "--label", "class_name", "--group", "patient_id", "--center-by", "slide_id"]) == 0
    assert "linear_balanced_accuracy" in capsys.readouterr().out
