"""Summaries of experiment results."""

import json

import numpy as np
import pytest

from mpn_al.experiment.aggregate import curves, holm, load_runs, main, paired_tests


def write_result(root, model, fraction, seed, pq, mpp=None, split="validation"):
    folder = root / "g" / f"{model}-{fraction}-{seed}"
    folder.mkdir(parents=True)
    metrics = {
        "tiles": 4, "patients": 2, "pq": pq, "dq": pq, "sq": 1.0, "mpq": pq, "seconds_per_tile": 0.5,
        "pq_ci": {"low": pq - 0.05, "high": pq + 0.05},
        "effort": {"objects_to_add": 4, "objects_to_delete": 4, "objects_to_adjust": 0, "pixels_to_fix": 400},
    }
    (folder / "result.json").write_text(json.dumps({
        "run_id": folder.name, "config": {"model": model, "seg_mpp": mpp, "fraction": fraction, "seed": seed},
        "train_tiles": ["a"] * int(10 * fraction), "train_seconds": 3.0, "metrics": {split: metrics}}))


@pytest.fixture()
def results(tmp_path):
    rng = np.random.default_rng(0)
    for seed in range(4):
        for fraction in (0.5, 1.0):
            noise = rng.normal(0, 0.01)
            write_result(tmp_path, "ref", fraction, seed, 0.40 + 0.1 * fraction + noise)
            write_result(tmp_path, "better", fraction, seed, 0.50 + 0.1 * fraction + noise)
            write_result(tmp_path, "same", fraction, seed, 0.40 + 0.1 * fraction + noise)
    return tmp_path


def test_curves_average_over_seeds(results):
    runs = load_runs(results)
    assert len(runs) == 24 and set(runs["seg_mpp"]) == {-1.0}
    table = curves(runs)
    row = table[(table["model"] == "better") & (table["fraction"] == 1.0)].iloc[0]
    assert row["seeds"] == 4 and row["pq"] == pytest.approx(0.6, abs=0.02) and row["pixels_to_fix_per_tile"] == 100
    with pytest.raises(FileNotFoundError):
        load_runs(results, "test")  # a split that was never evaluated


def test_paired_comparison_with_holm_correction(results):
    table = paired_tests(load_runs(results), "ref").set_index("model")
    assert table.loc["better", "pairs"] == 8 and table.loc["better", "mean_difference"] == pytest.approx(0.1)
    assert table.loc["better", "p_holm"] < 0.05
    assert np.isnan(table.loc["same", "p"]) or table.loc["same", "p_holm"] > 0.05  # identical scores: no evidence
    with pytest.raises(ValueError):
        paired_tests(load_runs(results), "missing")


def test_holm_adjustment():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_cli_writes_the_files(results):
    assert main([str(results), "--reference", "ref"]) == 0
    for name in ("runs.csv", "curves.csv", "comparison.csv", "report.md"):
        assert (results / name).exists()
    assert "Paired comparison" in (results / "report.md").read_text()
