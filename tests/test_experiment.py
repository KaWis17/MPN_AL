"""Metrics for the experiment and the grid runner."""

import json

import numpy as np
import pandas as pd
import pytest
from conftest import TINY_VIT

from mpn_al.evaluation import (
    aggregate,
    bootstrap_ci,
    correction_effort,
    instance_class_ids,
    instance_counts,
    mean_pq,
    per_class_counts,
)
from mpn_al.experiment import expand_grid
from mpn_al.experiment.runner import main, nested_subset, predicted_classes, read_split
from mpn_al.training.synthetic import write_demo_dataset


def two_class_scene():
    truth = np.zeros((40, 40), dtype=np.int32)
    truth[2:12, 2:12], truth[20:30, 20:30] = 1, 2
    types = np.zeros_like(truth, dtype=np.uint8)
    types[truth == 1], types[truth == 2] = 1, 2
    return truth, types


def test_per_class_counts_separate_the_classes():
    truth, types = two_class_scene()
    true_classes = instance_class_ids(truth, types)
    assert true_classes.tolist() == [0, 1, 2]
    pred_classes = np.array([0, 1, 1])  # the second cell is found but called class 1
    counts = per_class_counts(truth, true_classes, truth, pred_classes, 2)
    assert (counts[1]["tp"], counts[1]["fp"], counts[1]["fn"]) == (1, 1, 0)
    assert (counts[2]["tp"], counts[2]["fp"], counts[2]["fn"]) == (0, 0, 1)
    scores = mean_pq(counts)
    assert scores["pq_class_2"] == 0 and 0 < scores["pq_class_1"] < 1 and scores["mpq"] == pytest.approx(scores["pq_class_1"] / 2)


def test_predicted_classes_follow_the_probability_mass():
    instances = np.zeros((8, 8), dtype=np.int32)
    instances[:4], instances[4:] = 1, 2
    probs = np.zeros((3, 8, 8), dtype=np.float32)
    probs[1, :4], probs[2, 4:] = 0.9, 0.9
    assert predicted_classes(instances, probs).tolist() == [0, 1, 2]
    assert predicted_classes(instances, None).tolist() == [0, 1, 1]


def test_bootstrap_interval_is_over_patients():
    perfect = instance_counts(*(two_class_scene()[0],) * 2)
    empty = instance_counts(two_class_scene()[0], np.zeros((40, 40), np.int32))
    groups = {f"p{i}": [perfect if i % 2 else empty] for i in range(8)}
    interval = bootstrap_ci(groups, "pq", resamples=300)
    assert interval["estimate"] == pytest.approx(aggregate([perfect] * 4 + [empty] * 4)["pq"])
    assert interval["low"] < interval["estimate"] < interval["high"] and interval["patients"] == 8
    assert np.isnan(bootstrap_ci({"a": [perfect], "b": [empty]}, "pq")["low"])  # two patients: no honest interval


def test_correction_effort():
    truth, _ = two_class_scene()
    pred = np.zeros_like(truth)
    pred[2:12, 2:12] = 1          # exact
    pred[20:30, 22:32] = 2        # shifted: needs adjusting
    pred[35:38, 35:38] = 3        # spurious
    effort = correction_effort(truth, pred)
    assert effort == {"objects_to_delete": 1.0, "objects_to_add": 0.0, "objects_to_adjust": 1.0, "pixels_to_fix": 40.0 + 9.0}


def test_grid_expansion_is_deterministic_and_validated():
    document = {"models": ["a", "b"], "seg_mpp": [None, 0.5], "fractions": [0.5, 1.0], "seeds": [0, 1]}
    runs = expand_grid(document)
    assert len(runs) == 16 and runs == expand_grid(document)
    assert runs[0].run_id == "a__native__f050__s0" and runs[-1].run_id == "b__0.5um__f100__s1"
    assert {r.model for r in runs[:8]} == {"a"}  # a model is loaded once per block
    with pytest.raises(ValueError, match="fractions"):
        expand_grid({**document, "fractions": [0]})
    with pytest.raises(ValueError, match="unknown training keys"):
        expand_grid({**document, "train": {"epoch": 3}})


def test_subsets_are_nested_per_seed():
    ids = [f"t{i}" for i in range(20)]
    small, large = nested_subset(ids, 0.25, 3), nested_subset(ids, 0.5, 3)
    assert len(small) == 5 and len(large) == 10 and set(small) <= set(large)
    assert nested_subset(ids, 0.25, 3) == small and nested_subset(ids, 0.25, 4) != small


def test_split_file_may_not_share_patients(tmp_path):
    path = tmp_path / "split.csv"
    pd.DataFrame({"tile_id": ["a", "b"], "patient_id": ["P1", "P1"], "split": ["train", "test"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="more than one split"):
        read_split(path)


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("exp")
    ids = write_demo_dataset(root, tiles=8, size=256, count=10)
    split = ["train"] * 4 + ["validation"] * 2 + ["test"] * 2
    pd.DataFrame({"tile_id": ids, "patient_id": [f"P{i // 2}" for i in range(8)], "split": split}).to_csv(root / "split.csv", index=False)
    (root / "grid.yaml").write_text(
        f"name: tiny\nmodels: ['{TINY_VIT}']\nfractions: [0.5, 1.0]\nseeds: [0]\n"
        "train: {base_width: 8, epochs: 1, steps_per_epoch: 3, batch_size: 2, crop_size: 128}\n", encoding="utf-8")
    return root


def run(root, index, *extra):
    return main([
        "--grid", str(root / "grid.yaml"), "--task-index", str(index), "--images", str(root / "images"),
        "--masks", str(root / "masks"), "--split-file", str(root / "split.csv"), "--mpp", "0.25",
        "--out", str(root / "out"), "--random-init", "--device", "cpu", "--resamples", "50", *extra,
    ])


def test_a_run_trains_on_a_subset_and_never_touches_the_test_split(experiment):
    assert main(["--grid", str(experiment / "grid.yaml"), "--list"]) == 0
    assert run(experiment, 0) == 0
    (folder,) = (experiment / "out" / "tiny").glob("*__native__f050__s0")
    result = json.loads((folder / "result.json").read_text())
    assert len(result["train_tiles"]) == 2 and set(result["metrics"]) == {"validation"}
    assert result["metrics"]["validation"]["patients"] == 1 and "mpq" in result["metrics"]["validation"]
    splits = {json.loads(line)["split"] for line in (folder / "tile_counts.jsonl").read_text().splitlines()}
    assert splits == {"validation"}
    assert (folder / "provenance.json").exists()


def test_final_flag_opens_the_test_split_and_skip_existing_skips(experiment):
    assert run(experiment, 1, "--final") == 0
    (folder,) = (experiment / "out" / "tiny").glob("*__native__f100__s0")
    result = json.loads((folder / "result.json").read_text())
    assert set(result["metrics"]) == {"validation", "test"} and len(result["train_tiles"]) == 4
    stamp = (folder / "result.json").stat().st_mtime_ns
    assert run(experiment, 1, "--final", "--skip-existing") == 0
    assert (folder / "result.json").stat().st_mtime_ns == stamp
    with pytest.raises(SystemExit):
        run(experiment, 5)  # outside the grid
