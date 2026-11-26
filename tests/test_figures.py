"""Figures drawn from the predictions an experiment saved: error matching, scores, and the script end to end."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import TINY_VIT

from mpn_al.evaluation import instance_counts
from mpn_al.experiment import figures as fg
from mpn_al.experiment.runner import main as run_experiment
from mpn_al.training.synthetic import write_demo_dataset

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def disc(shape, centre, radius, label, canvas):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    canvas[(yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 <= radius**2] = label
    return canvas


def test_error_masks_agree_with_the_pq_counts():
    shape = (96, 96)
    reference, predicted = np.zeros(shape, np.int32), np.zeros(shape, np.int32)
    disc(shape, (20, 20), 8, 1, reference)  # found exactly
    disc(shape, (20, 20), 8, 1, predicted)
    disc(shape, (60, 20), 8, 2, reference)  # found far too small / shifted: no match
    disc(shape, (60, 32), 8, 2, predicted)
    disc(shape, (80, 70), 6, 3, reference)  # missed
    disc(shape, (30, 70), 6, 3, predicted)  # invented
    data = {"reference": reference, "instances": predicted, "ignore": np.zeros(shape, bool)}
    counts = instance_counts(reference, predicted)
    masks = fg.error_masks(data)
    sizes = {k: len(np.unique(v[v > 0])) for k, v in masks.items()}
    assert (sizes["tp"], sizes["fp"], sizes["fn"]) == (counts["tp"], counts["fp"], counts["fn"]) == (1, 2, 2)
    scores = fg.score_prediction(data)
    assert scores["pq"] == pytest.approx(counts["tp"] / (counts["tp"] + 0.5 * counts["fp"] + 0.5 * counts["fn"]))


def test_ignored_regions_are_neither_errors_nor_hits():
    shape = (64, 64)
    reference, predicted = np.zeros(shape, np.int32), np.zeros(shape, np.int32)
    disc(shape, (32, 32), 8, 1, predicted)
    ignore = np.zeros(shape, bool)
    ignore[16:48, 16:48] = True  # nobody annotated here: the prediction is not a false positive
    masks = fg.error_masks({"reference": reference, "instances": predicted, "ignore": ignore})
    assert all(not m.any() for m in masks.values())
    empty = np.zeros(shape, np.int32)
    nothing = fg.error_masks({"reference": empty, "instances": empty, "ignore": np.zeros(shape, bool)})
    assert all(not m.any() for m in nothing.values())  # no cells at all must not crash


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("figs")
    ids = write_demo_dataset(root, tiles=8, size=256, count=10)
    split = ["train"] * 4 + ["validation"] * 2 + ["test"] * 2
    pd.DataFrame({"tile_id": ids, "patient_id": [f"P{i // 2}" for i in range(8)], "split": split}).to_csv(
        root / "split.csv", index=False)
    (root / "grid.yaml").write_text(
        f"name: tiny\nmodels: ['{TINY_VIT}', 'timm:vit_small_patch16_224']\nfractions: [0.5, 1.0]\nseeds: [0, 1]\n"
        "train: {base_width: 8, epochs: 1, steps_per_epoch: 2, batch_size: 2, crop_size: 128}\n"
        "visualize: {per_split: 2, seed: 0}\n", encoding="utf-8")
    for index in range(8):
        assert run_experiment([
            "--grid", str(root / "grid.yaml"), "--task-index", str(index), "--images", str(root / "images"),
            "--masks", str(root / "masks"), "--split-file", str(root / "split.csv"), "--mpp", "0.25",
            "--out", str(root / "out"), "--random-init", "--device", "cpu", "--resamples", "20",
        ]) == 0
    return root


def test_runs_are_discovered_and_saved_scores_match_their_counts(experiment):
    runs = fg.discover_runs(experiment / "out")
    assert len(runs) == 8 and {r.fraction for r in runs} == {0.5, 1.0} and {r.seed for r in runs} == {0, 1}
    table = fg.scores_table(runs, "validation")
    assert len(table) == 8 * 2 and table["matches_counts"].all()
    assert {"pq", "dq", "sq", "tp", "fp", "fn", "group"} <= set(table.columns)


def test_script_writes_every_figure_kind_and_an_index(experiment, tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    import visualize_experiment

    out = tmp_path / "figs"
    code = visualize_experiment.main([
        str(experiment / "out"), "--images", str(experiment / "images"), "--out", str(out),
        "--zoom", "128", "--panel", "96",
    ])
    assert code == 0
    tiles = pd.read_csv(sorted((experiment / "out").rglob("index.csv"))[0])["tile_id"]
    for tile_id in tiles:
        for kind in fg.KINDS:
            assert (out / f"{tile_id}_{kind}.png").exists(), (tile_id, kind)
    page = (out / "index.html").read_text(encoding="utf-8")
    assert all(f"{tile_id}_gallery.png" in page for tile_id in tiles) and "pseudo-labels" in page
    assert pd.read_csv(out / "viz_scores.csv")["matches_counts"].all()


def test_the_test_split_needs_a_final_run(experiment, tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    import visualize_experiment

    with pytest.raises(SystemExit):
        visualize_experiment.main([str(experiment / "out"), "--images", str(experiment / "images"),
                                   "--out", str(tmp_path / "t"), "--split", "test"])
