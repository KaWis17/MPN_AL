"""Tile selection for figures and the predictions an experiment run keeps for them."""

import json

import numpy as np
import pandas as pd
import pytest
from conftest import TINY_VIT, synthetic_cells

from mpn_al.evaluation import instance_counts
from mpn_al.evaluation.visual_compare import token_pca_grid
from mpn_al.experiment.runner import main
from mpn_al.experiment.viz import VizConfig, load_visualize, select_viz_tiles
from mpn_al.training.dataset import TrainingTile
from mpn_al.training.synthetic import write_demo_dataset


def make_tile(tile_id, size, count, seed):
    image, instances = synthetic_cells(size=size, count=count, seed=seed)
    return TrainingTile(tile_id, image, instances, (instances > 0).astype(np.uint8),
                        np.zeros(instances.shape, bool), objects=[None] * int(instances.max()))


@pytest.fixture()
def tiles():
    sizes = [128, 128, 128, 128, 128, 128, 192, 192]
    counts = [4, 20, 8, 14, 6, 10, 8, 12]
    return [make_tile(f"t{i}", s, c, i) for i, (s, c) in enumerate(zip(sizes, counts))]


def patients_of(tiles):
    return {t.tile_id: f"P{i // 2}" for i, t in enumerate(tiles)}  # two tiles per patient


def test_selection_depends_only_on_the_data_and_the_seed(tiles):
    patients = patients_of(tiles)
    first = select_viz_tiles(tiles, patients, 4, seed=0)
    assert first == select_viz_tiles(list(reversed(tiles)), patients, 4, seed=0)  # order of the tiles is irrelevant
    assert select_viz_tiles(tiles, patients, 4, seed=1) != first or {r["group"] for r in first} <= {"dense", "atypical"}
    assert [r["group"] for r in first].count("dense") == 1 and [r["group"] for r in first].count("atypical") == 1
    atypical = next(r for r in first if r["group"] == "atypical")
    assert (atypical["height"], atypical["width"]) == (192, 192)  # the 128 px size is the modal one
    dense = next(r for r in first if r["group"] == "dense")
    assert dense["density"] == max(r["density"] for r in select_viz_tiles(tiles, patients, 8, seed=0))


def test_one_tile_per_patient_before_any_patient_repeats(tiles):
    patients = patients_of(tiles)
    chosen = select_viz_tiles(tiles, patients, 4, seed=3)
    assert len({r["patient_id"] for r in chosen}) == 4 and len({r["tile_id"] for r in chosen}) == 4
    everything = select_viz_tiles(tiles, patients, 100)
    assert len(everything) == len(tiles)  # asking for more than exists returns all, no duplicates
    assert select_viz_tiles(tiles, patients, 0) == [] and select_viz_tiles([], patients, 3) == []


def test_visualize_section_is_validated(tmp_path):
    path = tmp_path / "grid.yaml"
    path.write_text("name: x\n", encoding="utf-8")
    assert load_visualize(path) == VizConfig()
    path.write_text("visualize: {per_split: 5, seed: 2}\n", encoding="utf-8")
    assert load_visualize(path) == VizConfig(per_split=5, seed=2)
    path.write_text("visualize: {per_split: 5, colour: red}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="colour"):
        load_visualize(path)
    with pytest.raises(ValueError):
        VizConfig(per_split=-1)


def test_token_pca_grid_is_deterministic():
    tokens = np.random.default_rng(0).normal(size=(16, 5, 7)).astype(np.float32)
    first = token_pca_grid(tokens)
    assert first.shape == (5, 7, 3) and first.dtype == np.uint8 and np.array_equal(first, token_pca_grid(tokens))
    assert token_pca_grid(tokens[:2]).shape == (5, 7, 3)  # fewer than three channels still works


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("viz")
    ids = write_demo_dataset(root, tiles=8, size=256, count=10)
    split = ["train"] * 4 + ["validation"] * 2 + ["test"] * 2
    pd.DataFrame({"tile_id": ids, "patient_id": [f"P{i // 2}" for i in range(8)], "split": split}).to_csv(
        root / "split.csv", index=False)
    (root / "grid.yaml").write_text(
        f"name: tiny\nmodels: ['{TINY_VIT}']\nfractions: [1.0]\nseeds: [0]\n"
        "train: {base_width: 8, epochs: 1, steps_per_epoch: 3, batch_size: 2, crop_size: 128}\n"
        "visualize: {per_split: 2, seed: 0}\n", encoding="utf-8")
    return root


def run(root, out, *extra):
    return main([
        "--grid", str(root / "grid.yaml"), "--task-index", "0", "--images", str(root / "images"),
        "--masks", str(root / "masks"), "--split-file", str(root / "split.csv"), "--mpp", "0.25",
        "--out", str(root / out), "--random-init", "--device", "cpu", "--resamples", "20", *extra,
    ])


def run_dir(root, out):
    (folder,) = (root / out / "tiny").glob("*")
    return folder


def test_run_keeps_predictions_that_reproduce_its_scores_and_never_the_test_split(experiment):
    assert run(experiment, "plain") == 0
    folder = run_dir(experiment, "plain")
    index = pd.read_csv(folder / "viz" / "index.csv")
    assert set(index["split"]) == {"validation"} and len(index) == 2  # no --final: no test tile is kept
    assert not (folder / "viz" / "test").exists()
    assert json.loads((folder / "result.json").read_text())["viz"]["tiles"] == 2

    counts = {json.loads(line)["tile_id"]: json.loads(line)["counts"]
              for line in (folder / "tile_counts.jsonl").read_text().splitlines()}
    for tile_id in index["tile_id"]:
        with np.load(folder / "viz" / "validation" / f"{tile_id}.npz") as saved:
            assert saved["prob"].dtype == np.uint8 and saved["hv"].dtype == np.int8
            assert saved["token_pca"].ndim == 3 and saved["token_pca"].shape[2] == 3
            again = instance_counts(saved["reference"].astype(np.int32), saved["instances"].astype(np.int32),
                                    ignore=saved["ignore"])
        for key in ("tp", "fp", "fn"):
            assert again[key] == counts[tile_id][key]


def test_final_run_also_keeps_test_tiles_and_flags_control_it(experiment):
    assert run(experiment, "final", "--final") == 0
    index = pd.read_csv(run_dir(experiment, "final") / "viz" / "index.csv")
    assert set(index["split"]) == {"validation", "test"} and len(index) == 4

    assert run(experiment, "off", "--no-viz") == 0
    folder = run_dir(experiment, "off")
    assert not (folder / "viz").exists() and "viz" not in json.loads((folder / "result.json").read_text())

    assert run(experiment, "more", "--viz-per-split", "1") == 0
    assert len(pd.read_csv(run_dir(experiment, "more") / "viz" / "index.csv")) == 1
