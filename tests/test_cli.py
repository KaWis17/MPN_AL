"""The command-line scripts, run as a user would run them (in-process, tiny random models)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import TINY_VIT, synthetic_cells
from PIL import Image

from mpn_al.embeddings.cli import main_extract, main_models
from mpn_al.segmenter.cli import main as main_segment
from mpn_al.training.train_head import main as main_train

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Six 256 px tiles named like real tile ids, with instance masks and QuPath-style annotations."""
    from mpn_al.image_backend import TileRef
    from mpn_al.segmenter import LabelSchema, cells_to_feature_collection, instances_to_cells, write_geojson

    root = tmp_path_factory.mktemp("dataset")
    (root / "images").mkdir()
    (root / "masks").mkdir()
    schema = LabelSchema.single_class("meg_norm")
    for index in range(6):
        image, instances = synthetic_cells(size=256, count=14, seed=index)
        x = 4096 + 256 * index
        tile_id = f"tl_S1_x{x}_y8192_s256"
        Image.fromarray(image).save(root / "images" / f"{tile_id}.png")
        np.save(root / "masks" / f"{tile_id}.npy", instances)
        cells = instances_to_cells(instances, origin=(x, 8192), labels=schema, simplify=0.5)
        write_geojson(root / "annotations" / f"{tile_id}.geojson", cells_to_feature_collection(cells, schema))
    (root / "labels.yaml").write_text(
        "name: megakaryocyte_binary\n"
        "classes:\n  - {id: 1, name: megakaryocyte, aliases: [meg_norm]}\n"
        "ignore: [uncertain]\nroi: [ROI]\n",
        encoding="utf-8",
    )
    return root


def test_list_base_models_prints_every_candidate(capsys):
    assert main_models([]) == 0
    listing = capsys.readouterr().out
    assert "uni2-h" in listing and "h-optimus-1" in listing and "CC-BY-NC-ND-4.0" in listing
    assert main_models(["--format", "markdown"]) == 0
    assert capsys.readouterr().out.startswith("| name | architecture |")
    assert main_models(["--format", "json"]) == 0
    assert {"name", "license", "gated", "url"} <= set(json.loads(capsys.readouterr().out)[0])


def test_check_loads_a_model_and_runs_native_and_larger_inputs(capsys):
    assert main_models(["--check", TINY_VIT, "--random-init", "--device", "cpu"]) == 0
    report = capsys.readouterr().out
    assert '"embed_dim": 192' in report and '"patch_size": 16' in report
    assert "input 224x224: patch tokens (1, 192, 14, 14)" in report
    assert "input 448x448: patch tokens (1, 192, 28, 28)" in report and report.rstrip().endswith("ok")


def test_extract_embeddings_writes_tile_and_window_tables(dataset, tmp_path):
    out = tmp_path / "rounds" / "round_00" / "embeddings.parquet"
    code = main_extract([
        "--base-model", TINY_VIT, "--random-init", "--device", "cpu", "--images", str(dataset / "images"),
        "--mpp", "0.5", "--out", str(out), "--windows-out", str(tmp_path / "windows.parquet"),
        "--windows-table", str(tmp_path / "model_windows.parquet"), "--limit", "4", "--batch-size", "5",
    ])
    assert code == 0
    tiles = pd.read_parquet(out)
    assert len(tiles) == 4 and tiles["tile_id"].str.startswith("tl_S1_x").all()
    assert (tiles["n_windows"] == 4).all() and (tiles["slide_id"] == "S1").all()  # 256 px at 0.5 um/px: 2x2 windows
    assert np.stack(tiles["embedding"].to_numpy()).shape == (4, 192)
    windows = pd.read_parquet(tmp_path / "windows.parquet")
    geometry = pd.read_parquet(tmp_path / "model_windows.parquet")
    assert len(windows) == len(geometry) == 16 and set(windows["window_id"]) == set(geometry["window_id"])
    assert (geometry["role"] == "embed").all() and (geometry["size_model"] == 224).all()
    run = json.loads(out.with_suffix(".run.json").read_text())
    assert run["base_model"]["embed_dim"] == 192 and run["tiles"] == 4 and run["windows"]["mode"] == "grid"


def test_segment_cells_baseline_then_resume(dataset, tmp_path):
    out = tmp_path / "predictions"
    arguments = [
        "--baseline", "stain", "--base-model", TINY_VIT, "--random-init", "--device", "cpu",
        "--images", str(dataset / "images"), "--mpp", "0.25", "--out", str(out),
    ]
    assert main_segment([*arguments, "--limit", "2"]) == 0
    assert len(list((out / "by_tile").glob("*.geojson"))) == 2
    assert main_segment([*arguments, "--skip-existing"]) == 0  # the remaining four
    assert len(list((out / "by_tile").glob("*.geojson"))) == 6
    cells = pd.read_parquet(out / "cells")
    assert cells["tile_id"].nunique() == 6 and (cells["method"] == "stain-baseline").all()
    assert cells["x"].between(4096, 4096 + 6 * 256).all() and cells["y"].between(8192, 8448).all()
    stats = pd.read_json(out / "tile_stats.jsonl", lines=True)
    assert len(stats) == 6 and stats["n_cells"].between(12, 16).all()
    run = json.loads((out / "run.json").read_text())
    assert run["method"] == "stain-baseline" and run["tiles"] == 4 and run["failed_tiles"] == []
    with pytest.raises(SystemExit):
        main_segment(["--images", str(dataset / "images"), "--out", str(out)])  # neither --head nor --baseline


def test_train_a_head_from_annotations_and_segment_with_it(dataset, tmp_path):
    """The whole loop on GeoJSON annotations: train, then predict with the head alone."""
    run_dir = tmp_path / "models" / "round_00" / "run_001"
    code = main_train([
        "--base-model", TINY_VIT, "--random-init", "--device", "cpu",
        "--images", str(dataset / "images"), "--annotations", str(dataset / "annotations"),
        "--labels", str(dataset / "labels.yaml"), "--mpp", "0.25", "--val-fraction", "0.34",
        "--base-width", "16", "--epochs", "2", "--steps-per-epoch", "12", "--batch-size", "4",
        "--crop-size", "128", "--lr", "2e-3", "--max-size", "256", "--out", str(run_dir),
    ])
    assert code == 0
    run = json.loads((run_dir / "run.json").read_text())
    assert len(run["train_tiles"]) == 4 and len(run["val_tiles"]) == 2 and "val_pq" in run["best"]
    assert run["labels"]["classes"][0]["name"] == "megakaryocyte"
    from mpn_al.segmenter import read_head

    assert read_head(run_dir / "head.pt")["extra"]["training"]["max_size"] == 256

    out = tmp_path / "predictions"
    # No --base-model: the head checkpoint names its encoder. (Random weights here are rebuilt
    # from the timm architecture, so the encoder differs from training; only the plumbing is tested.)
    code = main_segment([
        "--head", str(run_dir / "head.pt"), "--random-init", "--device", "cpu", "--images", str(dataset / "images"),
        "--mpp", "0.25", "--out", str(out), "--limit", "2", "--object-type", "detection", "--shard-index", "1",
        "--num-shards", "2",
    ])
    assert code == 0
    assert (out / "run.shard-001-of-002.json").exists() and (out / "tile_stats.shard-001-of-002.jsonl").exists()
    run = json.loads((out / "run.shard-001-of-002.json").read_text())
    assert run["method"] == "head" and len(run["head"]["sha256"]) == 64 and run["tiles"] == 2
    geojson = sorted((out / "by_tile").glob("*.geojson"))
    assert [p.stem for p in geojson] == ["tl_S1_x4352_y8192_s256", "tl_S1_x4864_y8192_s256"]  # shard 1 of 2
    for feature in json.loads(geojson[0].read_text())["features"]:
        assert feature["properties"]["objectType"] == "detection"
        assert feature["properties"]["classification"]["name"] == "megakaryocyte"

    warm = tmp_path / "models" / "round_01" / "run_001"
    assert main_train([
        "--base-model", TINY_VIT, "--random-init", "--device", "cpu", "--images", str(dataset / "images"),
        "--masks", str(dataset / "masks"), "--base-width", "16", "--epochs", "1", "--steps-per-epoch", "2",
        "--batch-size", "2", "--crop-size", "128", "--init-head", str(run_dir / "head.pt"), "--out", str(warm),
    ]) == 0
    assert (warm / "head.pt").exists()


def test_scripts_run_from_a_plain_checkout():
    """`python scripts/<name>.py` must work without installing the package."""
    for script in ("list_base_models.py", "extract_embeddings.py", "segment_cells.py", "train_segmentation_head.py"):
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / script), "--help"], capture_output=True, text=True, timeout=300
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
