"""Tools for running on a cluster: job-array shards, merging, weight download, demo data."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import TINY_VIT

from mpn_al.embeddings import resolve_spec
from mpn_al.embeddings.cli import main_extract
from mpn_al.embeddings.download import download_model, hub_repo
from mpn_al.embeddings.merge import find_shards, merge_embedding_shards, merge_tile_stats
from mpn_al.embeddings.merge import main as main_merge
from mpn_al.embeddings.tables import shard_path
from mpn_al.evaluation import panoptic_quality
from mpn_al.image_backend import TileRef, parse_tile_id
from mpn_al.segmenter import LabelSchema, read_features, read_tile_stats
from mpn_al.segmenter.cli import main as main_segment
from mpn_al.training import rasterize_annotations
from mpn_al.training.synthetic import write_demo_dataset


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo")
    return root, write_demo_dataset(root, tiles=5, size=256, count=10)


def extract(root, out, *extra):
    torch.manual_seed(0)  # --random-init builds a new random encoder per call; keep the calls comparable
    return main_extract([
        "--base-model", TINY_VIT, "--random-init", "--device", "cpu", "--images", str(root / "images"),
        "--mpp", "0.5", "--out", str(out), "--batch-size", "4", *extra,
    ])


def test_shard_paths():
    assert shard_path("a/embeddings.parquet", 3, 8).as_posix() == "a/embeddings.shard-003-of-008.parquet"
    assert shard_path("a/embeddings.parquet", 0, 1).as_posix() == "a/embeddings.parquet"


def test_sharded_extraction_merges_to_the_single_run_result(demo, tmp_path):
    root, _ = demo
    assert extract(root, tmp_path / "single.parquet") == 0
    out = tmp_path / "embeddings.parquet"
    for index in range(3):
        assert extract(root, out, "--shard-index", str(index), "--num-shards", "3") == 0
    shards = find_shards(out)
    assert [p.name for p in shards] == [f"embeddings.shard-00{i}-of-003.parquet" for i in range(3)]
    assert (tmp_path / "embeddings.shard-001-of-003.run.json").exists()

    assert merge_embedding_shards(out) == 5
    merged = pd.read_parquet(out).set_index("tile_id")
    single = pd.read_parquet(tmp_path / "single.parquet")
    assert set(merged.index) == set(single["tile_id"])
    assert np.allclose(
        np.stack(merged.loc[single["tile_id"], "embedding"].to_numpy()), np.stack(single["embedding"].to_numpy()), atol=1e-4
    )

    before = shards[1].stat().st_mtime_ns  # --skip-existing redoes nothing that is finished
    assert extract(root, out, "--shard-index", "1", "--num-shards", "3", "--skip-existing") == 0
    assert shards[1].stat().st_mtime_ns == before


def test_tasks_beyond_the_tile_count_exit_cleanly_and_the_gap_is_explained(demo, tmp_path):
    root, _ = demo
    out = tmp_path / "embeddings.parquet"
    for index in range(8):  # five tiles, eight tasks: tasks 5-7 have nothing to do
        assert extract(root, out, "--shard-index", str(index), "--num-shards", "8") == 0
    assert len(find_shards(out, allow_missing=True)) == 5
    with pytest.raises(FileNotFoundError, match=r"shards \[5, 6, 7\] of 8 are missing"):
        find_shards(out)
    assert main_merge(["embeddings", str(out)]) == 1
    assert main_merge(["embeddings", str(out), "--allow-missing"]) == 0
    assert len(pd.read_parquet(out)) == 5

    assert main_segment([
        "--baseline", "stain", "--images", str(root / "images"), "--mpp", "0.25", "--out", str(tmp_path / "pred"),
        "--num-shards", "8", "--shard-index", "6",
    ]) == 0
    assert not (tmp_path / "pred").exists()


def test_mixed_or_overlapping_shards_are_refused(demo, tmp_path):
    root, _ = demo
    out = tmp_path / "e.parquet"
    assert extract(root, out, "--shard-index", "0", "--num-shards", "2") == 0
    assert extract(root, out, "--shard-index", "0", "--num-shards", "3") == 0
    with pytest.raises(ValueError, match="different arrays"):
        find_shards(out)
    pd.read_parquet(tmp_path / "e.shard-000-of-002.parquet").to_parquet(tmp_path / "f.shard-000-of-002.parquet")
    pd.read_parquet(tmp_path / "e.shard-000-of-002.parquet").to_parquet(tmp_path / "f.shard-001-of-002.parquet")
    with pytest.raises(ValueError, match="duplicates"):
        merge_embedding_shards(tmp_path / "f.parquet")


def test_tile_stats_of_shards_are_read_and_merged(tmp_path):
    def lines(*records):
        return "".join(json.dumps(r) + "\n" for r in records)

    (tmp_path / "tile_stats.shard-000-of-002.jsonl").write_text(lines({"tile_id": "a", "n_cells": 1}, {"tile_id": "b", "n_cells": 2}))
    (tmp_path / "tile_stats.shard-001-of-002.jsonl").write_text(lines({"tile_id": "c", "n_cells": 3}, {"tile_id": "b", "n_cells": 5}))
    (tmp_path / "tile_stats.shard-002-of-003.jsonl").write_text("")  # an empty file must not break reading
    stats = read_tile_stats(tmp_path)
    assert dict(zip(stats["tile_id"], stats["n_cells"])) == {"a": 1, "b": 5, "c": 3}  # the redone tile keeps its last record
    assert merge_tile_stats(tmp_path) == 3
    assert len(read_tile_stats(tmp_path / "tile_stats.jsonl")) == 3
    (tmp_path / "tile_stats.jsonl").unlink()
    assert len(read_tile_stats(tmp_path / "tile_stats.jsonl")) == 3  # falls back to the shards beside it
    with pytest.raises(FileNotFoundError):
        read_tile_stats(tmp_path / "nowhere")


def test_repository_ids_follow_from_the_registry():
    expected = {
        "uni2-h": "MahmoodLab/UNI2-h",
        "h-optimus-0": "bioptimus/H-optimus-0",
        "virchow2": "paige-ai/Virchow2",
        "prov-gigapath": "prov-gigapath/prov-gigapath",  # registered with the 'hf_hub:' spelling used on its card
        "phikon-v2": "owkin/phikon-v2",
        "midnight-12k": "kaiko-ai/midnight",
        "lunit-dino-s16": "1aurent/vit_small_patch16_224.lunit_dino",
        "dinov2-s14": "timm/vit_small_patch14_dinov2.lvd142m",  # no hf-hub: prefix: taken from timm's pretrained config
    }
    for name, repo in expected.items():
        assert hub_repo(resolve_spec(name)) == repo, name


def test_download_prefers_safetensors_and_explains_gated_repositories(monkeypatch, tmp_path):
    import huggingface_hub

    seen = {}

    def list_files(self, repo_id, **kwargs):
        return ["config.json", "model.safetensors", "pytorch_model.bin"]

    def snapshot(repo_id, **kwargs):
        seen.update(repo=repo_id, **kwargs)
        folder = tmp_path / "snapshot"
        folder.mkdir(exist_ok=True)
        (folder / "model.safetensors").write_bytes(b"x" * 100)
        return str(folder)

    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", list_files)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    result = download_model("phikon-v2")
    assert (result.status, result.repo, result.size_bytes) == ("ok", "owkin/phikon-v2", 100)
    assert "*.bin" in seen["ignore_patterns"] and "*.msgpack" in seen["ignore_patterns"]  # no duplicate pickles

    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", lambda self, repo_id, **kw: ["pytorch_model.bin"])
    download_model("phikon-v2")
    assert "*.bin" not in seen["ignore_patterns"]  # without safetensors the pickles are all there is

    def gated(self, repo_id, **kwargs):
        raise RuntimeError("401 Client Error: GatedRepoError")

    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", gated)
    denied = download_model("uni2-h")
    assert denied.status == "denied" and not denied.ok
    assert "request access at https://huggingface.co/MahmoodLab/UNI2-h" in denied.detail
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", lambda self, repo_id, **kw: (_ for _ in ()).throw(ConnectionError("no network")))
    assert download_model("phikon-v2").status == "failed"
    assert download_model("timm:no_such_model_xyz").status == "failed"


def test_demo_dataset_annotations_match_its_masks(tmp_path):
    tile_ids = write_demo_dataset(tmp_path, tiles=2, size=256, count=10, mpp=0.5)
    assert json.loads((tmp_path / "dataset.json").read_text())["mpp"] == 0.5
    schema = LabelSchema.single_class("meg_norm")
    for tile_id in tile_ids:
        origin = parse_tile_id(tile_id)
        tile = TileRef(tile_id, x=origin["x"], y=origin["y"], width=256, height=256)
        instances, _, ignore = rasterize_annotations(read_features(tmp_path / "annotations" / f"{tile_id}.geojson"), tile, schema)
        truth = np.load(tmp_path / "masks" / f"{tile_id}.npz")["instances"]
        assert panoptic_quality(truth, instances)["dq"] == 1.0 and not ignore.any()
