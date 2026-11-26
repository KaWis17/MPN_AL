"""Figures of the clustering: helper functions, then both scripts end to end on synthetic embedding tables."""

import json
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from test_clustering import planted

from mpn_al.evaluation import cluster_figures as cf
from mpn_al.evaluation.cluster_cli import main as cluster_main
from mpn_al.evaluation.cluster_viz_cli import main as viz_main
from mpn_al.evaluation.cluster_viz_cli import parse_variant


def test_variant_names_are_parsed():
    assert parse_variant("center+l2+pca50") == (True, 50)
    assert parse_variant("l2") == (False, None)
    assert parse_variant("l2+pca128") == (False, 128)


def test_pivot_pairs_builds_a_symmetric_matrix():
    table = pd.DataFrame({"variant": ["l2", "l2", "l2", "x"], "k": [3, 3, 3, 3], "model_a": ["a", "a", "b", "a"],
                          "model_b": ["b", "c", "c", "b"], "ari": [0.9, 0.2, 0.4, 0.0]})
    matrix = cf.pivot_pairs(table, "ari", variant="l2", k=3)
    assert matrix.loc["a", "b"] == matrix.loc["b", "a"] == 0.9 and matrix.loc["c", "a"] == 0.2 and (np.diag(matrix) == 1).all()


def test_projection_is_deterministic_and_falls_back_when_umap_fails(monkeypatch):
    x = np.random.default_rng(0).normal(size=(40, 8))
    first, method = cf.project_2d(x, method="tsne")
    assert method == "t-SNE" and first.shape == (40, 2) and np.allclose(first, cf.project_2d(x, method="tsne")[0])

    class Broken:
        @staticmethod
        def UMAP(*args, **kwargs):
            raise TypeError("incompatible build")

    monkeypatch.setitem(sys.modules, "umap", Broken)
    assert cf.project_2d(x, method="auto")[1] == "t-SNE"  # a broken UMAP must not stop the figures
    with pytest.raises(TypeError):
        cf.project_2d(x, method="umap")  # but asking for it explicitly reports the problem


def test_cluster_members_and_gallery():
    vectors = np.concatenate([np.random.default_rng(1).normal(size=(10, 6)) + 5, np.random.default_rng(2).normal(size=(10, 6)) - 5])
    clusters = np.array([0] * 10 + [1] * 10)
    members = cf.cluster_members(vectors, clusters, near=3, random=2)
    assert set(members) == {0, 1} and all(len(m["nearest"]) == 3 and len(m["random"]) == 2 for m in members.values())
    assert members[0]["size"] == members[1]["size"] == 10
    assert all(i < 10 for i in members[0]["nearest"] + members[0]["random"]) and not set(members[0]["nearest"]) & set(members[0]["random"])
    unit = vectors[:10] / np.linalg.norm(vectors[:10], axis=1, keepdims=True)
    centre = vectors[:10].mean(axis=0) / np.linalg.norm(vectors[:10].mean(axis=0))
    assert members[0]["nearest"] == np.argsort(-(unit @ centre))[:3].tolist()  # nearest first, by cosine to the cluster centre
    montage = cf.render_gallery(members, lambda i: np.full((50, 80, 3), 10 * i % 255, np.uint8), thumb=32, title="t")
    assert montage.shape == (26 + 2 * 32, 70 + 5 * 32, 3)


def test_geojson_rectangles_with_cluster_classes(tmp_path):
    table = pd.DataFrame({"tile_id": ["tl_S1_x1024_y2048_s1024", "S1_x0_y0_s512", "other_image"], "cluster": [2, 0, 1]})
    written = cf.clusters_to_geojson(table, tmp_path, prefix="m_cluster")
    assert [p.name for p in written] == ["S1.geojson"]  # the tile whose id carries no position is skipped
    features = json.loads(written[0].read_text())["features"]
    assert len(features) == 2
    ring = features[0]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1] == [1024.0, 2048.0] and [2048.0, 3072.0] in ring
    assert {f["properties"]["classification"]["name"] for f in features} == {"m_cluster_2", "m_cluster_0"}


@pytest.fixture()
def clustering(tmp_path):
    vectors, truth, owner = planted(patients=6, per=12, classes=3, dim=24, noise=0.3)
    ids = [f"S_x{i * 256}_y0_s256" for i in range(len(vectors))]
    rng = np.random.default_rng(1)
    paths = {}
    for name, projected in (("A", vectors), ("B", vectors @ rng.normal(size=(24, 40)))):
        table = pd.DataFrame({"tile_id": ids, "base_model": name, "pooling": "cls", "window_mode": "grid",
                              "embedding": [row.astype(np.float32) for row in projected]})
        paths[name] = tmp_path / f"{name}.parquet"
        table.to_parquet(paths[name])
    meta = pd.DataFrame({"tile_id": ids, "patient_id": owner, "cls": truth, "source": np.where(np.arange(len(ids)) % 2, "x", "y")})
    meta.to_csv(tmp_path / "meta.csv", index=False)
    (tmp_path / "images").mkdir()
    for tile_id in ids:
        Image.fromarray(rng.integers(0, 255, (40, 60, 3), dtype=np.uint8)).save(tmp_path / "images" / f"{tile_id}.png")
    common = ["--meta", str(tmp_path / "meta.csv"), "--nuisance-col", "source", "--label-col", "cls"]
    embeddings = [x for n in paths for x in ("--embeddings", f"{n}={paths[n]}")]
    assert cluster_main([*embeddings, *common, "--ks", "2", "3", "4", "--seeds", "3", "--n-boot", "3", "--pca", "0", "10",
                         "--noise-control", "--out", str(tmp_path / "cl")]) == 0
    return tmp_path, embeddings, common


def test_script_draws_every_figure_and_an_index(clustering):
    root, embeddings, common = clustering
    assert viz_main(["--cluster-dir", str(root / "cl"), *embeddings, *common, "--images", str(root / "images"),
                     "--k", "3", "--variant", "l2", "--projection", "tsne", "--members", "2"]) == 0
    figures = root / "cl" / "figures"
    names = {p.name for p in figures.iterdir()}
    for expected in ("curves_l2.png", "curves_center+l2.png", "heatmap_silhouette_l2.png", "heatmap_ari_seed_mean_l2.png",
                     "matrix_ari.png", "matrix_cka.png", "matrix_knn10_overlap.png", "map_A.png", "map_B.png",
                     "composition_A.png", "gallery_A.png", "gallery_B.png", "index.html", "qupath"):
        assert expected in names, expected
    assert "map_noise-control.png" not in names  # the control has no embedding table to draw
    assert len(list((figures / "qupath" / "A_k3").glob("*.geojson"))) == 1
    page = (figures / "index.html").read_text(encoding="utf-8")
    assert "map_A.png" in page and "stable clustering can still be a batch effect" in page


def test_script_rejects_a_variant_or_k_that_was_not_clustered(clustering):
    root, embeddings, common = clustering
    with pytest.raises(SystemExit):
        viz_main(["--cluster-dir", str(root / "cl"), *embeddings, "--variant", "nope"])
    with pytest.raises(SystemExit):
        viz_main(["--cluster-dir", str(root / "cl"), *embeddings, "--k", "7"])
