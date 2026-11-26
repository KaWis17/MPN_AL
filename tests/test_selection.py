"""Tile selection: geometry, selections, scores, descriptors and the whole experiment on a small synthetic slide."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mpn_al.evaluation import selection as sel
from mpn_al.evaluation.clustering import fit_kmeans
from mpn_al.evaluation.selection_cli import main, rank_models, summarize
from mpn_al.evaluation.tile_descriptors import FEATURES, describe_tile, slide_background, standardize

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def grid(columns, rows, size=100, stride=50, x0=0, y0=0):
    return [(x0 + c * stride, y0 + r * stride) for r in range(rows) for c in range(columns)]


def table_for(positions, size=100):
    return pd.DataFrame({"tile_id": [f"S_x{x}_y{y}_s{size}" for x, y in positions], "slide_id": "S", "x": [p[0] for p in positions],
                         "y": [p[1] for p in positions], "width": size, "height": size, "mpp": 0.5})


def test_manifest_lattice_and_regions():
    positions = grid(5, 5) + grid(3, 3, x0=2000)  # two separate pieces of tissue
    manifest = sel.tile_manifest(table_for(positions))
    assert manifest["tile_id"].is_monotonic_increasing
    assert manifest["lattice"].sum() == 9 + 4  # every second column and row of each piece
    assert manifest["region"].nunique() == 2
    sizes = manifest["region"].value_counts()
    assert sizes[0] == 25 and sizes[1] == 9  # region 0 is the largest
    xy = manifest[["x", "y"]].to_numpy(float)
    conflicts = sel.overlap_matrix(xy[manifest["lattice"]], 100)
    assert conflicts.sum() == manifest["lattice"].sum()  # lattice tiles only overlap themselves


def test_overlap_is_shared_area_not_touching():
    xy = np.array([[0, 0], [50, 0], [100, 0], [100, 100]], float)
    assert sel.overlaps(xy, 0, [1, 2, 3], 100).tolist() == [True, False, False]
    assert sel.overlap_matrix(xy, 100)[1].tolist() == [True, True, True, False]


def planted_slide(seed=0):
    """An 12x12 slide with half-overlapping tiles and 4 well separated embedding groups in blocks."""
    rng = np.random.default_rng(seed)
    positions = np.array(grid(12, 12), float)
    groups = (positions[:, 0] >= 300).astype(int) + 2 * (positions[:, 1] >= 300).astype(int)
    vectors = np.eye(8)[groups] * 3 + 0.2 * rng.normal(size=(len(positions), 8))
    return positions, groups, vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_representatives_take_one_central_tile_per_cluster_without_overlap():
    positions, groups, vectors = planted_slide()
    model = fit_kmeans(vectors, 4, 0, n_init=5)
    conflicts = sel.overlap_matrix(positions, 100)
    chosen, fallbacks = sel.representatives(vectors, model.labels_, model.cluster_centers_, conflicts=conflicts)
    assert len(chosen) == 4 and fallbacks == 0 and sorted(groups[chosen]) == [0, 1, 2, 3]
    assert conflicts[np.ix_(chosen, chosen)].sum() == 4
    free, _ = sel.representatives(vectors, model.labels_, model.cluster_centers_)
    for index in free:  # without constraint it is exactly the most cosine-similar member
        cluster = model.labels_[index]
        members = np.flatnonzero(model.labels_ == cluster)
        centre = model.cluster_centers_[cluster] / np.linalg.norm(model.cluster_centers_[cluster])
        assert index == members[np.argmax(vectors[members] @ centre)]
    many = fit_kmeans(vectors, 30, 0)
    picked, _ = sel.representatives(vectors, many.labels_, many.cluster_centers_, conflicts=conflicts)
    assert len(picked) == len(set(picked.tolist())) and conflicts[np.ix_(picked, picked)].sum() == len(picked)


def test_a_fully_blocked_cluster_falls_back_to_a_free_tile():
    vectors = np.array([[1, 0], [1, 0.1], [0, 1], [0.7, 0.7]], float)
    labels = np.array([0, 0, 1, 2])
    centres = np.array([[1, 0.05], [0, 1], [0.7, 0.7]])
    conflicts = np.eye(4, dtype=bool)
    conflicts[:3, :3] = True  # tiles 0, 1 and 2 all overlap
    chosen, fallbacks = sel.representatives(vectors, labels, centres, conflicts=conflicts)
    assert fallbacks >= 1 and len(set(chosen.tolist())) == len(chosen)


def test_baselines_respect_the_constraint_and_spatial_spreads_more_than_random():
    positions = np.array(grid(30, 30), float)
    conflicts = sel.overlap_matrix(positions, 100)
    rng = np.random.default_rng(0)
    random = sel.random_selection(len(positions), 20, rng, conflicts)
    spatial = sel.spatial_selection(positions, 20, rng, conflicts)
    for chosen in (random, spatial):
        assert len(chosen) == 20 and conflicts[np.ix_(chosen, chosen)].sum() == 20

    def spread(indices):
        apart = np.sqrt(((positions[indices][:, None] - positions[indices][None]) ** 2).sum(-1))
        np.fill_diagonal(apart, np.inf)
        return apart.min(axis=1).mean()

    assert spread(spatial) > spread(random) * 1.3
    assert len(sel.random_selection(len(positions), 10_000, rng, conflicts)) < len(positions)  # the slide runs out


def test_scores_reward_a_selection_that_covers_every_kind_of_tile():
    positions, groups, _ = planted_slide()
    manifest = sel.tile_manifest(table_for([tuple(map(int, p)) for p in positions]))
    order_ids = [f"S_x{int(x)}_y{int(y)}_s100" for x, y in positions]
    position_of = {tile: i for i, tile in enumerate(order_ids)}
    index = np.array([position_of[t] for t in manifest["tile_id"]])
    groups, positions = groups[index], positions[index]
    descriptors = np.eye(4)[groups] * 4
    flags = (groups == 3)[:, None]
    scorer = sel.Scorer(manifest, descriptors, flags, groups, mpp=0.5, cell=300)
    good = [int(np.flatnonzero(groups == g)[0]) for g in range(4)]
    poor = np.flatnonzero(groups == 0)[:4]
    a, b = scorer.score(good), scorer.score(poor)
    assert a["coverage_mean"] == pytest.approx(0, abs=1e-6) and b["coverage_mean"] > 1
    assert a["rare_categories"] == 1 and b["rare_categories"] == 0
    assert a["jsd_groups"] < b["jsd_groups"] and a["diversity_mean"] > b["diversity_mean"]
    assert a["grid_cells"] == 4 and b["grid_cells"] == 1 and a["overlapping_pairs"] == 0


def test_divergence_agreement_and_neighbours():
    assert sel.jensen_shannon(np.array([1.0, 1]), np.array([1.0, 1])) == pytest.approx(0)
    assert sel.jensen_shannon(np.array([1.0, 0]), np.array([0.0, 1])) == pytest.approx(1)
    positions = np.array([[0, 0], [50, 0], [400, 0], [800, 0]], float)
    result = sel.agreement([0, 2], [1, 3], positions, 100)
    assert result["identical"] == 0 and result["identical_or_overlapping"] == 1 and result["matched_share"] == 0.5
    assert sel.agreement([0, 2], [2, 0], positions, 100)["identical_share"] == 1.0
    positions, groups, vectors = planted_slide()
    conflicts = sel.overlap_matrix(positions, 100)
    neighbours = sel.cosine_neighbours(vectors, conflicts, 5)
    assert not conflicts[np.arange(len(vectors))[:, None], neighbours].any()
    descriptors = np.eye(4)[groups] * 4
    assert sel.neighbour_consistency(vectors, descriptors, conflicts, 5) < 0.2 * sel.chance_consistency(descriptors)
    assert sel.neighbour_overlap(vectors, vectors, conflicts, 5) == 1.0
    flags = sel.rare_flags(pd.DataFrame({"a": [0, 0, 0, 0, 0, 0, 0, 0, 0, 9.0], "b": np.zeros(10)}), ["a", "b"], share=0.1)
    assert flags[:, 0].sum() == 1 and flags[:, 1].sum() == 0


def synthetic_tile(kind, rng, size=256):
    """Four recognisably different tiles: fat, cellular, bone-like and red cells."""
    image = np.clip(rng.normal((245, 242, 238), 2, (size, size, 3)), 0, 255)
    yy, xx = np.mgrid[:size, :size]
    if kind == "fat":
        image = np.clip(rng.normal((232, 190, 205), 3, (size, size, 3)), 0, 255)
        for cy in range(40, size, 80):
            for cx in range(40, size, 80):
                image[(yy - cy) ** 2 + (xx - cx) ** 2 <= 34**2] = (245, 242, 238)
    elif kind == "cells":
        image = np.clip(rng.normal((225, 170, 195), 3, (size, size, 3)), 0, 255)
        for _ in range(120):
            cy, cx = rng.integers(8, size - 8, 2)
            image[(yy - cy) ** 2 + (xx - cx) ** 2 <= 49] = (75, 40, 110)
    elif kind == "bone":
        image = np.clip(rng.normal((232, 130, 165), 2, (size, size, 3)), 0, 255)
    else:
        image = np.clip(rng.normal((235, 232, 230), 2, (size, size, 3)), 0, 255)
        for _ in range(400):
            cy, cx = rng.integers(6, size - 6, 2)
            image[(yy - cy) ** 2 + (xx - cx) ** 2 <= 25] = (190, 70, 60)
    return image.astype(np.uint8)


def test_descriptors_tell_tissue_types_apart():
    rng = np.random.default_rng(0)
    tiles = {kind: synthetic_tile(kind, rng) for kind in ("fat", "cells", "bone", "red")}
    background = slide_background([tiles["fat"], tiles["red"]])
    d = {kind: describe_tile(image, mpp=0.44, background=background) for kind, image in tiles.items()}
    assert set(d["fat"]) == set(FEATURES)
    assert d["fat"]["fat"] > 0.3 > d["bone"]["fat"] and d["fat"]["white"] > d["bone"]["white"] + 0.3
    assert d["cells"]["nuclear_density"] > 5 * max(d["bone"]["nuclear_density"], 1)
    assert d["bone"]["compact_eosin"] > 0.7 > d["cells"]["compact_eosin"]
    assert d["red"]["erythrocyte"] > 0.2 > 5 * d["fat"]["erythrocyte"]
    scaled = standardize(pd.DataFrame(d).T.reset_index(drop=True))
    assert scaled.shape == (4, len(FEATURES)) and np.isfinite(scaled).all() and np.abs(scaled).max() <= 5


def test_ranking_rule_prefers_the_cheaper_model_on_a_tie():
    rows = []
    for method, centre in (("midnight-12k", 1.000), ("dinov2-s14", 1.004), ("phikon-v2", 1.5)):
        for k in (50, 80):
            for seed in range(5):
                rows.append({"variant": "l2", "method": method, "k": k, "seed": seed, "constraint": "no-overlap",
                             "coverage_mean": centre + 0.02 * (seed - 2)})
    ranking = rank_models(pd.DataFrame(rows), [50, 80], ["midnight-12k", "dinov2-s14", "phikon-v2"])
    assert ranking["method"].tolist() == ["dinov2-s14", "midnight-12k", "phikon-v2"]
    assert ranking.loc[0, "tied_with"] == "midnight-12k" and ranking.loc[2, "tied_with"] == ""


def test_summary_reports_percentiles_against_random():
    rows = [{"variant": "baseline", "method": "random", "k": 5, "seed": i, "constraint": "no-overlap", "coverage_mean": 1 + i / 10,
             "coverage_p95": 2.0, "rare_categories": float(i % 3)} for i in range(10)]
    rows.append({"variant": "l2", "method": "m", "k": 5, "seed": 0, "constraint": "no-overlap", "coverage_mean": 0.5,
                 "coverage_p95": 2.0, "rare_categories": 5.0})
    table = summarize(pd.DataFrame(rows)).set_index("method")
    assert table.loc["m", "coverage_mean_beats_random_pct"] == 100 and table.loc["m", "rare_categories_beats_random_pct"] == 100
    assert table.loc["m", "coverage_p95_beats_random_pct"] == 50  # all equal


def test_the_whole_experiment_on_a_synthetic_slide(tmp_path):
    rng = np.random.default_rng(0)
    positions = grid(10, 10, size=256, stride=128) + grid(4, 4, size=256, stride=128, x0=4096)
    table = table_for(positions, size=256)
    table["mpp"] = 0.44
    table.to_parquet(tmp_path / "tiles.parquet")
    kinds = ["fat", "cells", "bone", "red"]
    images = tmp_path / "tiles"
    images.mkdir()
    truth = []
    for row in table.itertuples():
        kind = int((row.x // 640 + row.y // 640) % 4)
        truth.append(kind)
        Image.fromarray(synthetic_tile(kinds[kind], rng)).save(images / f"{row.tile_id}.png")
    good = np.eye(16)[truth] * 3 + 0.2 * rng.normal(size=(len(truth), 16))  # a model whose space follows the tissue
    useless = rng.normal(size=(len(truth), 24))
    arguments = ["--tiles-parquet", str(tmp_path / "tiles.parquet"), "--images", str(images)]
    for name, vectors in (("dinov2-s14", useless), ("phikon-v2", good)):
        path = tmp_path / f"{name}.parquet"
        pd.DataFrame({"tile_id": table["tile_id"], "base_model": name, "pooling": "cls", "window_mode": "grid",
                      "embedding": [v.astype(np.float32) for v in vectors]}).iloc[::-1].to_parquet(path)
        arguments += ["--embeddings", f"{name}={path}"]
    sys.path.insert(0, str(SCRIPTS))
    import run_tile_selection

    assert run_tile_selection.main is main
    out = tmp_path / "out"
    assert main([*arguments, "--out", str(out), "--ks", "4", "8", "--main-ks", "4", "8", "--seeds", "3", "--random-draws", "40",
                 "--spatial-draws", "5", "--pca", "8", "--top-n-init", "2", "--board-k", "8", "--workers", "1",
                 "--projection", "tsne", "--knn", "5"]) == 0

    ranking = pd.read_csv(out / "ranking.csv")
    assert ranking["method"].tolist()[0] == "phikon-v2"  # the space that follows the tissue covers its diversity best
    summary = pd.read_csv(out / "summary.csv").set_index(["variant", "method", "k", "constraint"])
    assert summary.loc[("l2", "phikon-v2", 4, "no-overlap"), "coverage_mean"] < summary.loc[("baseline", "random", 4, "no-overlap"), "coverage_mean"]
    assert summary.loc[("l2", "phikon-v2", 8, "no-overlap"), "overlapping_pairs"] == 0
    scores = pd.read_csv(out / "selection_scores.csv")
    assert set(scores["variant"]) == {"l2", "lattice", "pca8", "l2-ninit2", "baseline"}
    assert set(scores["method"]) >= {"random", "spatial", "descriptor-kmeans", "noise-control"}
    for name in ("clustering_metrics.csv", "selections.parquet", "seed_stability.csv", "models_agreement.csv", "neighbour_quality.csv",
                 "board_comparison.csv", "descriptors.parquet", "report.md", "run.json", "blind_key.json"):
        assert (out / name).exists(), name
    figures = {p.name for p in (out / "figures").iterdir()}
    assert {"curves_clustering.png", "curves_selection.png", "heatmap_silhouette.png", "heatmap_coverage_mean.png", "matrix_ari.png",
            "matrix_knn_overlap.png", "map_phikon-v2.png", "board_random_K8.png", "board_phikon-v2_K8.png", "slide_map_K8.png",
            "blind_A.png", "blind_B.png", "blind_C.png", "index.html"} <= figures
    assert sorted(json.loads((out / "blind_key.json").read_text()).values()) == ["dinov2-s14", "phikon-v2", "random"]
    quality = pd.read_csv(out / "neighbour_quality.csv").set_index("model")
    assert quality.loc["phikon-v2", "neighbour_distance"] < quality.loc["dinov2-s14", "neighbour_distance"]
    assert "Ranking" in (out / "report.md").read_text(encoding="utf-8")

    again = tmp_path / "again"  # cached descriptors give the same scores
    assert main([*arguments, "--out", str(again), "--descriptors", str(out / "descriptors.parquet"), "--ks", "4", "--main-ks", "4",
                 "--seeds", "3", "--random-draws", "40", "--spatial-draws", "5", "--pca", "0", "--top-n-init", "0", "--board-k", "4",
                 "--workers", "1", "--projection", "tsne", "--knn", "5"]) == 0
    first = pd.read_csv(out / "selection_scores.csv").query("variant == 'l2' and k == 4")
    second = pd.read_csv(again / "selection_scores.csv").query("variant == 'l2' and k == 4")
    assert np.allclose(first["coverage_mean"].to_numpy(), second["coverage_mean"].to_numpy())


def test_missing_embeddings_are_reported(tmp_path):
    table = table_for(grid(3, 3))
    table.to_parquet(tmp_path / "tiles.parquet")
    (tmp_path / "t").mkdir()
    for tile in table["tile_id"]:
        Image.fromarray(np.full((100, 100, 3), 200, np.uint8)).save(tmp_path / "t" / f"{tile}.png")
    pd.DataFrame({"tile_id": table["tile_id"][:-1], "base_model": "m", "pooling": "cls", "window_mode": "grid",
                  "embedding": [np.ones(4, np.float32)] * 8}).to_parquet(tmp_path / "e.parquet")
    with pytest.raises(ValueError, match="no embedding for 1 tiles"):
        main(["--tiles-parquet", str(tmp_path / "tiles.parquet"), "--images", str(tmp_path / "t"), "--embeddings",
              f"m={tmp_path / 'e.parquet'}", "--out", str(tmp_path / "o")])
