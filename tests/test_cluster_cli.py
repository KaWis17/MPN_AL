"""scripts/cluster_embeddings.py end to end on synthetic embedding tables of two 'models' with different dimensions."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_clustering import planted

from mpn_al.evaluation.cluster_cli import aligned_spaces, load_embeddings, main, make_variants

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def write_table(path, tile_ids, vectors, model, **extra):
    table = pd.DataFrame({"tile_id": tile_ids, "base_model": model, "pooling": "cls", "window_mode": "grid",
                          "embedding": [row.astype(np.float32) for row in vectors], **extra})
    table.to_parquet(path)
    return path


@pytest.fixture()
def data(tmp_path):
    vectors, truth, owner = planted(patients=6, per=12, classes=3, dim=24, noise=0.3)
    ids = [f"t{i:03d}" for i in range(len(vectors))]
    rng = np.random.default_rng(1)
    projection = rng.normal(size=(24, 40))  # a second 'model': another space of another dimension
    a = write_table(tmp_path / "a.parquet", ids, vectors, "model-a")
    b = write_table(tmp_path / "b.parquet", ids[:-4][::-1], (vectors @ projection)[:-4][::-1], "model-b")  # other order, 4 tiles fewer
    meta = pd.DataFrame({"tile_id": ids, "patient_id": owner, "cls": truth, "source": np.where(np.arange(len(ids)) % 2, "x", "y")})
    meta.to_csv(tmp_path / "meta.csv", index=False)
    return tmp_path, len(ids)


def run(root, *extra):
    return main(["--embeddings", f"A={root / 'a.parquet'}", "--embeddings", f"B={root / 'b.parquet'}", "--meta", str(root / "meta.csv"),
                 "--ks", "2", "3", "4", "--seeds", "3", "--n-boot", "4", "--pca", "0", "10", "--out", str(root / "out"), *extra])


def test_script_writes_every_table_and_finds_the_planted_structure(data):
    root, n = data
    assert run(root, "--label-col", "cls", "--nuisance-col", "source", "--noise-control") == 0
    out = root / "out"
    metrics = pd.read_csv(out / "metrics.csv")
    assert set(metrics["model"]) == {"A", "B", "noise-control"} and set(metrics["variant"]) == {
        "l2", "l2+pca10", "center+l2", "center+l2+pca10"}
    assert len(metrics) == 3 * 4 * 3 and {"nmi_source", "nmi_label", "nmi_nuisance", "ari_heldout_mean"} <= set(metrics.columns)
    best = pd.read_csv(out / "best_k.csv").set_index(["model", "variant"])
    assert best.loc[("A", "l2"), "k"] == 3 and best.loc[("B", "l2"), "k"] == 3  # both spaces carry the 3 planted groups
    assert best.loc[("noise-control", "l2"), "ari_seed_mean"] < best.loc[("A", "l2"), "ari_seed_mean"] - 0.3
    run_info = json.loads((out / "run.json").read_text())
    assert run_info["samples"] == n - 4 and run_info["dropped_per_model"] == {"A": 4, "B": 0}  # only tiles present in every model
    clusters = pd.read_parquet(out / "clusters.parquet")
    assert set(clusters["k"]) == {2, 3, 4} and clusters["cluster"].max() <= 3
    models = pd.read_csv(out / "compare_models.csv")
    row = models[(models.variant == "l2") & (models.k == 3) & (models.model_a == "A") & (models.model_b == "B")].iloc[0]
    assert row["ari"] > 0.9  # the same tiles, the same groups, whatever the dimension
    spaces = pd.read_csv(out / "compare_spaces.csv")
    assert {"cka", "knn10_overlap"} <= set(spaces.columns) and spaces["cka"].between(0, 1).all()
    same = spaces[(spaces.variant == "l2") & (spaces.model_a == "A") & (spaces.model_b == "B")].iloc[0]
    noise = spaces[(spaces.variant == "l2") & (spaces.model_a == "A") & (spaces.model_b == "noise-control")].iloc[0]
    assert same["knn10_overlap"] > noise["knn10_overlap"] and same["cka"] > noise["cka"]  # a projection of A is closer to A than noise
    assert "Best k by silhouette" in (out / "report.md").read_text(encoding="utf-8")


def test_row_order_of_the_inputs_does_not_matter(data):
    root, _ = data
    tables = {"A": load_embeddings(root / "a.parquet"), "B": load_embeddings(root / "b.parquet")}
    samples, spaces = aligned_spaces(tables, pd.read_csv(root / "meta.csv"))
    ids = samples["tile_id"].tolist()
    assert ids == sorted(ids) and spaces["A"].shape == (len(ids), 24) and spaces["B"].shape == (len(ids), 40)
    original = np.stack(tables["A"].set_index("tile_id").loc[ids]["embedding"].to_numpy())
    assert np.allclose(spaces["A"], original)


def test_without_patients_the_patient_variants_and_bootstrap_are_skipped(data, caplog):
    root, _ = data
    meta = pd.read_csv(root / "meta.csv").drop(columns=["patient_id"])
    meta.to_csv(root / "meta.csv", index=False)
    assert run(root) == 0
    metrics = pd.read_csv(root / "out" / "metrics.csv")
    assert set(metrics["variant"]) == {"l2", "l2+pca10"} and "ari_heldout_mean" not in metrics.columns


def test_bad_inputs_are_reported(data, tmp_path):
    root, _ = data
    with pytest.raises(SystemExit):  # two models with the same name
        main(["--embeddings", f"A={root / 'a.parquet'}", "--embeddings", f"A={root / 'b.parquet'}", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit):  # a column that does not exist
        run(root, "--label-col", "nope")
    table = pd.read_parquet(root / "a.parquet")
    pd.concat([table, table.iloc[:1]]).to_parquet(root / "dup.parquet")
    with pytest.raises(ValueError, match="several rows per tile_id"):
        load_embeddings(root / "dup.parquet")
    with pytest.raises(FileNotFoundError):
        load_embeddings(root / "missing.parquet")


def test_variants():
    assert [v[0] for v in make_variants(["none", "patient"], [0, 50], True)] == ["l2", "l2+pca50", "center+l2", "center+l2+pca50"]
    assert [v[0] for v in make_variants(["none", "patient"], [0], False)] == ["l2"]


def test_script_entry_point_exists():
    sys.path.insert(0, str(SCRIPTS))
    import cluster_embeddings

    assert callable(cluster_embeddings.main)
