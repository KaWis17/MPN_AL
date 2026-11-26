"""Cluster the embeddings of several base models and compare them: the command-line side of ``clustering.py``.

    python scripts/cluster_embeddings.py \
        --embeddings phikon-v2=runs/phikon/embeddings.parquet --embeddings lunit-dino-s16=runs/lunit/embeddings.parquet \
        --meta outputs/bambo_split.csv --out outputs/clustering/bambo --noise-control

Each ``--embeddings`` table has one row per tile (or cell) with an ``embedding`` column and ``tile_id`` (a merged file, or the
``.shard-*`` files of a job array next to a missing merged file). ``--meta`` (CSV or parquet with ``tile_id``) adds or
overrides columns such as ``patient_id`` and a class column. Only samples present in *every* model are used, in the same
order, so row ``i`` is the same tile everywhere.

Written to ``--out``:

    metrics.csv         one row per (model, variant, k): quality, stability, agreement with labels, confounding
    best_k.csv          per (model, variant) the k with the best silhouette, with its stability
    clusters.parquet    seed-0 cluster of every sample for every (model, variant, k)
    compare_models.csv  ARI/AMI between the models' clusterings (same variant, same k)
    compare_spaces.csv  linear CKA and kNN overlap between the models' spaces (same variant)
    report.md, run.json what was done, with what, and what the numbers cannot say

A *variant* is a way of preparing the embeddings: ``center`` subtracts each patient's mean vector, ``pca<d>`` reduces to
``d`` components; every variant is L2-normalised. ``--noise-control`` adds Gaussian noise vectors as a pseudo-model: chance level.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..embeddings.merge import find_shards
from .clustering import DEFAULT_KS, kmeans_labels, preprocess, run_grid
from .embedding_compare import knn_overlap, linear_cka, partition_agreement

logger = logging.getLogger("mpn_al.cluster")
NOISE = "noise-control"


def parse_named(text: str) -> tuple[str | None, Path]:
    name, sep, path = text.partition("=")
    return (name, Path(path)) if sep else (None, Path(text))


def load_embeddings(path: Path, *, window_mode: str | None = None, pooling: str | None = None) -> pd.DataFrame:
    """One model's table; a missing merged file is replaced by its shards."""
    if path.exists():
        table = pd.read_parquet(path)
    else:
        shards = find_shards(path)
        if not shards:
            raise FileNotFoundError(f"{path} does not exist and has no shard files")
        table = pd.concat([pd.read_parquet(shard) for shard in shards], ignore_index=True)
    for column, wanted in (("window_mode", window_mode), ("pooling", pooling)):
        if wanted is not None:
            table = table[table[column] == wanted]
    if table["tile_id"].duplicated().any():
        raise ValueError(f"{path}: several rows per tile_id; pick one with --window-mode / --pooling")
    return table.reset_index(drop=True)


def load_meta(path: Path) -> pd.DataFrame:
    table = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "tile_id" not in table.columns:
        raise ValueError(f"{path} has no tile_id column")
    return table.drop_duplicates("tile_id")


def aligned_spaces(tables: dict[str, pd.DataFrame], meta: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Samples present in every model, in sorted ``tile_id`` order: (sample table with the meta columns, vectors per model)."""
    common = sorted(set.intersection(*(set(t["tile_id"]) for t in tables.values())))
    if len(common) < 4:
        raise ValueError(f"only {len(common)} tiles are present in every model's table")
    spaces = {}
    for name, table in tables.items():
        ordered = table.set_index("tile_id").loc[common]
        spaces[name] = np.stack(ordered["embedding"].to_numpy()).astype(np.float64)
    samples = pd.DataFrame({"tile_id": common})
    if meta is not None:
        samples = samples.merge(meta, on="tile_id", how="left")
    else:
        first = next(iter(tables.values())).set_index("tile_id").loc[common].reset_index()
        samples = first.drop(columns=["embedding"])
    return samples, spaces


def make_variants(center: Sequence[str], pca: Sequence[int], has_patients: bool) -> list[tuple[str, bool, int | None]]:
    variants = []
    for centred in center:
        if centred == "patient" and not has_patients:
            logger.warning("no patient column: the 'patient' centring variant is skipped")
            continue
        for dim in pca:
            parts = (["center"] if centred == "patient" else []) + ["l2"] + ([f"pca{dim}"] if dim else [])
            variants.append(("+".join(parts), centred == "patient", dim or None))
    return variants


def markdown_table(table: pd.DataFrame, digits: int = 3) -> str:
    def cell(value) -> str:
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(map(str, table.columns)) + " |", "|" + "|".join("---" for _ in table.columns) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", action="append", required=True, type=parse_named, metavar="[NAME=]FILE",
                        help="embedding table of one model (repeat per model); NAME defaults to its base_model column")
    parser.add_argument("--meta", type=Path, help="CSV/parquet with tile_id and columns such as patient_id and a class")
    parser.add_argument("--patient-col", default="patient_id")
    parser.add_argument("--label-col", help="known classes, for agreement with the clusters")
    parser.add_argument("--nuisance-col", action="append", default=[], metavar="COL",
                        help="column that should NOT explain the clusters (slide_id, image size, scanner); repeatable")
    parser.add_argument("--window-mode", help="keep only rows with this window_mode (e.g. grid)")
    parser.add_argument("--pooling", help="keep only rows with this pooling (e.g. cls)")
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--seeds", type=int, default=10, help="number of KMeans seeds (0..N-1)")
    parser.add_argument("--n-boot", type=int, default=20, help="repetitions that hold out a share of the patients")
    parser.add_argument("--boot-fraction", type=float, default=0.8)
    parser.add_argument("--center", nargs="+", choices=["none", "patient"], default=["none", "patient"],
                        help="centring variants (patient = subtract each patient's mean vector)")
    parser.add_argument("--pca", type=int, nargs="+", default=[0, 50], help="PCA dimensions, 0 = the full space")
    parser.add_argument("--knn", type=int, default=10, help="neighbours for the overlap between spaces")
    parser.add_argument("--noise-control", action="store_true", help="add Gaussian noise as a pseudo-model (chance level)")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    started = time.perf_counter()

    tables: dict[str, pd.DataFrame] = {}
    for name, path in args.embeddings:
        table = load_embeddings(path, window_mode=args.window_mode, pooling=args.pooling)
        label = name or str(table["base_model"].iloc[0])
        if label in tables:
            parser.error(f"two models are called {label!r}; name them with NAME=FILE")
        tables[label] = table
    meta = load_meta(args.meta) if args.meta else None
    samples, spaces = aligned_spaces(tables, meta)
    dropped = {name: len(t) - len(samples) for name, t in tables.items()}
    logger.info("%d tiles common to all %d models (dropped per model: %s)", len(samples), len(tables), dropped)

    patients = samples[args.patient_col].to_numpy() if args.patient_col in samples.columns else None
    if patients is not None and pd.isna(patients).any():
        logger.warning("%d samples have no patient: patient-based measures are skipped", int(pd.isna(patients).sum()))
        patients = None
    truth = None
    if args.label_col:
        if args.label_col not in samples.columns:
            parser.error(f"--label-col {args.label_col!r} is not in the tables or --meta")
        truth = samples[args.label_col].to_numpy()
    nuisances = {}
    for column in args.nuisance_col:
        if column not in samples.columns:
            parser.error(f"--nuisance-col {column!r} is not in the tables or --meta")
        nuisances[column] = samples[column].astype(str).to_numpy()
    if args.noise_control:
        dimension = min(v.shape[1] for v in spaces.values())
        spaces[NOISE] = np.random.default_rng(0).normal(size=(len(samples), dimension))

    centerable = patients is not None
    if centerable and (pd.Series(patients).value_counts() < 2).any():
        logger.warning("some patients have a single sample, so the 'patient' centring variant is skipped")
        centerable = False
    variants = make_variants(args.center, args.pca, centerable)
    seeds = list(range(args.seeds))
    metric_rows, cluster_rows, model_rows, space_rows = [], [], [], []
    for variant, centred, dim in variants:
        prepared = {}
        for name, vectors in spaces.items():
            prepared[name] = preprocess(vectors, center_groups=patients if centred else None, pca_dim=dim)
            table = run_grid(prepared[name], ks=args.ks, seeds=seeds, groups=patients, truth=truth, nuisances=nuisances,
                             n_boot=args.n_boot, boot_fraction=args.boot_fraction)
            metric_rows.append(table.assign(model=name, variant=variant))
            logger.info("%s / %s: %d values of k", name, variant, len(table))
        ks = args.ks
        for k in sorted(k for k in ks if 2 <= k <= len(samples)):
            labels = {name: kmeans_labels(x, k, 0)[0] for name, x in prepared.items()}
            for name, lab in labels.items():
                cluster_rows.append(pd.DataFrame({"tile_id": samples["tile_id"], "model": name, "variant": variant, "k": k, "cluster": lab}))
            if len(labels) > 1:
                agreement = partition_agreement(labels)
                for a in labels:
                    for b in labels:
                        if a < b:
                            model_rows.append({"variant": variant, "k": k, "model_a": a, "model_b": b,
                                               "ari": agreement["ari"].loc[a, b], "ami": agreement["ami"].loc[a, b]})
        names = list(prepared)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                space_rows.append({"variant": variant, "model_a": a, "model_b": b, "cka": linear_cka(prepared[a], prepared[b]),
                                   f"knn{args.knn}_overlap": knn_overlap(prepared[a], prepared[b], k=min(args.knn, len(samples) - 1))})

    args.out.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat(metric_rows, ignore_index=True)
    first = ["model", "variant", "k"]
    metrics = metrics[first + [c for c in metrics.columns if c not in first]]
    metrics.to_csv(args.out / "metrics.csv", index=False)
    best = (metrics.sort_values("silhouette", ascending=False).groupby(["model", "variant"], sort=False).head(1)
            [["model", "variant", "k", "silhouette", "ari_seed_mean"] + [c for c in ["ari_heldout_mean", "nmi_nuisance", "nmi_label", *[f"nmi_{n}" for n in args.nuisance_col]] if c in metrics.columns]]
            .sort_values(["variant", "model"]).reset_index(drop=True))
    best.to_csv(args.out / "best_k.csv", index=False)
    pd.concat(cluster_rows, ignore_index=True).to_parquet(args.out / "clusters.parquet", index=False)
    pd.DataFrame(model_rows).to_csv(args.out / "compare_models.csv", index=False)
    pd.DataFrame(space_rows).to_csv(args.out / "compare_spaces.csv", index=False)

    notes = [
        f"{len(samples)} samples common to {len(tables)} models" + (f" ({len(set(patients))} patients)" if patients is not None else " (no patient information)"),
        f"k in {sorted(set(metrics['k']))}, {len(seeds)} seeds (one KMeans start each), {args.n_boot} repetitions holding out {100 * (1 - args.boot_fraction):.0f}% of the patients",
        "Silhouette is cosine; Davies-Bouldin and Calinski-Harabasz are comparable across k of one model only.",
        "The k with the best silhouette is a summary, not the true number of groups: look at the whole curve in metrics.csv.",
        "Stable clusters are not necessarily meaningful: nmi_nuisance (patient) and nmi_<column> (--nuisance-col) say how much of the clustering is a batch or source effect.",
        "Agreement with labels needs labels: without --label-col only stability and confounding are measured.",
    ]
    (args.out / "report.md").write_text(
        "# Clustering of embeddings\n\n" + "\n".join(f"- {n}" for n in notes) + "\n\n## Best k by silhouette\n\n" + markdown_table(best)
        + "\n\n## Agreement between the models' spaces\n\n" + (markdown_table(pd.DataFrame(space_rows)) if space_rows else "one model only") + "\n",
        encoding="utf-8")
    (args.out / "run.json").write_text(json.dumps({
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "embeddings"},
        "embeddings": {name: str(path) for name, path in args.embeddings}, "samples": len(samples), "dropped_per_model": dropped,
        "models": list(spaces), "variants": [v[0] for v in variants], "seconds": round(time.perf_counter() - started, 1),
    }, indent=2), encoding="utf-8")
    print(best.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
