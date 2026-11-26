"""Draw the results of ``cluster_embeddings.py``: curves, heatmaps, model-to-model matrices, 2-D maps, cluster composition,
galleries of tiles per cluster and QuPath files.

    python scripts/visualize_clusters.py --cluster-dir outputs/clustering/bambo \
        --embeddings phikon-v2=a/embeddings.parquet --embeddings lunit-dino-s16=b/embeddings.parquet \
        --meta outputs/bambo_meta.csv --images outputs/bambo_png --nuisance-col image_size --k 4

The embeddings, ``--meta`` and column options must be those of the clustering run: the maps and galleries are drawn from the same
prepared vectors and the clusters stored in ``clusters.parquet``. Output goes to ``<cluster-dir>/figures`` with an ``index.html``.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..experiment.figures import write_index_html
from . import cluster_figures as cf
from .cluster_cli import aligned_spaces, load_embeddings, load_meta, parse_named
from .clustering import preprocess

logger = logging.getLogger("mpn_al.cluster_viz")


def parse_variant(variant: str) -> tuple[bool, int | None]:
    """``center+l2+pca50`` -> (centred by patient, 50 components)."""
    match = re.search(r"pca(\d+)", variant)
    return "center" in variant.split("+"), int(match[1]) if match else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cluster-dir", type=Path, required=True, help="output directory of cluster_embeddings.py")
    parser.add_argument("--embeddings", action="append", required=True, type=parse_named, metavar="NAME=FILE",
                        help="the same embedding tables as in the clustering run (names must match)")
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--patient-col", default="patient_id")
    parser.add_argument("--label-col")
    parser.add_argument("--nuisance-col", action="append", default=[])
    parser.add_argument("--window-mode")
    parser.add_argument("--pooling")
    parser.add_argument("--variant", help="preparation variant to draw (default: the first one in metrics.csv)")
    parser.add_argument("--k", type=int, help="number of clusters for the maps, galleries and matrices (default: best silhouette of the first model)")
    parser.add_argument("--images", type=Path, help="directory of tile images named <tile_id>.<ext>, for the galleries")
    parser.add_argument("--thumb", type=int, default=128)
    parser.add_argument("--members", type=int, default=6, help="tiles per gallery half (nearest the centre / random)")
    parser.add_argument("--projection", choices=["auto", "umap", "tsne"], default="auto")
    parser.add_argument("--out", type=Path, help="default: <cluster-dir>/figures")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = args.out or args.cluster_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(args.cluster_dir / "metrics.csv")
    clusters = pd.read_parquet(args.cluster_dir / "clusters.parquet")
    variant = args.variant or metrics["variant"].iloc[0]
    if variant not in set(metrics["variant"]):
        parser.error(f"variant {variant!r} is not in metrics.csv: {sorted(set(metrics['variant']))}")
    names = [n for n, _ in args.embeddings]
    models = [m for m in dict.fromkeys(metrics["model"]) if m != cf.CONTROL]
    k = args.k
    if k is None:
        first = metrics[(metrics["model"] == models[0]) & (metrics["variant"] == variant)]
        k = int(first.sort_values("silhouette", ascending=False)["k"].iloc[0])
        logger.info("--k not given: using k = %d (best silhouette of %s)", k, models[0])
    if k not in set(metrics["k"]):
        parser.error(f"k = {k} was not clustered: {sorted(set(metrics['k']))}")

    figures: dict[str, list[tuple[str, str]]] = {"Metrics": [], "Models compared": []}
    for each in sorted(set(metrics["variant"])):
        name = f"curves_{each}.png"
        cf.plot_curves(metrics, each, out / name, extra=args.nuisance_col)
        figures["Metrics"].append((f"curves, variant {each}", name))
    for column in ("silhouette", "ari_seed_mean"):
        name = f"heatmap_{column}_{variant}.png"
        cf.plot_heatmap(metrics, variant, column, out / name)
        figures["Metrics"].append((f"{column} by model and k, variant {variant}", name))

    models_table = args.cluster_dir / "compare_models.csv"
    spaces_table = args.cluster_dir / "compare_spaces.csv"
    if models_table.stat().st_size > 2:
        agreement = pd.read_csv(models_table)
        if len(agreement):
            cf.plot_matrix(cf.pivot_pairs(agreement, "ari", variant=variant, k=k), f"ARI between clusterings, k = {k}, {variant}", out / "matrix_ari.png")
            figures["Models compared"].append((f"ARI between the models' clusterings, k = {k}", "matrix_ari.png"))
    if spaces_table.stat().st_size > 2:
        spaces_compared = pd.read_csv(spaces_table)
        for column in [c for c in spaces_compared.columns if c in ("cka",) or c.startswith("knn")]:
            if len(spaces_compared):
                cf.plot_matrix(cf.pivot_pairs(spaces_compared, column, variant=variant), f"{column} between spaces, {variant}", out / f"matrix_{column}.png")
                figures["Models compared"].append((f"{column} between the models' spaces", f"matrix_{column}.png"))

    tables = {}
    for name, path in args.embeddings:
        tables[name or "model"] = load_embeddings(path, window_mode=args.window_mode, pooling=args.pooling)
    samples, spaces = aligned_spaces(tables, load_meta(args.meta) if args.meta else None)
    centred, dimension = parse_variant(variant)
    patients = samples[args.patient_col].to_numpy() if args.patient_col in samples.columns else None
    categories = {}
    if args.label_col:
        categories[args.label_col] = samples[args.label_col].astype(str).to_numpy()
    for column in args.nuisance_col:
        categories[column] = samples[column].astype(str).to_numpy()
    scatter_extra = dict(categories)
    if patients is not None:
        scatter_extra["patient"] = np.asarray(patients).astype(str)

    paths = None
    if args.images is not None:
        from ..image_backend.tiles import ImageFolderSource

        paths = {tile.tile_id: tile.path for tile in ImageFolderSource(args.images).tiles}

    for name in models:
        if name not in spaces:
            logger.warning("%s is in the clustering but not among --embeddings: no map or gallery for it", name)
            continue
        prepared = preprocess(spaces[name], center_groups=patients if centred else None, pca_dim=dimension)
        chosen = clusters[(clusters["model"] == name) & (clusters["variant"] == variant) & (clusters["k"] == k)].set_index("tile_id")["cluster"]
        labels = chosen.reindex(samples["tile_id"]).to_numpy()
        if np.isnan(labels.astype(float)).any():
            parser.error(f"clusters.parquet has no clusters of {name} / {variant} / k={k} for every sample")
        labels = labels.astype(int)
        coords, method = cf.project_2d(prepared, method=args.projection)
        panels = {f"cluster (k={k})": labels, **scatter_extra}
        scatter = f"map_{name}.png"
        cf.plot_scatter(coords, panels, out / scatter, method=method, title=f"{name}, {variant}")
        figures.setdefault(name, []).append((f"2-D map of the space ({method}) coloured by cluster and by what should not explain it", scatter))
        if categories:
            composition = f"composition_{name}.png"
            cf.plot_composition(labels, categories, out / composition, title=f"{name}, k = {k}, {variant}")
            figures[name].append(("what each cluster is made of", composition))
        if paths is not None:
            from PIL import Image

            def image_of(index: int, _paths=paths, _ids=samples["tile_id"].to_numpy()) -> np.ndarray:
                with Image.open(_paths[_ids[index]]) as handle:
                    return np.asarray(handle.convert("RGB"))

            if all(t in paths for t in samples["tile_id"]):
                import cv2

                gallery = f"gallery_{name}.png"
                members = cf.cluster_members(prepared, labels, near=args.members, random=args.members)
                montage = cf.render_gallery(members, image_of, thumb=args.thumb, title=f"{name}, k = {k}")
                cv2.imwrite(str(out / gallery), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
                figures[name].append(("tiles nearest each cluster's centre, and random members", gallery))
            else:
                logger.warning("some tiles have no image in %s: no gallery for %s", args.images, name)
        table = pd.DataFrame({"tile_id": samples["tile_id"], "cluster": labels})
        for column in ("x", "y", "width", "size", "slide_id"):
            if column in samples.columns:
                table[column] = samples[column]
        written = cf.clusters_to_geojson(table, out / "qupath" / f"{name}_k{k}", prefix=f"{name}_cluster")
        if written:
            logger.info("%s: %d QuPath file(s) in %s", name, len(written), out / "qupath" / f"{name}_k{k}")

    sections = [{"tile_id": title, "caption": "", "figures": items} for title, items in figures.items() if items]
    write_index_html(
        out / "index.html", sections, title=f"Clustering of embeddings: {args.cluster_dir.name}",
        notes=[f"variant {variant}, k = {k} for the maps, matrices and galleries.",
               "Colour in a 2-D map is not comparable between models; the maps show structure, not distance.",
               "A stable clustering can still be a batch effect: read the NMI with patient and with the nuisance columns next to the stability."],
        summary=pd.read_csv(args.cluster_dir / "best_k.csv"),
    )
    logger.info("figures in %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
