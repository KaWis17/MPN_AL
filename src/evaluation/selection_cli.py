"""Which base model picks the most diverse tiles for annotation? The experiment of
``docs/architecture/tile-selection-experiment.md``, from embedding tables to the report.

    python scripts/run_tile_selection.py --tiles-parquet prep/tiles.parquet --images prep/slides/<id>/tiles \
        --embeddings dinov2-s14=.../embeddings.parquet --embeddings lunit-dino-s16=... --out outputs/selection/<id>

Written to ``--out``: ``descriptors.parquet``, ``clustering_metrics.csv``, ``selection_scores.csv`` (one row per
selection), ``selections.parquet``, ``summary.csv``, ``ranking.csv``, ``seed_stability.csv``, ``models_agreement.csv``,
``neighbour_quality.csv``, ``board_comparison.csv``, ``figures/`` (curves, heatmaps, maps, boards, slide map, blinded
boards), ``blind_key.json``, ``report.md`` and ``index.html``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from . import selection as sel
from . import selection_figures as sf
from .cluster_cli import load_embeddings, markdown_table, parse_named
from .cluster_figures import plot_heatmap, plot_matrix, project_2d
from .clustering import fit_kmeans, internal_metrics, preprocess
from .tile_descriptors import FEATURES, RARE_FEATURES, describe_tile, slide_background, standardize

logger = logging.getLogger("mpn_al.selection")

NOISE = "noise-control"
# Cheaper first: used to break ties. Models not listed count as the most expensive.
COST_ORDER = ("dinov2-s14", "lunit-dino-s16", "phikon-v2", "midnight-12k")
BASELINES = ("random", "spatial", "descriptor-kmeans")


def read_image(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"))


def _describe_path(path: str, mpp: float, background: np.ndarray) -> dict:
    import cv2

    cv2.setNumThreads(1)
    return describe_tile(read_image(path), mpp=mpp, background=background)


def compute_descriptors(manifest: pd.DataFrame, paths: Sequence[str], workers: int) -> pd.DataFrame:
    step = max(1, len(paths) // 100)
    background = slide_background([read_image(p) for p in paths[::step]])
    mpp = float(manifest["mpp"].iloc[0])
    describe = partial(_describe_path, mpp=mpp, background=background)
    logger.info("describing %d tiles with %d workers (slide background %s)", len(paths), workers, background.astype(int).tolist())
    if workers > 1:
        with Pool(workers) as pool:
            rows = pool.map(describe, paths, chunksize=8)
    else:
        rows = [describe(p) for p in paths]
    return pd.concat([manifest[["tile_id"]], pd.DataFrame(rows)], axis=1)


def aligned_vectors(path: Path, tile_ids: Sequence[str]) -> np.ndarray:
    """L2-normalised embeddings of one model in the order of ``tile_ids``; every tile must be there exactly once."""
    table = load_embeddings(path).set_index("tile_id")
    missing = [t for t in tile_ids if t not in table.index]
    if missing:
        raise ValueError(f"{path}: no embedding for {len(missing)} tiles, e.g. {missing[:3]}")
    vectors = np.stack(table.loc[list(tile_ids)]["embedding"].to_numpy()).astype(np.float64)
    if not np.isfinite(vectors).all():
        raise ValueError(f"{path}: embeddings contain NaN or infinite values")
    return preprocess(vectors)


class Experiment:
    def __init__(self, args, manifest: pd.DataFrame, descriptors: pd.DataFrame):
        self.args = args
        self.manifest = manifest
        self.table = descriptors
        self.D = standardize(descriptors)
        self.positions = manifest[["x", "y"]].to_numpy(dtype=np.float64)
        self.size = int(manifest["width"].iloc[0])
        self.conflicts = sel.overlap_matrix(self.positions, self.size)
        flags = sel.rare_flags(descriptors, RARE_FEATURES)
        groups = KMeans(n_clusters=min(8, len(self.D)), n_init=10, random_state=0).fit_predict(self.D)
        self.scorer = sel.Scorer(manifest, self.D, flags, groups, mpp=float(manifest["mpp"].iloc[0]))
        self.lattice = np.flatnonzero(manifest["lattice"].to_numpy())
        self.metrics: list[dict] = []
        self.scores: list[dict] = []
        self.picks: list[dict] = []
        self.labels: dict[tuple, np.ndarray] = {}
        self.selected: dict[tuple, np.ndarray] = {}

    def record(self, variant: str, method: str, k: int, seed: int, constraint: str, indices: np.ndarray, fallbacks: int = 0) -> None:
        self.selected[(variant, method, k, seed, constraint)] = indices
        self.scores.append({"variant": variant, "method": method, "k": k, "seed": seed, "constraint": constraint,
                            "fallbacks": fallbacks, **self.scorer.score(indices)})
        self.picks.extend({"variant": variant, "method": method, "k": k, "seed": seed, "constraint": constraint, "rank": r,
                           "tile_index": int(i)} for r, i in enumerate(indices))

    def run_model(self, variant: str, method: str, vectors: np.ndarray, ks: Sequence[int], seeds: Sequence[int],
                  *, candidates: np.ndarray | None = None, n_init: int = 1) -> None:
        """Cluster one model's vectors for every K and seed, then select and score representatives."""
        candidates = np.arange(len(self.D)) if candidates is None else candidates
        sub = vectors[candidates]
        conflicts = self.conflicts[np.ix_(candidates, candidates)]
        for k in ks:
            if k >= len(sub):
                continue
            for seed in seeds:
                model = fit_kmeans(sub, k, seed, n_init=n_init)
                labels = model.labels_
                self.labels[(variant, method, k, seed)] = labels
                self.metrics.append({"variant": variant, "method": method, "k": k, "seed": seed, "n": len(sub),
                                     "inertia": float(model.inertia_), **internal_metrics(sub, labels, sample_size=10000)})
                for constraint, matrix in (("no-overlap", conflicts), ("free", None)):
                    local, fallbacks = sel.representatives(sub, labels, model.cluster_centers_, conflicts=matrix)
                    self.record(variant, method, k, seed, constraint, candidates[local], fallbacks)
        logger.info("%s / %s done", variant, method)

    def run_baselines(self, ks: Sequence[int], seeds: Sequence[int]) -> None:
        for k in ks:
            if k >= len(self.D):
                continue
            rng = np.random.default_rng(1000 + k)
            for draw in range(self.args.random_draws):
                self.record("baseline", "random", k, draw, "no-overlap", sel.random_selection(len(self.D), k, rng, self.conflicts))
            rng = np.random.default_rng(2000 + k)
            for draw in range(self.args.spatial_draws):
                self.record("baseline", "spatial", k, draw, "no-overlap", sel.spatial_selection(self.positions, k, rng, self.conflicts))
            for seed in seeds:
                model = fit_kmeans(self.D, k, seed)
                chosen, fallbacks = sel.representatives(self.D, model.labels_, model.cluster_centers_, conflicts=self.conflicts, metric="euclidean")
                self.record("baseline", "descriptor-kmeans", k, seed, "no-overlap", chosen, fallbacks)
            logger.info("baselines for K = %d done", k)


# ---- summaries --------------------------------------------------------------------------------


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean and sd over seeds (or draws) per (variant, method, k, constraint), with percentiles against random selection."""
    keys = ["variant", "method", "k", "constraint"]
    values = [c for c in scores.columns if c not in keys + ["seed"]]
    grouped = scores.groupby(keys, sort=False)[values]
    table = grouped.mean().join(grouped.std(ddof=0).add_suffix("_sd")).reset_index()
    random = scores[scores["method"] == "random"]
    for column, lower_is_better in (("coverage_mean", True), ("coverage_p95", True), ("rare_categories", False)):
        percentiles = []
        for row in table.itertuples():
            draws = random[random["k"] == row.k][column].to_numpy()
            value = getattr(row, column)
            if not len(draws):
                percentiles.append(np.nan)
                continue
            beaten = (draws > value) if lower_is_better else (draws < value)
            percentiles.append(100.0 * (beaten.mean() + 0.5 * (draws == value).mean()))
        table[f"{column}_beats_random_pct"] = percentiles
    return table


def rank_models(scores: pd.DataFrame, main_ks: Sequence[int], models: Sequence[str]) -> pd.DataFrame:
    """The frozen rule: mean coverage over the main K and the seeds (lower is better); a gap smaller than the pooled sd over
    seeds is a tie, which the cheaper model wins."""
    part = scores[(scores["variant"] == "l2") & (scores["constraint"] == "no-overlap") & scores["k"].isin(main_ks) & scores["method"].isin(models)]
    per_seed = part.groupby(["method", "seed"])["coverage_mean"].mean()
    table = per_seed.groupby("method").agg(coverage="mean", coverage_sd=lambda s: float(s.std(ddof=0))).reset_index()
    cost = {name: position for position, name in enumerate(COST_ORDER)}
    table["cost_rank"] = [cost.get(m, len(cost)) for m in table["method"]]
    remaining = table.sort_values(["coverage", "cost_rank"]).to_dict("records")
    ordered = []
    while remaining:
        best = remaining[0]
        tied = [r for r in remaining
                if r["coverage"] - best["coverage"] < np.sqrt(0.5 * (r["coverage_sd"] ** 2 + best["coverage_sd"] ** 2)) or r is best]
        winner = min(tied, key=lambda r: (r["cost_rank"], r["coverage"]))
        winner["tied_with"] = ", ".join(r["method"] for r in tied if r is not winner)
        ordered.append(winner)
        remaining = [r for r in remaining if r is not winner]
    out = pd.DataFrame(ordered)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.drop(columns="cost_rank")


def seed_stability(exp: Experiment) -> pd.DataFrame:
    """Per (variant, method, k): ARI of the clusterings and agreement of the selected tiles between all pairs of seeds."""
    rows = []
    groups: dict[tuple, list[int]] = {}
    for variant, method, k, seed in exp.labels:
        groups.setdefault((variant, method, k), []).append(seed)
    for (variant, method, k), seeds in groups.items():
        pairs = list(itertools.combinations(sorted(seeds), 2))
        if not pairs:
            continue
        ari = [adjusted_rand_score(exp.labels[(variant, method, k, a)], exp.labels[(variant, method, k, b)]) for a, b in pairs]
        shared = [sel.agreement(exp.selected[(variant, method, k, a, "no-overlap")], exp.selected[(variant, method, k, b, "no-overlap")],
                                exp.positions, exp.size) for a, b in pairs]
        rows.append({"variant": variant, "method": method, "k": k, "ari_seed": float(np.mean(ari)), "ari_seed_sd": float(np.std(ari)),
                     "selection_identical": float(np.mean([s["identical_share"] for s in shared])),
                     "selection_matched": float(np.mean([s["matched_share"] for s in shared])),
                     "selection_matched_sd": float(np.std([s["matched_share"] for s in shared]))})
    return pd.DataFrame(rows)


def equivalent_share(exp: Experiment, a: np.ndarray, b: np.ndarray, threshold: float) -> float:
    """Share of a one-to-one greedy matching of ``a`` and ``b`` whose descriptor distance is below ``threshold``."""
    distance = np.sqrt(((exp.D[a][:, None, :] - exp.D[b][None, :, :]) ** 2).sum(axis=2))
    matched = 0
    while distance.size and np.isfinite(distance).any():
        i, j = np.unravel_index(np.argmin(distance), distance.shape)
        if distance[i, j] > threshold:
            break
        matched += 1
        distance[i, :], distance[:, j] = np.inf, np.inf
    return matched / max(min(len(a), len(b)), 1)


# ---- figures and report -----------------------------------------------------------------------


def draw(exp: Experiment, args, out: Path, paths: Sequence[str], spaces: dict, summary: pd.DataFrame, stability: pd.DataFrame,
         ranking: pd.DataFrame, agreement_rows: pd.DataFrame) -> tuple[list[tuple[str, list[tuple[str, str]]]], pd.DataFrame]:
    import cv2

    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    models = [m for m in spaces if m != NOISE]

    def save_rgb(name: str, rgb: np.ndarray) -> str:
        cv2.imwrite(str(figures / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return name

    cache: dict[int, np.ndarray] = {}

    def small_image(index: int) -> np.ndarray:
        if index not in cache:
            cache[index] = cv2.resize(read_image(paths[index]), (256, 256), interpolation=cv2.INTER_AREA)
        return cache[index]

    sections: list[tuple[str, list[tuple[str, str]]]] = []
    metrics = pd.DataFrame(exp.metrics)
    main = metrics[metrics["variant"] == "l2"]
    grouped = main.groupby(["method", "k"], sort=False)[["silhouette", "davies_bouldin", "calinski_harabasz"]]
    curve = grouped.mean().join(grouped.std(ddof=0).add_suffix("_sd")).reset_index()
    curve = curve.merge(stability[stability["variant"] == "l2"][["method", "k", "ari_seed", "ari_seed_sd"]], on=["method", "k"], how="left")
    sf.plot_metric_curves(curve, [("silhouette", "silhouette (cosine, higher = better)"), ("davies_bouldin", "Davies-Bouldin (lower, within model)"),
                                  ("calinski_harabasz", "Calinski-Harabasz (higher, within model)"), ("ari_seed", "ARI between seeds")],
                          figures / "curves_clustering.png", title="clustering, full L2-normalised embeddings")
    chosen = summary[(summary["variant"].isin(["l2", "baseline"])) & (summary["constraint"] == "no-overlap")]
    sf.plot_metric_curves(chosen, [("coverage_mean", "coverage: mean distance to nearest selected (lower)"), ("coverage_p95", "coverage, 95th percentile (lower)"),
                                   ("rare_categories", "rare categories covered (higher)"), ("jsd_groups", "JS divergence of descriptor groups (lower)"),
                                   ("grid_cells", "slide grid cells covered (higher)"), ("diversity_mean", "mean distance between selected (higher)")],
                          figures / "curves_selection.png", title="selection (no overlapping tiles)")
    items = [("clustering metrics against K", "curves_clustering.png"), ("selection scores against K, with baselines", "curves_selection.png")]
    pivot = curve.rename(columns={"method": "model"}).assign(variant="l2")
    pivot = pivot.merge(chosen[chosen["variant"] == "l2"][["method", "k", "coverage_mean"]].rename(columns={"method": "model"}), on=["model", "k"], how="left")
    for column in ("silhouette", "ari_seed", "coverage_mean"):
        plot_heatmap(pivot, "l2", column, figures / f"heatmap_{column}.png")
        items.append((f"{column}: model x K", f"heatmap_{column}.png"))
    sections.append(("Metrics", items))

    items = []
    matrix_k = max(k for k in args.main_ks if k in set(main["k"])) if set(args.main_ks) & set(main["k"]) else int(main["k"].max())
    for column, title in (("ari", f"ARI between clusterings, K = {matrix_k}"), ("knn_overlap", f"kNN overlap (k = {args.knn}, overlapping tiles excluded)")):
        rows = agreement_rows[agreement_rows["k"] == matrix_k]
        names = list(spaces)
        matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
        for row in rows.itertuples():
            matrix.loc[row.model_a, row.model_b] = matrix.loc[row.model_b, row.model_a] = getattr(row, column)
        plot_matrix(matrix, title, figures / f"matrix_{column}.png")
        items.append((title, f"matrix_{column}.png"))
    sections.append(("Models compared", items))

    board_k = args.board_k if (("l2", models[0], args.board_k, 0, "no-overlap") in exp.selected) else int(main["k"].max())
    items = []
    for name, vectors in spaces.items():
        coords, method = project_2d(vectors, method=args.projection)
        sf.plot_space_map(coords, exp.labels[("l2", name, board_k, 0)],
                          {"white (fat, glass)": exp.table["white"].to_numpy(), "nuclear density": exp.table["nuclear_density"].to_numpy(),
                           "x on the slide": exp.positions[:, 0]},
                          exp.selected[("l2", name, board_k, 0, "no-overlap")], figures / f"map_{name}.png", title=f"{name}, K = {board_k}, {method}")
        items.append((f"2-D map of {name}", f"map_{name}.png"))
    sections.append(("Embedding maps", items))

    top = [m for m in ranking["method"].tolist()[:2]]
    sets = {name: exp.selected[("l2", name, board_k, 0, "no-overlap")] for name in top}
    sets["random"] = exp.selected[("baseline", "random", board_k, 0, "no-overlap")]
    density = exp.table["nuclear_density"].to_numpy()
    captions = {i: f"x{int(exp.positions[i, 0])} y{int(exp.positions[i, 1])}" for s in sets.values() for i in s}
    exact, near = (0, 150, 0), (255, 140, 0)
    items = []
    for name, indices in sets.items():
        ordered = sorted(indices, key=lambda i: density[i])
        marks = {}
        others = [s for n, s in sets.items() if n != name and n in top]
        if name in top and others:
            other = set(int(i) for i in others[0])
            for i in ordered:
                if int(i) in other:
                    marks[i] = exact
                elif sel.overlaps(exp.positions, i, list(other), exp.size).any():
                    marks[i] = near
        title = f"{name}: {len(ordered)} tiles, ordered by nuclear density" + (" (green: also chosen by the other model, orange: overlaps one)" if name in top else "")
        file = save_rgb(f"board_{name}_K{board_k}.png", sf.board(ordered, small_image, title=title, marks=marks, captions=captions))
        items.append((f"selected by {name}", file))
    mosaic, origin = sf.slide_mosaic(exp.positions, exp.size, small_image)
    save_rgb(f"slide_map_K{board_k}.png", sf.slide_map(mosaic, origin, exp.positions, exp.size, {f"{n} ({len(s)})": s for n, s in sets.items()}))
    items.append(("where the selected tiles lie on the slide", f"slide_map_K{board_k}.png"))
    sections.append((f"Selections, K = {board_k}, seed 0", items))

    order = np.random.default_rng(args.blind_seed).permutation(list(sets))
    key = {}
    blind = []
    shuffle = np.random.default_rng(args.blind_seed + 1)
    for letter, name in zip("ABC", order):
        indices = list(sets[str(name)])
        shuffle.shuffle(indices)
        indices.sort(key=lambda i: density[i])
        file = save_rgb(f"blind_{letter}.png", sf.board(indices, small_image, title=f"Set {letter}"))
        key[letter] = str(name)
        blind.append((f"set {letter}", file))
    (out / "blind_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")

    threshold = float(np.quantile(np.sqrt(((exp.D[::3][:, None, :] - exp.D[1::3][None, :, :]) ** 2).sum(axis=2)), 0.05))
    comparison = []
    for a, b in itertools.combinations(sets, 2):
        row = {"set_a": a, "set_b": b, **sel.agreement(sets[a], sets[b], exp.positions, exp.size)}
        row["descriptor_equivalent_share"] = equivalent_share(exp, sets[a], sets[b], threshold)
        comparison.append(row)
    return sections, pd.DataFrame(comparison)


def write_report(out: Path, exp: Experiment, args, summary: pd.DataFrame, ranking: pd.DataFrame, stability: pd.DataFrame,
                 quality: pd.DataFrame, comparison: pd.DataFrame, chance: float) -> None:
    def view(frame: pd.DataFrame, columns: Sequence[str]) -> str:
        return markdown_table(frame[[c for c in columns if c in frame.columns]].reset_index(drop=True))

    columns = ["method", "k", "coverage_mean", "coverage_mean_sd", "coverage_mean_beats_random_pct", "coverage_p95", "rare_categories",
               "jsd_groups", "grid_cells", "nearest_neighbour_mm"]
    main = summary[(summary["variant"].isin(["l2", "baseline"])) & (summary["constraint"] == "no-overlap") & summary["k"].isin(args.main_ks)]
    free = summary[(summary["variant"] == "l2") & (summary["constraint"] == "free") & summary["k"].isin(args.main_ks)]
    parts = [
        "# Selekcja kafli: wyniki",
        f"\n{len(exp.D)} kafli, {len(set(exp.manifest['region']))} fragmentów tkanki, {len(exp.lattice)} kafli w siatce bez nakładania. "
        f"Cechy: {', '.join(FEATURES)}. Protokół: `docs/architecture/tile-selection-experiment.md`.",
        "\n## Ranking (kryterium główne: pokrycie dla K = " + ", ".join(map(str, args.main_ks)) + ", mniej = lepiej)\n",
        markdown_table(ranking, 4),
        "\n### Punkty odniesienia dla tych samych K (ta sama miara)\n",
        markdown_table(main[main["method"].isin([NOISE, *BASELINES])].groupby("method", sort=False)[
            ["coverage_mean", "coverage_mean_beats_random_pct", "rare_categories"]].mean().sort_values("coverage_mean").reset_index(), 4),
        "\nModel, który nie ma niższego pokrycia niż selekcja losowa i przestrzenna oraz kontrola szumem, nie wnosi tu informacji.",
        "\n## Selekcja bez nakładania, główne K (średnia po seedach lub losowaniach)\n", view(main.sort_values(["k", "coverage_mean"]), columns),
        "\n## Bez ograniczenia nakładania (ile par wybranych kafli się nakłada)\n", view(free, ["method", "k", "coverage_mean", "overlapping_pairs", "rare_categories"]),
        "\n## Analizy wrażliwości (pokrycie, główne K)\n",
        view(summary[(summary["variant"] != "baseline") & (summary["constraint"] == "no-overlap") & summary["k"].isin(args.main_ks)].sort_values(["variant", "k", "coverage_mean"]),
             ["variant", "method", "k", "coverage_mean", "coverage_mean_sd", "rare_categories"]),
        "\n## Stabilność między seedami (główne K)\n", view(stability[stability["k"].isin(args.main_ks)], ["variant", "method", "k", "ari_seed", "ari_seed_sd", "selection_identical", "selection_matched"]),
        f"\n## Czy sąsiedzi w przestrzeni modelu wyglądają podobnie (odległość cech do {args.knn} sąsiadów; losowa para: {chance:.3f})\n",
        markdown_table(quality),
        f"\n## Porównanie zestawów z plansz (K = {args.board_k}, seed 0)\n", markdown_table(comparison),
        "\n## Jak czytać\n",
        "- Pokrycie i reszta miar selekcji liczone są w cechach z pikseli (przybliżenia), nie w przestrzeni modeli.",
        "- Jeden slajd: różnice dotyczą tego preparatu. Rozrzut między seedami nie obejmuje zmienności między pacjentami.",
        "- Davies-Bouldin i Calinski-Harabasz porównuj tylko w obrębie modelu.",
        "- Plansze A, B, C dla patomorfologa są w `figures/blind_*.png`; klucz w `blind_key.json` (nie pokazywać oceniającemu).",
    ]
    (out / "report.md").write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiles-parquet", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="directory with <tile_id>.png")
    parser.add_argument("--embeddings", action="append", required=True, type=parse_named, metavar="NAME=FILE")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, help="reuse a descriptors.parquet instead of computing it")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 30, 50, 80, 100])
    parser.add_argument("--main-ks", type=int, nargs="+", default=[50, 80])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--random-draws", type=int, default=1000)
    parser.add_argument("--spatial-draws", type=int, default=100)
    parser.add_argument("--pca", type=int, default=128, help="extra sensitivity variant with PCA to this size (0 = skip)")
    parser.add_argument("--top-n-init", type=int, default=10, help="n_init for the two best models at the main K (0 = skip)")
    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("--board-k", type=int, default=50)
    parser.add_argument("--blind-seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=4, help="processes that compute the descriptors")
    parser.add_argument("--projection", choices=["auto", "umap", "tsne"], default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    started = time.perf_counter()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    manifest = sel.tile_manifest(pd.read_parquet(args.tiles_parquet))
    if manifest["slide_id"].nunique() != 1:
        parser.error("this experiment handles the tiles of one slide")
    paths = [str(args.images / f"{t}.png") for t in manifest["tile_id"]]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        parser.error(f"{len(missing)} tile images are missing, e.g. {missing[0]}")
    spaces = {}
    for name, path in args.embeddings:
        if name is None:
            parser.error("give embeddings as NAME=FILE")
        spaces[name] = aligned_vectors(path, manifest["tile_id"].tolist())
        logger.info("%s: %d x %d", name, *spaces[name].shape)
    dimension = min(v.shape[1] for v in spaces.values())
    spaces[NOISE] = preprocess(np.random.default_rng(0).normal(size=(len(manifest), dimension)))

    if args.descriptors:
        descriptors = pd.read_parquet(args.descriptors).set_index("tile_id").loc[manifest["tile_id"]].reset_index()
    else:
        descriptors = compute_descriptors(manifest, paths, args.workers)
    descriptors.to_parquet(out / "descriptors.parquet", index=False)
    manifest.to_parquet(out / "manifest.parquet", index=False)

    exp = Experiment(args, manifest, descriptors)
    seeds = list(range(args.seeds))
    models = [m for m in spaces if m != NOISE]
    for name, vectors in spaces.items():
        exp.run_model("l2", name, vectors, args.ks, seeds)
    for name, vectors in spaces.items():
        exp.run_model("lattice", name, vectors, args.ks, seeds, candidates=exp.lattice)
    if args.pca:
        for name in models:
            exp.run_model(f"pca{args.pca}", name, preprocess(spaces[name], pca_dim=args.pca), args.ks, seeds)
    exp.run_baselines(args.ks, seeds)

    ranking = rank_models(pd.DataFrame(exp.scores), args.main_ks, models)
    if args.top_n_init:
        for name in ranking["method"].tolist()[:2]:
            exp.run_model(f"l2-ninit{args.top_n_init}", name, spaces[name], [k for k in args.ks if k in args.main_ks], seeds, n_init=args.top_n_init)

    scores = pd.DataFrame(exp.scores)
    summary = summarize(scores)
    stability = seed_stability(exp)
    chance = sel.chance_consistency(exp.D)
    quality = pd.DataFrame([{"model": name, "neighbour_distance": sel.neighbour_consistency(vectors, exp.D, exp.conflicts, args.knn),
                             "random_pair_distance": chance} for name, vectors in spaces.items()]).sort_values("neighbour_distance")
    agreement_rows = []
    knn = {(a, b): sel.neighbour_overlap(spaces[a], spaces[b], exp.conflicts, args.knn) for a, b in itertools.combinations(spaces, 2)}
    for k in args.ks:
        for (a, b), overlap in knn.items():
            if ("l2", a, k, 0) not in exp.labels:
                continue
            values = [adjusted_rand_score(exp.labels[("l2", a, k, s)], exp.labels[("l2", b, k, s)]) for s in seeds]
            agreement_rows.append({"k": k, "model_a": a, "model_b": b, "ari": float(np.mean(values)), "ari_sd": float(np.std(values)), "knn_overlap": overlap})
    agreement_rows = pd.DataFrame(agreement_rows)

    metrics = pd.DataFrame(exp.metrics)
    metrics.to_csv(out / "clustering_metrics.csv", index=False)
    scores.to_csv(out / "selection_scores.csv", index=False)
    picks = pd.DataFrame(exp.picks)
    picks["tile_id"] = manifest["tile_id"].to_numpy()[picks["tile_index"]]
    picks[picks["method"] != "random"].to_parquet(out / "selections.parquet", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    ranking.to_csv(out / "ranking.csv", index=False)
    stability.to_csv(out / "seed_stability.csv", index=False)
    agreement_rows.to_csv(out / "models_agreement.csv", index=False)
    quality.to_csv(out / "neighbour_quality.csv", index=False)

    sections, comparison = draw(exp, args, out, paths, spaces, summary, stability, ranking, agreement_rows)
    comparison.to_csv(out / "board_comparison.csv", index=False)
    write_report(out, exp, args, summary, ranking, stability, quality, comparison, chance)
    from ..experiment.figures import write_index_html

    write_index_html(
        out / "figures" / "index.html", [{"tile_id": title, "caption": "", "figures": items} for title, items in sections],
        title=f"Tile selection, slide {manifest['slide_id'].iloc[0]}",
        notes=["Selection scores use pixel descriptors, not the models' own spaces. One slide only."], summary=ranking.drop(columns="tied_with"),
    )
    (out / "run.json").write_text(json.dumps(
        {"args": {k: ([str(x) for x in v] if k == "embeddings" else (str(v) if isinstance(v, Path) else v)) for k, v in vars(args).items()},
         "tiles": len(manifest), "models": models, "seconds": round(time.perf_counter() - started, 1)}, indent=2), encoding="utf-8")
    print(ranking.to_string(index=False))
    logger.info("done in %.0f s: %s", time.perf_counter() - started, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
