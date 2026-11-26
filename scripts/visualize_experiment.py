#!/usr/bin/env python
"""Draw an experiment's results on images, from the predictions its runs saved (no GPU, no encoder).

Runs must have been made with a ``visualize:`` section in the grid file (or ``--viz-per-split``), see
docs/architecture/experiment-plan.md. Then, on any machine that has the tile images:

    python scripts/visualize_experiment.py outputs/plgrid/exp_bambo_final --images data/bambo_png \
        --out outputs/figures/bambo --split validation --fraction 1.0 --seed 0

Per saved tile it writes, in ``--out``: ``<tile>_gallery.png`` (one row per model), ``_zoom.png`` (densest
window enlarged), ``_curve.png`` (more training tiles, same seed), ``_seeds.png`` (seeds overlaid),
``_disagreement.png`` (how many models call each pixel a cell), ``viz_scores.csv`` (every run x tile) and an
``index.html``. The test split exists only for runs made with ``--final``. Figure text is ASCII (OpenCV).
"""

import _bootstrap  # noqa: F401  (makes `mpn_al` importable from a plain checkout)

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mpn_al.experiment import figures as fg
from mpn_al.image_backend.tiles import ImageFolderSource

logger = logging.getLogger("visualize_experiment")


def parse_seg_mpp(text: str) -> float | None:
    return None if text == "native" else float(text)


def write_png(path: Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", type=Path, help="experiment directory (contains <grid>/<run_id>/result.json)")
    parser.add_argument("--images", type=Path, required=True, help="directory of tile images named <tile_id>.<ext>")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--fraction", type=float, help="share of training tiles shown in the gallery (default: the largest)")
    parser.add_argument("--seed", type=int, help="seed shown in the gallery and the learning curve (default: the smallest)")
    parser.add_argument("--seg-mpp", type=parse_seg_mpp, help="working resolution to show: a number or 'native' (needed if the grid has several)")
    parser.add_argument("--kinds", nargs="+", choices=fg.KINDS, default=list(fg.KINDS))
    parser.add_argument("--tiles", nargs="+", help="only these tile ids")
    parser.add_argument("--zoom", type=int, default=256, help="side in px of the zoom window (0 = no zoom figure)")
    parser.add_argument("--panel", type=int, default=320, help="size in px of each panel")
    parser.add_argument("--reference-note", default="reference = the masks the experiment was scored against "
                        "(stain-baseline pseudo-labels unless they come from a pathologist)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    runs = fg.discover_runs(args.experiment)
    if not runs:
        parser.error(f"no runs with saved predictions under {args.experiment} (was the grid run with a visualize: section?)")
    resolutions = sorted({r.seg_mpp for r in runs}, key=lambda v: -1 if v is None else v)
    if args.seg_mpp is None and len(resolutions) > 1:
        parser.error(f"several working resolutions {resolutions}: pick one with --seg-mpp")
    runs = [r for r in runs if len(resolutions) == 1 or r.seg_mpp == args.seg_mpp]
    runs = [r for r in runs if not fg.read_index(r, args.split).empty]
    if not runs:
        parser.error(f"no run saved predictions for the {args.split} split (the test split needs a --final run)")

    fraction = args.fraction if args.fraction is not None else max(r.fraction for r in runs)
    seed = args.seed if args.seed is not None else min(r.seed for r in runs)
    chosen = [r for r in runs if r.fraction == fraction and r.seed == seed]
    if not chosen:
        parser.error(f"no run with fraction {fraction} and seed {seed}")
    chosen.sort(key=lambda r: r.model)
    models = [r.model for r in chosen]

    tiles = fg.read_index(chosen[0], args.split)
    if args.tiles:
        tiles = tiles[tiles["tile_id"].isin(args.tiles)]
    source = ImageFolderSource(args.images)
    paths = {tile.tile_id: tile.path for tile in source.tiles}
    args.out.mkdir(parents=True, exist_ok=True)

    scores = fg.scores_table(runs, args.split)
    scores.to_csv(args.out / "viz_scores.csv", index=False)
    bad = scores[~scores["matches_counts"]]
    if len(bad):
        logger.warning("%d of %d saved predictions do NOT reproduce the run's own counts (see viz_scores.csv)", len(bad), len(scores))

    sections = []
    for tile in tiles.itertuples():
        if tile.tile_id not in paths:
            logger.warning("no image for %s in %s, skipped", tile.tile_id, args.images)
            continue

        def data_of(run):
            return fg.load_prediction(run, args.split, tile.tile_id)

        entries = [(run.model, data_of(run)) for run in chosen]
        shape = entries[0][1]["instances"].shape
        image = fg.load_image(paths[tile.tile_id], shape)
        box = fg.zoom_box(entries[0][1]["reference"], args.zoom)
        figures = []

        def emit(kind, rgb):
            name = f"{tile.tile_id}_{kind}.png"
            write_png(args.out / name, rgb)
            figures.append((kind, name))

        if "gallery" in args.kinds:
            emit("gallery", fg.render_gallery(image, entries, panel=args.panel))
        if "zoom" in args.kinds and box is not None:
            emit("zoom", fg.render_gallery(image, entries, box=box, panel=args.panel + 160))
        if "curve" in args.kinds:
            curves = {m: [(r.fraction, data_of(r)) for r in runs if r.model == m and r.seed == seed] for m in models}
            if any(len(points) > 1 for points in curves.values()):
                emit("curve", fg.render_curve(image, curves, panel=args.panel))
        if "seeds" in args.kinds:
            seeds = {m: [(r.seed, data_of(r)) for r in runs if r.model == m and r.fraction == fraction] for m in models}
            if any(len(points) > 1 for points in seeds.values()):
                emit("seeds", fg.render_seeds(image, seeds, panel=args.panel))
        if "disagreement" in args.kinds and len(entries) > 1:
            emit("disagreement", fg.render_disagreement(image, entries, panel=args.panel))
        sections.append({
            "tile_id": tile.tile_id,
            "caption": f"patient {tile.patient_id}, group {tile.group}, {tile.density:.0f} reference cells/Mpx, "
                       f"{shape[1]}x{shape[0]} px at working resolution",
            "figures": figures,
        })
        logger.info("%s: %d figures", tile.tile_id, len(figures))

    shown = scores[(scores["fraction"] == fraction) & (scores["seed"] == seed)]
    summary = shown.groupby("model")[["pq", "dq", "sq"]].mean().reset_index()
    fg.write_index_html(
        args.out / "index.html", sections,
        title=f"Results on images: {args.experiment.name}, {args.split}",
        notes=[f"{len(tiles)} tiles chosen from the data alone (groups: dense, atypical, random); gallery shows "
               f"{fraction * 100:g}% of the training tiles, seed {seed}.", args.reference_note,
               "Mean PQ over the shown tiles only; the experiment's own PQ is over the whole split."],
        summary=summary,
    )
    print(summary.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
