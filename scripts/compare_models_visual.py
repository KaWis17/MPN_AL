#!/usr/bin/env python
"""Look at what trained heads do with a tile, model by model, and how much neighbouring context matters.

    python scripts/compare_models_visual.py \
        --head dinov2-s14=runs/dinov2/head/head.pt --head lunit-dino-s16=runs/lunit/head/head.pt \
        --images tiles_png/ --mpp 0.25 --limit 6 --context-margin 128 --out outputs/visual

Per tile, in ``--out``:

    <tile_id>_compare.png   one row per model: outlines | cell probability | HV maps | encoder-token PCA
    <tile_id>_zoom.png      the same densest region of the tile, enlarged: image | reference | one outline panel per model
                            (with the model's PQ against the reference when --masks is given); --zoom 0 skips it
    reference_pq.csv        (with --masks) PQ/DQ/SQ and cell counts of every model against the reference, per tile
    <tile_id>_context.png   (with --context-margin) the central region alone vs. cut out of a larger image
    geometry.csv            what each encoder actually receives for a tile of this size
    agreement.csv           pairwise PQ between models, per tile
    context_effect.csv      per tile and model; context_summary.csv is the mean over tiles

``--masks`` adds the reference outlines (annotation or pseudo-label) to the first row. Tiles are
treated as independent images: ``--context-margin M`` crops M px off every side of each image and
compares "region alone" with "region inside the whole image", i.e. what a tile gains from neighbours.
"""

import _bootstrap  # noqa: F401  (makes `mpn_al` importable from a plain checkout)

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mpn_al.embeddings.cli import default_device
from mpn_al.evaluation.visual_compare import (
    agreement_table,
    context_effect,
    densest_box,
    model_geometry,
    reference_scores,
    render_comparison,
    render_zoom,
    render_context,
    run_model,
    summarize_context,
)
from mpn_al.image_backend.cli_args import add_source_arguments, source_from_arguments
from mpn_al.segmenter.model import load_segmenter, read_head
from mpn_al.training.dataset import _read_masks

logger = logging.getLogger("compare_models_visual")


def parse_head(text: str) -> tuple[str | None, Path]:
    name, sep, path = text.partition("=")
    return (name, Path(path)) if sep else (None, Path(text))


def write_png(path: Path, rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", action="append", required=True, type=parse_head, metavar="[NAME=]HEAD.pt",
                        help="head checkpoint (repeat for each model to compare)")
    parser.add_argument("--masks", type=Path, help="reference instance masks (layout of --masks in train_segmentation_head.py)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context-margin", type=int, default=0,
                        help="px cut off every side of each image for the context test (0 = skip it)")
    parser.add_argument("--zoom", type=int, default=256,
                        help="side in px of the enlarged region shown side by side for all models (0 = skip)")
    parser.add_argument("--zoom-panel", type=int, default=480, help="size in px of each panel of the zoom figure")
    parser.add_argument("--band", type=int, default=48, help="width in px of the border band the context test scores separately")
    parser.add_argument("--panel", type=int, default=360, help="size in px of each panel in the figures")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--random-init", action="store_true", help="random encoders; only to test that the script runs")
    add_source_arguments(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    source, tiles = source_from_arguments(args)
    segmenters = {}
    for name, path in args.head:
        segmenter = load_segmenter(path, pretrained=not args.random_init, device=args.device)
        label = name or read_head(path)["base_model"]["name"]
        if label in segmenters:
            parser.error(f"two heads are called {label!r}; name them with NAME=PATH")
        segmenters[label] = segmenter
        logger.info("%s: %s, working resolution %s", label, path, segmenter.seg_mpp or "native")

    geometry_rows, agreement_rows, context_rows, score_rows = [], [], [], []
    for tile in tiles:
        image = source.read(tile)
        mpp = tile.mpp
        views = [run_model(name, seg, image, mpp=mpp, amp=not args.no_amp) for name, seg in segmenters.items()]
        reference = None
        if args.masks is not None:
            read = _read_masks(args.masks, tile)
            if read is not None and read[0].shape == image.shape[:2]:
                reference = read[0]
        write_png(args.out / f"{tile.tile_id}_compare.png",
                  render_comparison(image, views, reference=reference, reference_name="reference", panel=args.panel))
        scores = reference_scores(views, reference) if reference is not None else None
        if scores is not None:
            score_rows += [{"tile_id": tile.tile_id, **row} for row in scores.to_dict("records")]
        if args.zoom:
            anchor = reference if reference is not None else max(views, key=lambda v: v.cell_count).instances
            write_png(args.out / f"{tile.tile_id}_zoom.png",
                      render_zoom(image, views, densest_box(anchor, args.zoom), reference=reference, scores=scores,
                                  panel=args.zoom_panel))
        for view in views:
            geometry_rows.append({"tile_id": tile.tile_id, **view.geometry, "model": view.name, "cells": view.cell_count})
        table = agreement_table(views)
        for a in table.index:
            for b in table.columns:
                if a != b:
                    agreement_rows.append({"tile_id": tile.tile_id, "model": a, "reference": b, "pq": table.loc[a, b]})

        if args.context_margin:
            results = [context_effect(name, seg, image, args.context_margin, mpp=mpp, band=args.band, amp=not args.no_amp)
                       for name, seg in segmenters.items()]
            write_png(args.out / f"{tile.tile_id}_context.png",
                      render_context(image, args.context_margin, results, panel=args.panel))
            context_rows += [{"tile_id": tile.tile_id, **row} for row, _, _ in results]
        logger.info("%s done", tile.tile_id)

    args.out.mkdir(parents=True, exist_ok=True)
    geometry = pd.DataFrame(geometry_rows)
    geometry.drop(columns=["tile_id", "cells"]).drop_duplicates().to_csv(args.out / "geometry.csv", index=False)
    geometry.to_csv(args.out / "geometry_per_tile.csv", index=False)
    pd.DataFrame(agreement_rows).to_csv(args.out / "agreement.csv", index=False)
    if score_rows:
        table = pd.DataFrame(score_rows)
        table.to_csv(args.out / "reference_pq.csv", index=False)
        print("mean PQ against the reference:")
        print(table.groupby("model")[["pq", "dq", "sq", "cells"]].mean().round(3).to_string())
    if context_rows:
        context = pd.DataFrame(context_rows)
        context.to_csv(args.out / "context_effect.csv", index=False)
        summary = summarize_context(context)
        summary.to_csv(args.out / "context_summary.csv", index=False)
        print(summary.round(3).to_string(index=False))
    if agreement_rows:
        mean = pd.DataFrame(agreement_rows).pivot_table(index="model", columns="reference", values="pq")
        print("\nmean pairwise PQ (row = prediction, column = reference):")
        print(mean.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
