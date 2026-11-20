"""Segment cells in tiles with an SSL base model.

Examples::

    # the base-model segmenter: frozen encoder + trained head (the head knows its encoder)
    python scripts/segment_cells.py --head data/06_models/round_01/run_x/head.pt \\
        --tiles-parquet data/01_preprocessing/runs/<prep_id>/tiles.parquet \\
        --slides-parquet data/00_raw/slides.parquet --out data/03_predictions/round_01

    # before any head exists: label-free nucleus baseline, cells embedded by the base model
    python scripts/segment_cells.py --baseline stain --base-model phikon-v2 \\
        --images tiles_png/ --mpp 0.25 --out outputs/baseline
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from ..embeddings.cli import add_model_arguments, model_from_arguments
from ..embeddings.provenance import run_provenance, write_json
from ..image_backend.cli_args import add_source_arguments, source_from_arguments
from .geojson import OBJECT_TYPES
from .labels import LabelSchema
from .model import load_segmenter, read_head
from .pipeline import PipelineConfig, RunWriter, SegmentationPipeline, Stopwatch
from .postprocess import PostprocessConfig

logger = logging.getLogger("mpn_al.segmenter")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segment cells in tiles with an SSL base model; writes QuPath GeoJSON and per-cell embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples::")[1],
    )
    method = parser.add_argument_group("segmentation method (give --head or --baseline)")
    method.add_argument("--head", type=Path, help="head checkpoint from train_segmentation_head.py")
    method.add_argument(
        "--baseline",
        choices=("stain",),
        help="label-free fallback: haematoxylin-based nucleus candidates (no trained head needed)",
    )
    method.add_argument("--labels", type=Path, help="with --baseline: label schema whose first class names the objects")
    add_model_arguments(parser, required=False)
    add_source_arguments(parser)

    output = parser.add_argument_group("output")
    output.add_argument("--out", type=Path, required=True, help="run directory, e.g. data/03_predictions/<round_id>")
    output.add_argument("--object-type", choices=OBJECT_TYPES, default="annotation", help="QuPath object type of the polygons")
    output.add_argument("--no-cell-embeddings", action="store_true", help="do not store an embedding per cell")
    output.add_argument("--save-masks", action="store_true", help="also save the instance map of every tile (masks/<tile_id>.npz)")
    output.add_argument("--skip-existing", action="store_true", help="skip tiles whose outputs are already in --out")

    tuning = parser.add_argument_group("inference settings")
    tuning.add_argument("--max-size", type=int, default=1024, help="largest side in px the segmenter sees in one pass")
    tuning.add_argument("--overlap", type=int, default=128, help="overlap in px between passes on larger images")
    tuning.add_argument("--prob-threshold", type=float, default=0.5, help="cell probability threshold")
    tuning.add_argument("--min-area-um2", type=float, default=3.0, help="drop instances smaller than this")
    tuning.add_argument("--embed-mpp", type=float, default=0.5, help="with --baseline: resolution of the embedding pass")
    tuning.add_argument("--num-workers", type=int, default=0, help="processes reading tiles")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if (args.head is None) == (args.baseline is None):
        parser.error("give exactly one of --head and --baseline")

    source, tiles = source_from_arguments(args)
    if not tiles:
        logger.info("shard %d of %d has no tiles: nothing to do", args.shard_index, args.num_shards)
        return 0
    config = PipelineConfig(
        default_mpp=args.mpp,
        amp=not args.no_amp,
        max_size=args.max_size,
        overlap=args.overlap,
        embed_mpp=args.embed_mpp,
        cell_embeddings=not args.no_cell_embeddings,
        postprocess=PostprocessConfig(prob_threshold=args.prob_threshold, min_area_um2=args.min_area_um2),
    )
    head_info = None
    if args.head is not None:
        base = model_from_arguments(args) if args.base_model else None
        segmenter = load_segmenter(
            args.head,
            base=base,
            encoder_checkpoint=args.checkpoint,
            pretrained=not args.random_init,
            device=args.device,
        )
        pipeline = SegmentationPipeline(segmenter=segmenter, config=config)
        payload = read_head(args.head)
        head_info = {
            "path": str(args.head),
            "sha256": _sha256(args.head),
            "seg_mpp": segmenter.seg_mpp,
            "layers": list(segmenter.layers),
            "decoder": payload["decoder"],
            "extra": payload.get("extra"),
        }
        logger.info(
            "head %s on %s: classes %s, working resolution %s",
            args.head.name, segmenter.base.name, [c.name for c in segmenter.labels.classes],
            f"{segmenter.seg_mpp} um/px" if segmenter.seg_mpp else "native",
        )
    else:
        base = model_from_arguments(args) if args.base_model else None
        if base is None and config.cell_embeddings:
            logger.info("no --base-model given: the baseline will not produce cell embeddings")
        labels = LabelSchema.from_yaml(args.labels) if args.labels else None
        pipeline = SegmentationPipeline(base_model=base, labels=labels, config=config)

    shard = f".shard-{args.shard_index:03d}-of-{args.num_shards:03d}" if args.num_shards > 1 else ""
    base_name = pipeline.base_model.name if pipeline.base_model is not None else None
    writer = RunWriter(
        args.out,
        labels=pipeline.labels,
        base_model=base_name,
        object_type=args.object_type,
        save_masks=args.save_masks,
        stats_name=f"tile_stats{shard}.jsonl",
    )
    if args.skip_existing:
        remaining = [tile for tile in tiles if not writer.is_done(tile.tile_id)]
        logger.info("%d of %d tiles already done, skipping them", len(tiles) - len(remaining), len(tiles))
        tiles = remaining
    logger.info("%d tiles, method %s, writing to %s", len(tiles), pipeline.method, args.out)

    from tqdm import tqdm

    loader = DataLoader(_TileDataset(source, tiles), batch_size=None, shuffle=False, num_workers=args.num_workers)
    n_cells, failed = 0, []
    started = time.perf_counter()
    for index, image in tqdm(loader, total=len(tiles), unit="tile"):
        tile = tiles[index]
        try:
            with Stopwatch() as watch:
                result = pipeline.run_tile(image.numpy(), tile)
            writer.write(result, seconds=watch.seconds)
            n_cells += len(result.cells)
        except Exception:  # noqa: BLE001 - one bad tile must not end a long job
            logger.exception("tile %s failed", tile.tile_id)
            failed.append(tile.tile_id)
    elapsed = time.perf_counter() - started

    write_json(
        args.out / f"run{shard}.json",
        run_provenance(
            vars(args),
            method=pipeline.method,
            base_model=pipeline.base_model.describe() if pipeline.base_model is not None else None,
            head=head_info,
            labels=pipeline.labels.to_dict(),
            config=asdict(config),
            tiles=len(tiles),
            cells=n_cells,
            failed_tiles=failed,
            seconds=round(elapsed, 1),
        ),
    )
    logger.info(
        "%d tiles, %d cells in %.1fs (%.2fs/tile)%s",
        len(tiles) - len(failed), n_cells, elapsed, elapsed / max(len(tiles), 1),
        f"; {len(failed)} tiles FAILED, see the log" if failed else "",
    )
    return 1 if failed else 0


class _TileDataset(Dataset):
    """Item ``i`` is ``(i, image)`` with the tile as a ``[H, W, 3]`` uint8 tensor."""

    def __init__(self, source, tiles):
        self.source = source
        self.tiles = tiles

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int):
        return index, torch.from_numpy(self.source.read(self.tiles[index]).copy())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
