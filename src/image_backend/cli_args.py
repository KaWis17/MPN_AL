"""Command-line options shared by every script that reads tiles."""

from __future__ import annotations

import argparse
from pathlib import Path

from .tiles import ImageFolderSource, TileRef, TileSource, WsiTileSource, select_tiles


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    """Options choosing where tiles come from and which of them to process."""
    group = parser.add_argument_group("input tiles (give --images or --tiles-parquet)")
    group.add_argument("--images", type=Path, help="image file, or directory of tile images named <tile_id>.<ext>")
    group.add_argument("--recursive", action="store_true", help="with --images: also search sub-directories")
    group.add_argument("--tiles-parquet", type=Path, help="tiles.parquet of a preprocessing run (tiles are read from the WSIs)")
    group.add_argument("--slides-parquet", type=Path, help="slides.parquet mapping slide_id to the WSI path")
    group.add_argument("--wsi-root", type=Path, help="directory searched for <slide_id>.* when slides.parquet has no path")
    group.add_argument("--split", help="with --tiles-parquet: keep one split (train / validation / test)")
    group.add_argument("--include-qc-fail", action="store_true", help="with --tiles-parquet: keep tiles with qc_pass = false")
    group.add_argument("--mpp", type=float, help="slide resolution in um/px for tiles that do not carry one (40x is about 0.25)")
    group.add_argument("--tile-ids", type=Path, help="text file with one tile_id per line; only these are processed")
    group.add_argument("--limit", type=int, help="process at most this many tiles")
    group.add_argument("--shard-index", type=int, default=0, help="which shard to process (for Slurm job arrays)")
    group.add_argument("--num-shards", type=int, default=1, help="number of shards the tile list is split into")


def source_from_arguments(args: argparse.Namespace) -> tuple[TileSource, list[TileRef]]:
    """Build the tile source and the list of tiles selected by the options above."""
    if (args.images is None) == (args.tiles_parquet is None):
        raise SystemExit("give exactly one of --images and --tiles-parquet")
    if args.images is not None:
        source: TileSource = ImageFolderSource(args.images, mpp=args.mpp, recursive=args.recursive)
    else:
        source = WsiTileSource(
            args.tiles_parquet,
            slides_parquet=args.slides_parquet,
            wsi_root=args.wsi_root,
            qc_pass_only=not args.include_qc_fail,
            split=args.split,
            default_mpp=args.mpp,
        )
    tile_ids = None
    if args.tile_ids is not None:
        tile_ids = [line.strip() for line in args.tile_ids.read_text(encoding="utf-8").splitlines() if line.strip()]
    candidates = select_tiles(source.tiles, tile_ids=tile_ids)
    if not candidates:
        raise SystemExit("no tiles left after filtering")
    # One shard may legitimately be empty (more array tasks than tiles): callers then have nothing to do.
    tiles = select_tiles(
        candidates, shard_index=args.shard_index, num_shards=args.num_shards, limit=args.limit
    )
    return source, tiles
