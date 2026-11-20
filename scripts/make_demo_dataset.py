#!/usr/bin/env python
"""Write a small synthetic data set (tiles, QuPath-style annotations, instance masks).

    python scripts/make_demo_dataset.py outputs/demo --tiles 6 --size 256

It lets you try the whole pipeline without any real data, e.g.

    python scripts/train_segmentation_head.py --base-model lunit-dino-s16 --images outputs/demo/images \
        --annotations outputs/demo/annotations --mpp 0.25 --val-fraction 0.34 --out outputs/demo/run
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  (makes `mpn_al` importable from a plain checkout)

from mpn_al.training.synthetic import write_demo_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path, help="directory to write; created if needed")
    parser.add_argument("--tiles", type=int, default=6)
    parser.add_argument("--size", type=int, default=256, help="tile side in pixels")
    parser.add_argument("--cells", type=int, default=14, help="cells per tile")
    parser.add_argument("--mpp", type=float, default=0.25, help="resolution to record in dataset.json (um/px)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    tile_ids = write_demo_dataset(args.out, tiles=args.tiles, size=args.size, count=args.cells, mpp=args.mpp, seed=args.seed)
    print(f"wrote {len(tile_ids)} tiles to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
