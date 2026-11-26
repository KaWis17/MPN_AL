#!/usr/bin/env python
"""Segment cells in tiles with an SSL base model.

Writes, per tile, a QuPath GeoJSON (level-0 slide coordinates) and a table of cells with the
base model's embedding of each cell.

    # base-model segmenter: frozen encoder + trained head
    python scripts/segment_cells.py --head data/06_models/<round_id>/<run_id>/head.pt \
        --tiles-parquet data/01_preprocessing/runs/<prep_id>/tiles.parquet \
        --slides-parquet data/00_raw/slides.parquet --out data/03_predictions/<round_id>

    # before any head exists: label-free nucleus baseline + base-model cell embeddings
    python scripts/segment_cells.py --baseline stain --base-model phikon-v2 \
        --images tiles_png/ --mpp 0.25 --out outputs/baseline

Run with --help for all options.
"""

import _bootstrap  # noqa: F401  (makes `mpn_al` importable from a plain checkout)

from mpn_al.segmenter.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
