#!/usr/bin/env python
"""Embed tiles with an SSL base model: one hidden-representation vector per tile.

    python scripts/extract_embeddings.py --base-model uni2-h \
        --tiles-parquet data/01_preprocessing/runs/<prep_id>/tiles.parquet \
        --slides-parquet data/00_raw/slides.parquet \
        --out data/02_active_learning/rounds/<round_id>/embeddings.parquet

    python scripts/extract_embeddings.py --base-model phikon-v2 --images tiles_png/ --mpp 0.25 \
        --out outputs/embeddings.parquet

Run with --help for the window modes (how a 1024 px tile is fed to a 224 px model).
"""

import _bootstrap  # noqa: F401  (makes `mpn_al` importable from a plain checkout)

from mpn_al.embeddings.cli import main_extract

if __name__ == "__main__":
    raise SystemExit(main_extract())
