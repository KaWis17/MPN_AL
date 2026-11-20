#!/usr/bin/env python
"""Merge the shard files written by the tasks of a Slurm job array.

    python scripts/merge_shards.py embeddings data/02_active_learning/rounds/round_01/embeddings.parquet
    python scripts/merge_shards.py tile-stats data/03_predictions/round_01
"""

import _bootstrap  # noqa: F401  (makes `mpn_al` importable from a plain checkout)

from mpn_al.embeddings.merge import main

if __name__ == "__main__":
    raise SystemExit(main())
