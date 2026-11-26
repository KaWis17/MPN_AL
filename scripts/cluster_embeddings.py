#!/usr/bin/env python
"""Cluster the embeddings of several base models (KMeans over k and seeds) and compare them; see mpn_al.evaluation.cluster_cli.

    python scripts/cluster_embeddings.py --embeddings phikon-v2=a/embeddings.parquet --embeddings lunit-dino-s16=b/embeddings.parquet \
        --meta outputs/bambo_split.csv --out outputs/clustering/bambo --noise-control

Run with --help for all options.
"""

import _bootstrap  # noqa: F401  (makes `mpn_al` importable from a plain checkout)

from mpn_al.evaluation.cluster_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
