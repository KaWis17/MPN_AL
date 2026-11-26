#!/usr/bin/env python
"""Draw the results of cluster_embeddings.py (curves, maps, matrices, galleries, QuPath files); see mpn_al.evaluation.cluster_viz_cli.

    python scripts/visualize_clusters.py --cluster-dir outputs/clustering/bambo --embeddings phikon-v2=a/embeddings.parquet \
        --meta outputs/bambo_meta.csv --images outputs/bambo_png --nuisance-col image_size --k 4

Needs matplotlib (and optionally umap-learn): pip install -e ".[viz]". Run with --help for all options.
"""

import _bootstrap  # noqa: F401  (makes `mpn_al` importable from a plain checkout)

from mpn_al.evaluation.cluster_viz_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
