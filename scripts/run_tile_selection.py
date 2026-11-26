#!/usr/bin/env python
"""Compare base models as tile selectors for annotation (one slide); see mpn_al.evaluation.selection_cli.

    python scripts/run_tile_selection.py --tiles-parquet prep/tiles.parquet --images prep/slides/<id>/tiles \
        --embeddings lunit-dino-s16=runs/lunit/embeddings.parquet --out outputs/selection/<id>

Protocol: docs/architecture/tile-selection-experiment.md. Run with --help for all options.
"""

import _bootstrap  # noqa: F401  (makes `mpn_al` importable from a plain checkout)

from mpn_al.evaluation.selection_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
