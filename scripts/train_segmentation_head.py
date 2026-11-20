#!/usr/bin/env python
"""Train a cell-segmentation head on a frozen SSL base model.

    python scripts/train_segmentation_head.py --base-model h-optimus-0 \
        --images tiles_png/ --annotations data/04_annotations/round_00_initial_manual/by_tile \
        --labels configs/labels/megakaryocyte_binary.yaml --mpp 0.25 --val-fraction 0.2 \
        --out data/06_models/round_00/run_001

Run with --help for all options.
"""

import _bootstrap  # noqa: F401  (makes `mpn_al` importable from a plain checkout)

from mpn_al.training.train_head import main

if __name__ == "__main__":
    raise SystemExit(main())
