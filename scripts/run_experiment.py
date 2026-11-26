#!/usr/bin/env python
"""Run one cell of an experiment grid (see configs/experiments/ and docs/architecture/experiment-plan.md).

    python scripts/run_experiment.py --grid configs/experiments/pilot.yaml --list
    python scripts/run_experiment.py --grid configs/experiments/pilot.yaml --task-index 0 \
        --images tiles --masks masks --split-file split.csv --mpp 0.25 --out outputs/experiments

On PLGrid every array task runs one cell:
    bash scripts/plgrid/remote.sh submit --array=0-N scripts/run_experiment.py --grid ... --skip-existing
The test split is evaluated only with --final.
"""

import _bootstrap  # noqa: F401  (makes `mpn_al` importable from a plain checkout)

from mpn_al.experiment.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
