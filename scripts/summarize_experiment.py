#!/usr/bin/env python
"""Summarise an experiment directory into runs.csv, curves.csv, comparison.csv and report.md.

    python scripts/summarize_experiment.py outputs/experiments --split validation --reference dinov2-s14
"""

import _bootstrap  # noqa: F401  (makes `mpn_al` importable from a plain checkout)

from mpn_al.experiment.aggregate import main

if __name__ == "__main__":
    raise SystemExit(main())
