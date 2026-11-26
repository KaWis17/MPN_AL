#!/usr/bin/env python
"""Patient-grouped linear/kNN probe on an embedding table (see mpn_al.evaluation.probe).

    python scripts/probe_embeddings.py cells.parquet --label class_name --group patient_id --center-by slide_id
"""

import _bootstrap  # noqa: F401  (makes `mpn_al` importable from a plain checkout)

from mpn_al.evaluation.probe import main

if __name__ == "__main__":
    raise SystemExit(main())
