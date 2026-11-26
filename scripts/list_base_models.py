#!/usr/bin/env python
"""List the registered SSL base models, or load one and verify it end to end.

    python scripts/list_base_models.py                     # table of candidates
    python scripts/list_base_models.py --format markdown   # the same, for documents
    python scripts/list_base_models.py --check phikon-v2   # download, load, run test forwards
"""

import _bootstrap  # noqa: F401  (makes `mpn_al` importable from a plain checkout)

from mpn_al.embeddings.cli import main_models

if __name__ == "__main__":
    raise SystemExit(main_models())
