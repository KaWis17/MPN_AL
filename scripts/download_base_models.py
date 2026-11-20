#!/usr/bin/env python
"""Download base-model weights into the Hugging Face cache (run on a machine with internet).

    python scripts/download_base_models.py phikon-v2 lunit-dino-s16 --verify
    python scripts/download_base_models.py --open

On PLGrid run it on the login node; the GPU jobs then read the cache offline.
"""

import _bootstrap  # noqa: F401  (makes `mpn_al` importable from a plain checkout)

from mpn_al.embeddings.download import main_download

if __name__ == "__main__":
    raise SystemExit(main_download())
