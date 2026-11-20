#!/bin/bash
# Creates (or updates) the Python environment of MPN_AL on PLGrid. Run it on the LOGIN node: it needs internet.
#
#     bash scripts/plgrid/remote.sh setup        # from your own machine
#     bash scripts/plgrid/remote_setup.sh        # or on the cluster, from the repository root
#
# Optional environment variables:
#   MPN_AL_TORCH_SPEC   pip requirement for torch, e.g. "torch==2.13.0" (default: the latest release)
#   MPN_AL_TORCH_INDEX  PyTorch wheel index, e.g. https://download.pytorch.org/whl/cu128 - use it when the GPU
#                       driver is too old for the default CUDA build (the smoke test then reports no CUDA device)
#   MPN_AL_EXTRAS       extras of the package to install (default: hf,wsi,dev)
#
# AL_System, which ran on the same A100 nodes, locks torch 2.13.0 with CUDA 13 runtime wheels, so the default
# build should work there; this is checked by the smoke test, not assumed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

source scripts/plgrid/setup_env.sh

if [ ! -f "$MPN_AL_VENV/bin/activate" ]; then
    echo "Creating the virtual environment $MPN_AL_VENV"
    python -m venv "$MPN_AL_VENV"
    # shellcheck disable=SC1091
    source "$MPN_AL_VENV/bin/activate"  # setup_env.sh could not activate what did not exist yet
fi

python -m pip install --upgrade pip

if [ -n "${MPN_AL_TORCH_SPEC:-}" ] || [ -n "${MPN_AL_TORCH_INDEX:-}" ]; then
    python -m pip install ${MPN_AL_TORCH_INDEX:+--index-url "$MPN_AL_TORCH_INDEX"} "${MPN_AL_TORCH_SPEC:-torch}" torchvision
fi
python -m pip install -e ".[${MPN_AL_EXTRAS:-hf,wsi,dev}]"

python - <<'PY'
import importlib

import torch

print(f"python {__import__('platform').python_version()}, torch {torch.__version__} "
      f"(CUDA runtime {torch.version.cuda}; no GPU is visible on a login node: cuda_available={torch.cuda.is_available()})")
for name in ("timm", "transformers", "huggingface_hub", "cv2", "pandas", "pyarrow", "openslide", "mpn_al"):
    try:
        module = importlib.import_module(name)
        print(f"  {name:16s} {getattr(module, '__version__', 'ok')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {name:16s} MISSING ({type(exc).__name__}: {exc})")
PY
echo "Environment ready: $MPN_AL_VENV"
