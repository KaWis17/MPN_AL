#!/bin/bash
# Environment for MPN_AL on PLGrid (Athena). Source it on the login node and at the top of every job:
#
#     source scripts/plgrid/setup_env.sh
#
# $HOME has a small quota (10 GB), so the virtual environment and all caches live on $SCRATCH. $SCRATCH
# is purged periodically: it is a cache, not an archive. Results worth keeping belong on group storage.
#
# Variables you may set before sourcing:
#   MPN_AL_HF_HOME  Hugging Face cache (default $SCRATCH/hf-cache). Gated models are licensed per person:
#                   share a cache holding them only with people who accepted the model's terms themselves.
#   MPN_AL_VENV     virtual environment (default $SCRATCH/mpn_al-venv)
#   MPN_AL_RUNS     where jobs write their results (default $SCRATCH/mpn_al_runs)
#
# First-time setup: `bash scripts/plgrid/remote_setup.sh` (or `remote.sh setup` from your own machine).

if command -v module >/dev/null 2>&1; then
    module load Python/3.10.4
else
    echo "setup_env.sh: no 'module' command here (not a PLGrid node?); keeping the current Python" >&2
fi

_mpn_scratch="${SCRATCH:-$HOME}"
export HF_HOME="${MPN_AL_HF_HOME:-$_mpn_scratch/hf-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$_mpn_scratch/pip-cache}"
export MPN_AL_VENV="${MPN_AL_VENV:-$_mpn_scratch/mpn_al-venv}"
export MPN_AL_RUNS="${MPN_AL_RUNS:-$_mpn_scratch/mpn_al_runs}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$MPN_AL_RUNS"
unset _mpn_scratch

if [ -f "$MPN_AL_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$MPN_AL_VENV/bin/activate"
else
    echo "setup_env.sh: no virtual environment at $MPN_AL_VENV yet; run scripts/plgrid/remote_setup.sh" >&2
fi

echo "HF_HOME=$HF_HOME  MPN_AL_VENV=$MPN_AL_VENV  python=$(command -v python || echo none)"
