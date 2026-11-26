#!/bin/bash
#SBATCH --job-name=mpn-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#
# GPU smoke test: scripts/plgrid/smoke_test.py (synthetic data, every requested model) followed by the unit
# tests. Results go to $MPN_AL_RUNS/smoke-<jobid>/ (smoke_report.md and .json, pytest.txt). Submitted by
#
#     bash scripts/plgrid/remote.sh smoke [--models lunit-dino-s16 phikon-v2 ...]
#
# which also waits for the job and fetches the report. The arguments are passed on to smoke_test.py; the
# weights of the models must already be in the cache (scripts/download_base_models.py on the login node).

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/plgrid/setup_env.sh || exit 1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

out="$MPN_AL_RUNS/smoke-${SLURM_JOB_ID:-manual}"
mkdir -p "$out"
nvidia-smi || true

python scripts/plgrid/smoke_test.py --out "$out" "$@"
smoke=$?

# The tests need no GPU and no network; they show that the Python environment on the node is complete.
python -m pytest tests -q -p no:cacheprovider > "$out/pytest.txt" 2>&1
tests=$?
tail -n 5 "$out/pytest.txt"

echo "smoke test exit code: $smoke, unit tests exit code: $tests"
[ "$smoke" -eq 0 ] && [ "$tests" -eq 0 ]
