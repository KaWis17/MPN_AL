#!/bin/bash
#SBATCH --job-name=mpn-al
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#
# Runs one of the scripts in scripts/ on a GPU node. Normally submitted through remote.sh, which adds the
# grant (-A), the partition (-p) and the log file:
#
#     bash scripts/plgrid/remote.sh submit scripts/extract_embeddings.py --base-model phikon-v2 ...
#
# or by hand, from ~/mpn_al on the cluster:
#
#     sbatch -A "$PLG_GRANT" -p "$PLG_PARTITION" --output=logs/%x-%j.out scripts/plgrid/run_gpu.sbatch SCRIPT ARGS
#
# Job arrays (numbered 0..N-1) split the tiles over the tasks: every task gets --shard-index and --num-shards
# for extract_embeddings.py and segment_cells.py. Embedding tasks write one file each
# (embeddings.shard-III-of-NNN.parquet): merge them afterwards with scripts/merge_shards.py. Resubmitting
# with --skip-existing redoes only the shards that are missing.
#
# Weights must already be in the Hugging Face cache (scripts/download_base_models.py on the login node):
# compute nodes have no internet, so the job runs with HF_HUB_OFFLINE=1.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/plgrid/setup_env.sh
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

script="${1:?usage: run_gpu.sbatch SCRIPT [ARGUMENTS]}"
shift
cpus="${SLURM_CPUS_PER_TASK:-4}"
workers=$(( cpus > 2 ? cpus - 2 : 0 ))

# Options that only some scripts know: add them per script.
extra=()
case "$(basename "$script")" in
    extract_embeddings.py|segment_cells.py)
        extra+=(--num-workers "$workers")
        if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
            extra+=(--shard-index "$SLURM_ARRAY_TASK_ID" --num-shards "${SLURM_ARRAY_TASK_COUNT:?job arrays must be numbered 0..N-1}")
        fi
        ;;
    train_segmentation_head.py)
        extra+=(--num-workers "$workers")
        ;;
esac

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "running: python $script $* ${extra[*]:-}"
exec python "$script" "$@" ${extra[@]+"${extra[@]}"}
