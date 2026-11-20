#!/bin/sh

#SBATCH --job-name=foundation-model
#SBATCH --account=plgaktywneuczenie-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=foundation_model_%j.out
#SBATCH --error=foundation_model_%j.err

set -euo pipefail

# --------------------------------------------------
# 1. Load Python module
# --------------------------------------------------

module purge
module load GCCcore/13.3.0
module load Python/3.12.3

# --------------------------------------------------
# 2. Set project and virtual environment paths
# --------------------------------------------------

PROJECT_DIR="$SCRATCH/MPN_AL"
VENV_PATH="$PROJECT_DIR/.venv"

cd "$PROJECT_DIR"

if [ ! -f "$VENV_PATH/bin/python" ]; then
    echo "ERROR: Python environment not found: $VENV_PATH"
    exit 1
fi

# --------------------------------------------------
# 3. Use the virtual environment
# --------------------------------------------------

source "$VENV_PATH/bin/activate"

PYTHON="$VENV_PATH/bin/python"

echo "========== PYTHON ENVIRONMENT =========="
echo "Python executable: $PYTHON"
"$PYTHON" --version
"$PYTHON" -m pip show torch

# --------------------------------------------------
# 4. Check GPU
# --------------------------------------------------

echo "========== GPU INFORMATION =========="

nvidia-smi

echo "========== PYTORCH INFORMATION =========="

"$PYTHON" -c "
import torch

print('PyTorch:', torch.__version__)
print('CUDA compiled version:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU count:', torch.cuda.device_count())
else:
    print('WARNING: CUDA is not available!')
"

# --------------------------------------------------
# 5. Select model
# --------------------------------------------------

MODEL="${MODEL:-cellpose_sam}"

case "$MODEL" in
    pathosam|cellvit|cellsam|cellpose_sam|sam)
        ;;
    *)
        echo "ERROR: Invalid model: $MODEL"
        exit 1
        ;;
esac

# --------------------------------------------------
# 6. Input and output
# --------------------------------------------------

cd "$SLURM_SUBMIT_DIR"

INPUT="${INPUT:-src/model_registry/test_data/tiles/testZL26_x4608_y34304.png}"

OUTPUT_BASE="${OUTPUT_BASE:-data/03_predictions/foundation_model_runs}"

if [ ! -f "$INPUT" ]; then
    echo "ERROR: Input image not found: $INPUT"
    exit 1
fi

mkdir -p "$OUTPUT_BASE"

RUN_DIR="$OUTPUT_BASE/${MODEL}-${SLURM_JOB_ID}"

echo "Output directory: $RUN_DIR"

# --------------------------------------------------
# 7. Run model
# --------------------------------------------------

echo "========== STARTING $MODEL =========="

srun "$PYTHON" -u -m src.model_registry.run_foundation_model \
    --model "$MODEL" \
    --input "$INPUT" \
    --device auto \
    --output "$RUN_DIR"

echo "========== $MODEL FINISHED =========="
