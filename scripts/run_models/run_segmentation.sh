#!/bin/bash

#SBATCH --job-name=foundation-model
#SBATCH --account=plgaktywneuczenie-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=foundation_model_%j.out
#SBATCH --error=foundation_model_%j.err

set -euo pipefail

# --------------------------------------------------
# 1. Parse arguments
# --------------------------------------------------

MODEL="cellvit"
INPUT_DIR=""
OUTPUT_BASE="data/03_predictions/foundation_model_runs"

usage() {
    echo "Usage: sbatch run_segmentation.sh --input-dir DIR [--model MODEL] [--output-base DIR]"
    echo ""
    echo "Arguments:"
    echo "  --input-dir DIR      Directory containing PNG image tiles (required)"
    echo "  --model MODEL        pathosam|cellvit|cellsam|cellpose_sam|sam"
    echo "  --output-base DIR    Base output directory"
    echo "  --help               Show this help"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir)
            [[ $# -ge 2 ]] || { echo "ERROR: Missing input directory"; exit 1; }
            INPUT_DIR="$2"
            shift 2
            ;;
        --model)
            [[ $# -ge 2 ]] || { echo "ERROR: Missing model name"; exit 1; }
            MODEL="$2"
            shift 2
            ;;
        --output-base)
            [[ $# -ge 2 ]] || { echo "ERROR: Missing output directory"; exit 1; }
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

# --------------------------------------------------
# 2. Select virtual environment
# --------------------------------------------------

PROJECT_DIR="$SCRATCH/MPN_AL"

case "$MODEL" in
    pathosam) VENV_PATH="$PROJECT_DIR/.pathosam_venv" ;;
    cellvit) VENV_PATH="$PROJECT_DIR/.cellvit_venv" ;;
    cellsam) VENV_PATH="$PROJECT_DIR/.cellsam_venv" ;;
    cellpose_sam) VENV_PATH="$PROJECT_DIR/.cellpose_venv" ;;
    sam) VENV_PATH="$PROJECT_DIR/.sam_venv" ;;
    *)
        echo "ERROR: Invalid model: $MODEL"
        exit 1
        ;;
esac

# --------------------------------------------------
# 3. Load environment
# --------------------------------------------------

module purge
module load GCCcore/13.3.0
module load Python/3.12.3

cd "$PROJECT_DIR"

if [[ ! -f "$VENV_PATH/bin/python" ]]; then
    echo "ERROR: Python environment not found: $VENV_PATH"
    exit 1
fi

source "$VENV_PATH/bin/activate"

PYTHON="$VENV_PATH/bin/python"

echo "========== PYTHON ENVIRONMENT =========="
echo "Model: $MODEL"
echo "Python: $PYTHON"

"$PYTHON" --version

# --------------------------------------------------
# 4. GPU information
# --------------------------------------------------

echo "========== GPU INFORMATION =========="

nvidia-smi

"$PYTHON" -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

# --------------------------------------------------
# 5. Validate input
# --------------------------------------------------

cd "$SLURM_SUBMIT_DIR"

if [[ -z "$INPUT_DIR" ]]; then
    echo "ERROR: --input-dir is required"
    usage
    exit 1
fi

if [[ "$INPUT_DIR" != /* ]]; then
    INPUT_DIR="$SLURM_SUBMIT_DIR/$INPUT_DIR"
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    exit 1
fi

shopt -s nullglob
INPUT_FILES=( "$INPUT_DIR"/*.png )

TILE_COUNT=${#INPUT_FILES[@]}

if [[ "$TILE_COUNT" -eq 0 ]]; then
    echo "ERROR: No PNG tiles found in $INPUT_DIR"
    exit 1
fi

# --------------------------------------------------
# 6. Prepare output
# --------------------------------------------------

if [[ "$OUTPUT_BASE" != /* ]]; then
    OUTPUT_BASE="$SLURM_SUBMIT_DIR/$OUTPUT_BASE"
fi

JOB_ID="${SLURM_JOB_ID:-manual}"

RUN_DIR="$OUTPUT_BASE/$MODEL/$JOB_ID"

mkdir -p "$RUN_DIR"

echo "========== RUN CONFIGURATION =========="
echo "Model: $MODEL"
echo "Input: $INPUT_DIR"
echo "Output: $RUN_DIR"
echo "Tiles: $TILE_COUNT"
echo "Job ID: $JOB_ID"

# --------------------------------------------------
# 7. Check existing output directories
# --------------------------------------------------

for INPUT in "${INPUT_FILES[@]}"; do

    TILE_NAME="$(basename "${INPUT%.*}")"

    if [[ -e "$RUN_DIR/$TILE_NAME" ]]; then
        echo "ERROR: Output already exists: $RUN_DIR/$TILE_NAME"
        exit 1
    fi

done

# --------------------------------------------------
# 8. Process all tiles in ONE Python process
# --------------------------------------------------

echo ""
echo "========== STARTING $MODEL =========="
echo "Loading model once and processing $TILE_COUNT tiles."

TILE_LIST="$RUN_DIR/tile_list.txt"

printf '%s\n' "${INPUT_FILES[@]}" > "$TILE_LIST"

START_TIME=$(date +%s)

srun "$PYTHON" -u -m src.model_registry.run_foundation_model \
    --model "$MODEL" \
    --input-list "$TILE_LIST" \
    --output-base "$RUN_DIR" \
    --device auto

EXIT_CODE=$?

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "========== JOB SUMMARY =========="
echo "=========================================="

echo "Model: $MODEL"
echo "Total tiles: $TILE_COUNT"
echo "Total time: $((TOTAL_TIME / 3600))h $(((TOTAL_TIME % 3600) / 60))m $((TOTAL_TIME % 60))s"
echo "Output directory: $RUN_DIR"

if [[ "$EXIT_CODE" -ne 0 ]]; then
    echo "ERROR: Segmentation process failed."
    exit "$EXIT_CODE"
fi

echo "========== $MODEL FINISHED SUCCESSFULLY =========="

