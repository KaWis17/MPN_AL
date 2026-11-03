#!/usr/bin/env bash
# Create the local data/ tree (gitignored). Safe to re-run.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)/data"

mkdir -p \
  "$root/00_raw/wsi" \
  "$root/00_raw/clinical" \
  "$root/01_annotations/round_00_initial_manual" \
  "$root/01_annotations/round_NN_reviewed" \
  "$root/01_annotations/protocol_versions" \
  "$root/02_targets/dataset_vNN" \
  "$root/03_active_learning/round_NN" \
  "$root/03_active_learning/packages/round_NN_model" \
  "$root/03_active_learning/packages/round_NN_model__doctor__YYYYMMDD_HHMMSS" \
  "$root/04_models/round_NN" \
  "$root/04_models/fm_weights" \
  "$root/05_predictions/round_NN_for_review" \
  "$root/06_manifests" \
  "$root/07_evaluation" \
  "$root/08_preprocessing/slide_id/tissue" \
  "$root/08_preprocessing/slide_id/qc"

echo "Created data layout under $root"
