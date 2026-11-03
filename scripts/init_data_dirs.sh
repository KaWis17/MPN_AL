#!/usr/bin/env bash
# Create the local data/ tree (gitignored). Safe to re-run.
# Numbers follow pipeline order. See docs/architecture/data-layout.md.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)/data"
prep_id="${PREP_ID:-prep_example}"
round_id="${ROUND_ID:-round_00}"
dataset_id="${DATASET_ID:-dataset_v00}"
slide_id="${SLIDE_ID:-slide_id}"

mkdir -p \
  "$root/00_raw/wsi" \
  "$root/00_raw/clinical" \
  "$root/01_preprocessing/runs/${prep_id}/slides/${slide_id}/tissue" \
  "$root/01_preprocessing/runs/${prep_id}/slides/${slide_id}/qc" \
  "$root/02_active_learning/rounds/${round_id}" \
  "$root/02_active_learning/packages" \
  "$root/03_predictions/${round_id}/by_tile" \
  "$root/04_annotations/protocol_versions" \
  "$root/04_annotations/round_00_initial_manual/by_tile" \
  "$root/04_annotations/${round_id}_reviewed/by_tile" \
  "$root/05_targets/${dataset_id}/by_tile" \
  "$root/06_models/fm_weights" \
  "$root/06_models/${round_id}" \
  "$root/07_manifests/datasets" \
  "$root/08_evaluation/${round_id}"

echo "Created data layout under $root"
echo "  example prep_id=${prep_id}  round_id=${round_id}  dataset_id=${dataset_id}"
