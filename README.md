# MPN Ph-negative Active Learning

Active learning pipeline for multi-class cell instance segmentation on bone-marrow trephine WSIs in Ph-negative MPN (PV, ET, PMF), with QuPath review and training on PLGrid.

The repository holds **documentation, source code, configs, and schemas**. Working data is **not committed** (see `.gitignore`). Create the local tree with:

```bash
scripts/init_data_dirs.sh
```

Per-step storage and IDs: [docs/architecture/data-layout.md](docs/architecture/data-layout.md).

## Repository structure

```text
MPN_AL/
  docs/
    architecture/          # system + data architecture
    meetings/              # meeting notes
  src/
    backend/
    image_backend/
    embeddings/
    preprocessing/
    segmenter/
    selector/
    task_manager/
    annotation_validation/
    training/
    evaluation/
    model_registry/
  tools/
    qupath_converter/
  configs/
    labels/
    selection/
    training/
  schemas/
  scripts/
  containers/
```

## Data layout (not in Git)

Pipeline order. Join key after preprocessing: **`tile_id`** (annotation FOV 1024×1024). Each preprocessing config is a frozen **`prep_id`** run.

```text
data/
  00_raw/
    wsi/<patient_id>/<slide_id>.ndpi
    clinical/
    slides.parquet
  01_preprocessing/
    runs/<prep_id>/
      config.yaml
      provenance.json
      tiles.parquet
      model_windows.parquet
      slides/<slide_id>/tissue|qc/
  02_active_learning/
    rounds/<round_id>/
    packages/
  03_predictions/
    <round_id>/by_tile/<tile_id>.geojson
  04_annotations/
    protocol_versions/
    round_00_initial_manual/by_tile/<tile_id>.geojson
    <round_id>_reviewed/by_tile/<tile_id>.geojson
  05_targets/
    <dataset_id>/by_tile/<tile_id>/
  06_models/
    fm_weights/
    <round_id>/<run_id>/
  07_manifests/
    patient_split_fixed.csv
    annotation_manifest.parquet
    datasets/<dataset_id>.parquet
  08_evaluation/
    <round_id_or_run_id>/
```
