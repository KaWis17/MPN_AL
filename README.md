# MPN Ph-negative Active Learning

Active learning pipeline for multi-class cell instance segmentation on bone-marrow trephine WSIs in Ph-negative MPN (PV, ET, PMF), with QuPath review and training on PLGrid.

The repository holds **documentation, source code, configs, and schemas**. Working data (WSIs, masks, annotations, models, predictions) is **not committed**; it lives under a local or PLGrid `data/` tree that matches the layout below. That directory is listed in `.gitignore`. Create it with:

```bash
scripts/init_data_dirs.sh
```

## Repository structure

```text
MPN_AL/
  docs/
    architecture/          # system architecture
    meetings/              # meeting notes
  src/
    backend/               # shared backend and state
    image_backend/         # tile read/cache (1024×1024 @ 40×), fat/bone filters
    embeddings/            # appearance embeddings (e.g. H-optimus, UNI2-h)
    preprocessing/         # tissue/QC masks, tile grid
    segmenter/             # cell instance masks and error signals
    selector/              # tile selection (uncertainty, rarity, cost, dedup)
    task_manager/          # queries: tile + instructions + status
    annotation_validation/ # geometry checks, versions, ignore vs ground truth
    training/              # fine-tune on dataset_vN
    evaluation/            # held-out validation and test
    model_registry/        # model versions, configs, metrics
  tools/
    qupath_converter/      # QuPath package sync with pathologist
  configs/
    labels/
    selection/
    training/
  schemas/
  scripts/
  containers/
```

## Data layout (not in Git)

Expected path on disk / group storage. Create with `scripts/init_data_dirs.sh`.

```text
data/
  00_raw/
    wsi/                   # de-identified WSI pyramids
    clinical/              # pseudonymised clinical table
  01_annotations/
    round_00_initial_manual/
    round_NN_reviewed/
    protocol_versions/
  02_targets/
    dataset_vNN/           # instance / type / ignore masks for training
  03_active_learning/
    round_NN/              # scores, selected tiles, embeddings, time logs
    packages/              # QuPath packages (model + doctor export)
  04_models/
    round_NN/
    fm_weights/
  05_predictions/
    round_NN_for_review/   # AI suggestions before review
  06_manifests/            # splits, dataset manifests, checksums
  07_evaluation/           # metrics and learning curves
  08_preprocessing/
    <slide_id>/
      tissue/
      qc/
      tiles.csv
```
