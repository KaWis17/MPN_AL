# MPN Ph-negative Active Learning

Active learning pipeline for multi-class cell instance segmentation on bone-marrow trephine WSIs in Ph-negative MPN (PV, ET, PMF), with QuPath review and training on PLGrid.

The repository holds **documentation, source code, configs, and schemas**. Working data is **not committed** (see `.gitignore`). Create the local tree with:

```bash
scripts/init_data_dirs.sh
```

Per-step storage and IDs: [docs/architecture/data-layout.md](docs/architecture/data-layout.md).

## SSL base models: embeddings and cell segmentation

Self-supervised pathology foundation models (UNI2-h, H-optimus, Virchow2, Phikon-v2, ...) are
wrapped behind one interface. They provide the hidden representations of tiles and cells and,
with a small trained head on top of the frozen encoder, the segmented cells. Plan, candidate
models, licences and status (in Polish): [docs/architecture/ssl-base-models.md](docs/architecture/ssl-base-models.md).

```bash
pip install -e ".[hf,dev]"      # add ",wsi" to read tiles straight from WSIs
pytest                          # offline, about two minutes on a CPU
```

| Script | Purpose |
|---|---|
| `scripts/list_base_models.py` | List the registered base models; `--check <model>` loads one and runs test forwards. |
| `scripts/extract_embeddings.py` | One embedding per tile → `embeddings.parquet` for the selector. |
| `scripts/segment_cells.py` | Segment cells with a base model → QuPath GeoJSON, per-cell embeddings, per-tile statistics. |
| `scripts/train_segmentation_head.py` | Train the segmentation head on annotations; the encoder stays frozen. |
| `scripts/download_base_models.py` | Fetch base-model weights into the Hugging Face cache (login nodes have internet, compute nodes do not). |
| `scripts/merge_shards.py` | Merge the per-task files of a Slurm job array (embeddings, tile statistics). |
| `scripts/make_demo_dataset.py` | Synthetic tiles with annotations, to try the pipeline without data. |
| `scripts/plgrid/remote.sh` | Run everything on PLGrid from your own machine: `sync`, `setup`, `download`, `smoke`, `submit`, `run`, `status`, `logs`, `put`, `fetch`. See [docs/architecture/plgrid.md](docs/architecture/plgrid.md). |
| `scripts/plgrid/smoke_test.py` | End-to-end check of the chain on synthetic data (GPU or CPU); what `remote.sh smoke` runs on a node. |

```bash
python scripts/list_base_models.py
python scripts/extract_embeddings.py --base-model phikon-v2 --images tiles_png --mpp 0.25 --out outputs/embeddings.parquet
python scripts/train_segmentation_head.py --base-model h-optimus-0 --images tiles_png --annotations annotations_geojson --labels configs/labels/megakaryocyte_binary.yaml --mpp 0.25 --val-fraction 0.2 --out data/06_models/round_00/run_001
python scripts/segment_cells.py --head data/06_models/round_00/run_001/head.pt --images tiles_png --mpp 0.25 --out data/03_predictions/round_01
```

The PLGrid scripts are tested against a fake cluster, not yet on Athena itself; the first real run is
`bash scripts/plgrid/remote.sh smoke` (see [docs/architecture/plgrid.md](docs/architecture/plgrid.md)).

Until a head has been trained, `segment_cells.py --baseline stain` runs the same pipeline on
label-free nucleus candidates (a classical stain-based fallback, not a base-model segmentation).
Output formats are described in [schemas/](schemas/). Gated models need an approved Hugging
Face account; the scripts run from a plain checkout as well as after installation.

## Repository structure

```text
MPN_AL/
  docs/
    architecture/          # system + data architecture
    meetings/              # meeting notes
  src/                     # importable as the package `mpn_al` (see pyproject.toml)
    backend/
    image_backend/         # tile identity, reading tiles from image files or WSIs
    embeddings/            # SSL base models: registry, hidden representations, tile embeddings
    preprocessing/
    segmenter/             # segmentation head, instance decoding, GeoJSON, per-cell embeddings
    selector/
    task_manager/
    annotation_validation/
    training/              # training of the segmentation head
    evaluation/            # panoptic quality and related metrics
    model_registry/
  tools/
    qupath_converter/
  configs/
    base_models/           # additional base-model entries
    labels/                # label schemas
    selection/
    training/
  schemas/
  scripts/
    plgrid/
  tests/
  containers/
  pyproject.toml
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
