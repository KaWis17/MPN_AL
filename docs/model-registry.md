# Foundation model cell segmentation

This guide covers the tile inference harness under `src/model_registry/`. One
command runs one model on a PNG tile or one coordinate-selected NDPI tile.
Outputs are stored below `segment_result/<model>/result/<run-id>/` by default.
A `0.30` border adds 30% of the canonical width and height **on every side**;
the saved prediction is the centered 1024-by-1024 crop. This is not a
production validation claim: reviewed whole-cell masks and slide-held-out
metrics are still required.

## Run a model

From the repository root, run the common command template:

```bash
python -m src.model_registry.run_foundation_model \
  --model cellsam --input /path/to/1024x1024.png --device cuda
```

Replace `cellsam` with `pathosam`, `cellvit`, `cellpose_sam`, or `sam`. The
`--output` argument is optional; the default run directory is created under
`segment_result/<model>/result/`. Each run contains `instance_mask.tif`,
`binary_mask.png`, the model input `input.png`, and a provenance `run.json`.
Existing run directories are never overwritten.

For PNG tiles, provide the extracted slide resolution with `--mpp` when known.
For NDPI, MPP is read from slide metadata where available. `--target-mpp`
supplies the model reference scale. The default CellViT x40 checkpoint uses
0.25 µm/px; custom CellViT checkpoints and the other models require an explicit
reference scale. Runs record `scale_unverified` when either scale is unknown.
Do not compare physical-scale results until both source and model MPP are
established.

```bash
python -m src.model_registry.run_foundation_model \
  --model cellpose_sam --input /path/to/1024x1024.png \
  --mpp 0.25 --target-mpp 0.50 --device cuda
```

For an NDPI, pass a level-0 top-left coordinate pair:

```bash
python -m src.model_registry.run_foundation_model \
  --model pathosam --input /path/to/slide.ndpi --x 147456 --y 66048 \
  --level 0 --device cuda
```

This command assumes the selected model's Python dependencies and a compatible
PyTorch/CUDA environment are already installed. They are intentionally not
installed dynamically at inference time: model environments conflict, and
PLGrid CUDA and system-library setup depends on the selected cluster and job.
First-party weights are loaded or downloaded by the adapter where available.
SAM's official Meta ViT-B checkpoint (the base model weights) is downloaded
and SHA-256 checked in the model cache on its first run. CellSAM downloads the
official v1.2 checkpoint archive from the gated Hugging Face repository on its
first run; accept the repository license and provide an authorized `HF_TOKEN`
to the job environment (or use `hf auth login`). CellViT downloads its official
x40 checkpoint on first use; set
`CELLVIT_CACHE` to choose its persistent cache directory.

## Models, checkpoints, and licenses

Stable model IDs are `pathosam`, `cellvit`, `cellsam`, `cellpose_sam`, and
See [`../configs/model_registry/foundation_models.yaml`](../configs/model_registry/foundation_models.yaml)
for upstream revisions, weight sources, licenses, and native output semantics.
Check terms independently for code, weights, and generated outputs before
redistribution. `--checkpoint` selects a local checkpoint override;
`--checkpoint-root` or `MPN_AL_CHECKPOINT_ROOT` selects the SAM download cache.
Keep weights and source WSIs out of Git.

PathoSAM uses the upstream `vit_b_histopathology` loader and its built-in
weights by default. Cellpose uses the upstream `cpsam_v2` weights and downloads
them on first use. CellSAM downloads the official gated v1.2 checkpoint from
`vanvalenlab/cellsam` on Hugging Face and loads it through CellSAM's local model
loader. SAM downloads the official Meta ViT-B base checkpoint into the shared
cache. CellViT's adapter
downloads the official CellViT-SAM-H x40 checkpoint by default. These pretrained
weights have not been validated against the project's marrow whole-cell
annotations; PathoSAM and CellViT are nucleus-oriented, and vanilla SAM is a
generic proposal baseline.

The project has no frozen marrow cell taxonomy or reviewed per-instance
training set yet, so this implementation does not invent or train a disease
classifier. CellViT's native PanNuke predictions are preserved as
`instances.json` when returned, but they are nucleus labels and are not remapped
to MPN categories. Other adapters only write `instances.json` if their native
inference result supplies instance-aligned classes or probabilities.

## Windows 11 with WSL2

Use a Linux distribution under WSL2 and keep the checkout in that distribution's
Linux filesystem (for example, `~/work/MPN_AL`) for better filesystem
performance. Windows paths such as `C:\\...` are not valid paths in a Linux
job. Microsoft's current setup and GPU guidance: [Install WSL](https://learn.microsoft.com/windows/wsl/install)
and [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

In an elevated Windows PowerShell, install WSL if it is not already present,
then restart if requested:

```powershell
wsl --install -d Ubuntu
wsl --update
wsl --status
```

Install a recent NVIDIA Windows driver that supports CUDA on WSL from NVIDIA;
do not install a separate Linux display driver inside WSL. In Ubuntu, check
GPU visibility with `nvidia-smi`. GPU access is optional for tile I/O and
`--dry-run`, but real model inference generally needs a compatible GPU and
PyTorch build.

Clone or copy the repository into the Linux filesystem, then create an isolated
environment. Choose the Python version and package pins from the upstream
model requirements; the five models may need separate environments.

```bash
cd ~/work/MPN_AL
python3 -m venv .venv-model-registry
source .venv-model-registry/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Install the pinned, model-specific requirements here.
```

The shared `requirements.txt` installs the project dependencies used for image
handling, comparison outputs, and tests. The distribution is named `Pillow`,
although Python code imports it through the `PIL` module. Model inference
packages are intentionally not combined into this file: install the
selected adapter's versions from its upstream instructions because the model
environments may be incompatible with one another.

Verify the basic components independently:

```bash
python --version
python -c 'import openslide; print("OpenSlide Python:", openslide.__version__)'
ldconfig -p | grep -i openslide
python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA build:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available())'
nvidia-smi
```

OpenSlide is optional for PNG input. NDPI reading requires both the Python
binding and system OpenSlide libraries; follow the OpenSlide installation
instructions for the chosen Ubuntu release. A successful CUDA check requires a
compatible driver, PyTorch build, and WSL GPU passthrough. Check each separately
rather than inferring GPU readiness from `nvidia-smi` alone.

## Local smoke test

First check that the selected fixture exists. This checkout currently documents
the expected fixture but may not contain `sample_tile.png` until it has been
extracted from the local NDPI; do not mistake the zero-mask contract run for
model inference.

```bash
test -f src/model_registry/test_data/sample_tile.png
python -m src.model_registry.run_foundation_model --list-models
python -m src.model_registry.run_foundation_model \
  --model pathosam \
  --input src/model_registry/test_data/sample_tile.png \
  --output data/03_predictions/foundation_model_runs/pathosam-dry-run \
  --dry-run --border none
```

`--dry-run` extracts/validates the tile and writes a zero-mask contract result;
it does not load a checkpoint or execute a model. Inspect `run.json` for
`dry_run`, tile geometry, input and output shapes, and checksums. Open
`input.png`, `instance_mask.tif`, and `binary_mask.png` in an image viewer. The
mask should be empty in this contract run.

For PathoSAM, install its upstream environment and run a tile without
`--checkpoint`; the first run may download/cache the built-in weights:

```bash
python -m src.model_registry.run_foundation_model \
  --model pathosam \
  --input src/model_registry/test_data/sample_tile.png \
  --device cuda --seed 0 --border none \
  --output data/03_predictions/foundation_model_runs/pathosam-no-border
```

Install the selected adapter's dependencies according to its upstream
instructions. PathoSAM, Cellpose, and SAM use their first-party default
checkpoints when no `--checkpoint` is supplied. CellSAM's default model may
require an access token; CellViT downloads its x40 checkpoint through the
official cache helper. To use a local override, pass
`--checkpoint /path/to/checkpoint`. `--checkpoint-root` or
`MPN_AL_CHECKPOINT_ROOT` selects the SAM cache; `CELLVIT_CACHE` selects the
CellViT cache. Create a new output directory for every run because existing
run directories are never overwritten:

```bash
python -m src.model_registry.run_foundation_model \
  --model pathosam \
  --input src/model_registry/test_data/sample_tile.png \
  --device cuda --seed 0 --border 0.30 \
  --output data/03_predictions/foundation_model_runs/pathosam-border-030
```

For PLGrid, store model caches on project or durable storage when possible.
Set the relevant cache location before running:

```bash
export MPN_AL_CHECKPOINT_ROOT=/path/to/project/fm_weights
export CELLVIT_CACHE=/path/to/project/cellvit_cache
python -m src.model_registry.run_foundation_model \
  --model cellpose_sam --input "$INPUT" --device cuda
```

For `.ndpi`, provide both level-0 `--x` and `--y`; coordinates are top-left
coordinates. `--level` selects the pyramid level while the request retains
level-0 coordinates. Compare each `run.json` and visually inspect the saved
input and masks, confirming the prediction is 1024 × 1024 and the border case
is centered. Keep model, checkpoint, device, seed, and all overrides identical
between border conditions. Do not claim accuracy without a reviewed instance
mask.

## Compare saved runs

`src/model_registry/comparison.py` reads completed run directories (the
`run.json`, `input.png`, and `instance_mask.tif` files) and writes
`metrics.csv`, `summary.csv`, a provenance `manifest.json`, and qualitative
overlays. Point `--runs` at a directory containing model run directories;
recursive discovery finds each `run.json`.

For example, after transferring the results to the project machine:

```bash
python -m src.model_registry.comparison \
  --runs /path/to/segmentation_results \
  --tile-id testZL26_x15872_y34304
```

By default, the comparison output is saved to
`src/model_registry/results/`. Pass `--output /path/to/another/directory` to
choose a different location. If the destination already exists, add `--force`
to allow the run to write or replace its output files.

The output includes per-run `metrics.csv` and `summary.csv`, the input/run
provenance in `manifest.json`, and a `qualitative_<tile-hash>.png` overlay
showing each model's predicted boundaries. When Matplotlib is installed,
`runtime_cell_count.png` compares inference time with predicted object count.
With reviewed references, the run also writes `metric_by_model.png` and/or
`megakaryocyte_metrics.png` for the scores that were supplied; `metrics.parquet`
is written when pandas and a Parquet engine are available.

Without reference annotations the reports include predicted instance count,
foreground fraction, connected-component areas, inference time, and visual
overlays. Accuracy metrics are `NA`; these descriptive values cannot establish
which model segments cells or megakaryocytes most accurately.

For accuracy scoring, prepare a reviewed integer instance-label TIFF at the
same dimensions as the saved masks. Give every whole cell a unique positive
label, use `0` for background, and optionally `-1` for pixels to ignore. To
score megakaryocytes separately, prepare a second integer TIFF containing only
reviewed megakaryocyte instances (unique positive IDs; `0` for other pixels;
optional `-1` for ignored pixels), then pass both masks:

```bash
python -m src.model_registry.comparison \
  --runs /path/to/segmentation_results \
  --tile-id testZL26_x15872_y34304 \
  --reference /path/to/reviewed_whole_cell_instances.tif \
  --megakaryocyte-reference /path/to/reviewed_megakaryocyte_instances.tif \
  --force
```

Whole-cell scoring reports foreground Dice/IoU, instance precision/recall/F1,
and panoptic quality (PQ). Megakaryocyte scoring uses one-to-one maximum-IoU
matching at IoU 0.5, reporting detection recall and mean matched IoU across
reviewed target megakaryocytes (no-overlap targets count as IoU zero; overlaps
below 0.5 still contribute to mean IoU but not detection recall). It does not
count other predicted cells as megakaryocyte false positives. Check
`megakaryocyte_metrics.png` and the corresponding columns in `metrics.csv` /
`summary.csv`. The supplied `binary_mask.png` files are predictions, not
reviewed references, and do not contain per-instance IDs; do not use them as
ground truth.

Reference masks can only be used when comparing runs for one tile at a time;
`--tile-id` selects that tile when the run root has multiple tiles. Repeat for
other reviewed tiles (and preferably held-out slides) before choosing a model:
a single tile, especially one megakaryocyte, is not a robust benchmark.

## PLGrid: ARES and ATHENA

The examples below are Slurm templates, not guaranteed resource names or
quotas. ARES and ATHENA availability, partition names, GPU types/counts,
accounting strings, limits, software modules, and storage paths depend on the
current system configuration and the user's active PLGrid project. Before
submitting, read the [ARES documentation](https://docs.cyfronet.pl/ares) or
[ATHENA documentation](https://docs.cyfronet.pl/athena), confirm the project
grant includes the desired system/GPU/storage resources in the PLGrid Portal,
and query the live cluster:

```bash
sinfo -o '%P %a %l %D %G'
scontrol show partition <partition-from-sinfo>
sacctmgr show assoc user="$USER" format=Account,Partition,QOS
```

Replace every angle-bracket placeholder below. Do not copy a partition value
from an old example: use a partition/QOS and GPU request exposed to your grant.
The official [Cyfronet ARES and ATHENA guides](https://docs.cyfronet.pl/)
describe site-specific login, storage, modules, and scheduler policy. Site
training material documents standard Slurm forms such as `--partition`,
`--gres=gpu:N`, `--mem`, `--time`, and `srun`; the exact allowable values remain
site- and grant-dependent.

### Environment and staging

1. Request/confirm project allocation for the selected system, GPU computation
   (if needed), and durable storage. Select the project/account and the site
   when logging in or transferring data as instructed by the current portal
   and system guide.
2. Clone the repository into a project or user work directory on the cluster.
   Stage only the required source tile or WSI region. Official weights are
   downloaded on first use where their upstream loader supports it; ensure
   outbound network access or pre-populate that loader's cache. Use the
   system's scratch area for temporary caches and temporary model files; use
   project/durable storage for completed run
   directories and logs. Check the local storage guide for quotas, purge
   policies, and valid paths.
3. Use a site-supported Apptainer/Singularity image when available, or load the
   documented compiler/CUDA/Python modules and create a compatible virtual
   environment in an approved location. A container should match the host GPU
   driver interface and include model-specific dependencies. Record its
   immutable image digest. Do not assume that a Windows path or a login-node
   environment is visible to compute jobs.
4. Put temporary caches on scratch, for example `export
   TORCH_HOME="$SCRATCH/torch"` only after confirming `$SCRATCH` is defined by
   the site. Do not keep the only copy of a checkpoint or result in purgeable
   scratch.

### ARES batch template

Save as `run_pathosam_ares.sbatch`. Confirm the partition, GPU syntax, account,
memory, and time limits for ARES before use. Set `REPO`, `INPUT`, and the log
directory to paths visible on the compute node. Official PathoSAM weights are
loaded by its upstream model loader.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=mpn-pathosam
#SBATCH --account=<project-account>
#SBATCH --partition=<ARES-GPU-partition-from-sinfo>
#SBATCH --gres=gpu:<count>
#SBATCH --cpus-per-task=<cpu-count>
#SBATCH --mem=<memory>
#SBATCH --time=<HH:MM:SS>
#SBATCH --output=<persistent-log-dir>/pathosam-%j.out
#SBATCH --error=<persistent-log-dir>/pathosam-%j.err

set -euo pipefail
REPO=<absolute-repository-path>
INPUT=<absolute-input-png-or-ndpi-path>
cd "$REPO"
# Use the exact modules/container and environment documented for this ARES setup.
# module load <site-approved-python-and-cuda-modules>
source <venv-path>/bin/activate
python -m src.model_registry.run_foundation_model \
  --model pathosam --input "$INPUT" \
  --device cuda --seed 0 --border none
```

Submit with `sbatch run_pathosam_ares.sbatch`. For `.ndpi`, append `--x <level-0-x>
--y <level-0-y>`. Repeat with a different `OUT` and `--border 0.30` for the
context case.

Interactive debugging template (use only a short, site-permitted allocation):

```bash
srun --account=<project-account> --partition=<ARES-GPU-partition> \
  --gres=gpu:<count> --cpus-per-task=<cpu-count> --mem=<memory> \
  --time=<short-limit> --pty bash -l
```

### ATHENA batch template

ATHENA resource labels and limits can differ from ARES. Re-run `sinfo` and
`scontrol` on ATHENA and replace these placeholders with values allowed for the
active project. Select memory, CPU count, and wall time for a single-tile
inference job; start small and adjust from observed usage.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=mpn-pathosam
#SBATCH --account=<project-account>
#SBATCH --partition=<ATHENA-GPU-partition-from-sinfo>
#SBATCH --gres=gpu:<count>
#SBATCH --cpus-per-task=<cpu-count>
#SBATCH --mem=<memory>
#SBATCH --time=<HH:MM:SS>
#SBATCH --output=<persistent-log-dir>/pathosam-%j.out
#SBATCH --error=<persistent-log-dir>/pathosam-%j.err

set -euo pipefail
REPO=<absolute-repository-path>
INPUT=<absolute-input-png-or-ndpi-path>
cd "$REPO"
# Load only module names listed by the current ATHENA documentation/site query.
# module load <site-approved-python-and-cuda-modules>
source <venv-path>/bin/activate
python -m src.model_registry.run_foundation_model \
  --model pathosam --input "$INPUT" \
  --device cuda --seed 0 --border 0.30
```

Run with `sbatch run_pathosam_athena.sbatch`. The interactive equivalent is:

```bash
srun --account=<project-account> --partition=<ATHENA-GPU-partition> \
  --gres=gpu:<count> --cpus-per-task=<cpu-count> --mem=<memory> \
  --time=<short-limit> --pty bash -l
```

### Moving data and outputs

Use the site's documented transfer endpoint/tool (for example its supported
SSH/SFTP/Globus route) to stage only required inputs and weights. Never launch
inference against an unverified Windows-mounted path. Write temporary
extractions and caches to scratch, and copy the completed run directory
(`input.png`, masks, `run.json`, and `README.md`) plus scheduler stdout/stderr to
durable project storage when the job succeeds. Confirm copies with checksums.
Protect patient data under the project's data governance rules; do not put
patient identifiers in public logs or filenames.

## Reproducibility record

The dispatcher records input/checkpoint checksums when applicable, model and
adapter identity, tile coordinates, border and padding settings, seed,
overrides, Python/Pillow/NumPy/PyTorch versions, device, timings, output
checksums, and warnings in `run.json`. For every benchmark run, also record or
archive:

- Git commit (`git rev-parse HEAD`) and whether the working tree was modified;
- exact Python lock/requirements file and model repository commit;
- full command line, scheduler job ID, system, partition, requested/allocated
  resources, and relevant module list (`module list`);
- container image digest, if used;
- checkpoint SHA-256, input SHA-256, tile ID, level-0 coordinates, level,
  border mode, seed, and all model overrides.

Retain the immutable run directory. Choose a fresh output directory for every
run; the dispatcher intentionally refuses to overwrite one.

## Verification status

Verification on 2026-11-07 used the repository's configured Windows venv
(Python 3.13.6):

- `python -m compileall -q src/model_registry` completed successfully.
- Unit tests were not run: `pytest` is not installed in that environment.
  The current test tree contains only `tests/model_registry/test_tile_io.py`;
  dispatcher, comparison, and adapter contract tests are still needed.
- No fixture inference or visual mask inspection was possible because
  `src/model_registry/test_data/sample_tile.png` is not present. Consequently,
  neither the dry-run output checks nor real inference in both border modes
  were exercised.
- Checkpoint-backed inference, comparison against a reviewed reference,
  clean WSL/container execution, and ARES/ATHENA jobs remain unverified. No
  reviewed `sample_mask.tif` is present.

These are environment/data availability gaps, not passing acceptance results.
After installing pytest and adding the fixture, run the checks above before
claiming the workflow is verified. Record real model failures separately as
checkpoint, dependency, or hardware limitations, as required by the plan.
