# Foundation model cell segmentation

This guide covers the single-tile benchmark under `src/model_registry/`. It is
an inference harness for one 1024-by-1024 field of view at a time. A `0.30`
border adds 30% of the canonical width and height **on every side**; the input
is larger and the saved prediction is the centered 1024-by-1024 crop. It does
not yet implement WSI tiling or provide a quality ranking. Quality metrics
require a reviewed reference instance mask.

## Models, checkpoints, and licenses

Stable model IDs are `pathosam`, `cellvit`, `cellsam`, `cellpose_sam`, and
`sam`. See [`../configs/model_registry/foundation_models.yaml`](../configs/model_registry/foundation_models.yaml)
for upstream repositories, papers, expected checkpoint locations, and
dependency notes. The adapters have different upstream APIs and environments;
do not assume one shared set of model dependencies will work for all five.

Before running a model, consult its upstream installation and inference guide,
pin the repository revision and compatible package versions, obtain the
checkpoint from its documented source, and record any required license or
checkpoint terms. Check terms independently for code, weights, and generated
outputs before distributing any of them. The harness does not download
checkpoints. By default, checkpoint directories are under
`data/06_models/fm_weights/`; keep weights and source WSIs out of Git.

The requested paper and code references are collected in
[`../../Instruction.md`](../../Instruction.md) and the YAML configuration.
These upstream references are starting points, not confirmation that model
weights are redistributable or that model outputs can be published.

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
# Install the pinned, model-specific requirements here.
```

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

For a real inference run, install that adapter's pinned dependencies and pass a
local checkpoint explicitly (or use the adapter's documented default path if
supported). Create a new output directory for every run because existing
directories are never overwritten:

```bash
python -m src.model_registry.run_foundation_model \
  --model pathosam \
  --input src/model_registry/test_data/sample_tile.png \
  --checkpoint data/06_models/fm_weights/pathosam/weights.pth \
  --device cuda --seed 0 --border none \
  --output data/03_predictions/foundation_model_runs/pathosam-no-border

python -m src.model_registry.run_foundation_model \
  --model pathosam \
  --input src/model_registry/test_data/sample_tile.png \
  --checkpoint data/06_models/fm_weights/pathosam/weights.pth \
  --device cuda --seed 0 --border 0.30 \
  --output data/03_predictions/foundation_model_runs/pathosam-border-030
```

For `.ndpi`, provide both level-0 `--x` and `--y`; coordinates are top-left
coordinates. `--level` selects the pyramid level while the request retains
level-0 coordinates. Compare each `run.json` and visually inspect the saved
input and masks, confirming the prediction is 1024 × 1024 and the border case
is centered. Keep model, checkpoint, device, seed, and all overrides identical
between border conditions. Do not claim accuracy without a reviewed instance
mask.

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
   Stage only the fixture or the required source WSI region and the checkpoint
   needed for the run. Use the system's scratch area for temporary caches and
   temporary model files; use project/durable storage for completed run
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
memory, and time limits for ARES before use. Set `REPO`, `INPUT`, `CHECKPOINT`,
and `OUT` to paths visible on the compute node. Create `OUT`'s parent and the
log directory on persistent storage before submitting.

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
CHECKPOINT=<absolute-checkpoint-path>
OUT=<new-unique-output-directory>
cd "$REPO"
# Use the exact modules/container and environment documented for this ARES setup.
# module load <site-approved-python-and-cuda-modules>
source <venv-path>/bin/activate
mkdir -p "$(dirname "$OUT")"
python -m src.model_registry.run_foundation_model \
  --model pathosam --input "$INPUT" --checkpoint "$CHECKPOINT" \
  --device cuda --seed 0 --border none --output "$OUT"
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
CHECKPOINT=<absolute-checkpoint-path>
OUT=<new-unique-output-directory>
cd "$REPO"
# Load only module names listed by the current ATHENA documentation/site query.
# module load <site-approved-python-and-cuda-modules>
source <venv-path>/bin/activate
mkdir -p "$(dirname "$OUT")"
python -m src.model_registry.run_foundation_model \
  --model pathosam --input "$INPUT" --checkpoint "$CHECKPOINT" \
  --device cuda --seed 0 --border 0.30 --output "$OUT"
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
