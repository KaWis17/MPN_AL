# `tiles.parquet` (per `prep_id`)

| Column | Type | Notes |
|---|---|---|
| `tile_id` | string | Primary key. Annotation FOV. |
| `prep_id` | string | Preprocessing run that created this row. |
| `slide_id` | string | |
| `patient_id` | string | |
| `x` | int | Level-0 top-left. |
| `y` | int | Level-0 top-left. |
| `width` | int | Always 1024 for annotation tiles. |
| `height` | int | Always 1024. |
| `mpp` | float | Slide microns per pixel. |
| `stride` | int | Grid stride used in this prep. |
| `tissue_frac` | float | |
| `artefact_frac` | float | |
| `qc_pass` | bool | |
| `split` | string | `train` / `validation` / `test` from patient split. |

# `model_windows.parquet` (optional, same `prep_id`)

| Column | Type | Notes |
|---|---|---|
| `window_id` | string | Primary key. |
| `tile_id` | string | Parent annotation FOV. |
| `prep_id` | string | |
| `role` | string | `embed` or `segment`. |
| `x`, `y` | int | Level-0 top-left of the window. |
| `size_level0` | int | Side in level-0 pixels. |
| `size_model` | int | Side fed to the network after resize. |
| `mpp_model` | float | Target MPP of the encoder. |
