# `embeddings.parquet` (per round, per base model)

Written by `scripts/extract_embeddings.py`. One row per annotation tile: the hidden
representation of the tile from an SSL base model. Intended location:
`02_active_learning/rounds/<round_id>/embeddings.parquet`.

| Column | Type | Notes |
|---|---|---|
| `tile_id` | string | Annotation FOV; join key. |
| `prep_id` | string | Preprocessing run of the tile (null for loose image files). |
| `slide_id` | string | |
| `patient_id` | string | |
| `base_model` | string | Registry name or ad-hoc reference, e.g. `uni2-h`. |
| `pooling` | string | `cls`, `mean` or `cls+mean` (class token and mean patch token concatenated). |
| `window_mode` | string | `grid`, `center`, `full` or `thumbnail`: how the tile was fed to the model. |
| `mpp_model` | float32 | Resolution the model read at, in µm/px. |
| `n_windows` | int64 | Model reads averaged into this row. |
| `embedding` | fixed_size_list<float32>[D] | Mean of the window embeddings. D depends on the model and pooling. |

Embeddings of different base models, poolings or window modes are not comparable: keep one
file per combination. The sidecar `embeddings.run.json` records the exact model, settings,
code commit and command line.

```python
table = pd.read_parquet("embeddings.parquet")
matrix = np.stack(table["embedding"].to_numpy())   # [n_tiles, D] float32
```

## Job arrays (`--num-shards`)

With `--shard-index I --num-shards N` (Slurm job arrays do this for you, see
[plgrid.md](../docs/architecture/plgrid.md)) every task writes its own file,
`embeddings.shard-III-of-NNN.parquet`, plus the matching `.run.json`; so do `--windows-out` and
`--windows-table`. `python scripts/merge_shards.py embeddings <path>/embeddings.parquet` joins them
(`--key window_id` for window tables). It refuses incomplete or overlapping sets of shards. A task whose shard
holds no tiles writes no file (`--allow-missing` skips such gaps). `--skip-existing` makes a task exit at once
when its shard file exists, so a resubmitted array redoes only what is missing.

## Per-window embeddings (`--windows-out`)

| Column | Type | Notes |
|---|---|---|
| `window_id` | string | Primary key; same ids as `model_windows.parquet`. |
| `tile_id` | string | Parent annotation FOV. |
| `prep_id` | string | |
| `base_model` | string | |
| `pooling` | string | |
| `embedding` | fixed_size_list<float32>[D] | Embedding of this one model read. |

## Window geometry (`--windows-table`)

Rows in the format of `model_windows.parquet` ([tiles.parquet.md](tiles.parquet.md)) with
`role = embed`, plus `height_level0` and `height_model`, which equal `size_level0` and
`size_model` for square windows. `mpp_model` is the resolution the network actually sees
(`mpp_slide * size_level0 / size_model`). Window ids are
`win_<tile id without tl_>_dx<dx>_dy<dy>_s<size_level0>_<size_model>_mpp<mpp>`, with `dx`,`dy`
the offset inside the tile.
