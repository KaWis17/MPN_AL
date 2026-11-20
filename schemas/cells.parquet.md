# Segmentation outputs (per run directory)

Written by `scripts/segment_cells.py --out <run_dir>`. Intended location:
`03_predictions/<round_id>/`.

```text
<run_dir>/
  by_tile/<tile_id>.geojson     cell polygons for QuPath
  cells/<tile_id>.parquet       one row per cell, with its embedding
  tile_stats.jsonl              one line per tile
  run.json                      provenance: model, head, settings, code commit, command line
  masks/<tile_id>.npz           only with --save-masks: instance map at working resolution
```

A tile is finished when both of its files exist; each is moved into place only when complete.
With job arrays, `tile_stats` and `run` carry a `.shard-XXX-of-YYY` suffix; `read_tile_stats(<run_dir>)` reads
them together and `python scripts/merge_shards.py tile-stats <run_dir>` writes one `tile_stats.jsonl`. The
per-tile files need no merging.

## `by_tile/<tile_id>.geojson`

A `FeatureCollection` in the dialect QuPath imports. Coordinates are level-0 slide pixels
(tile-local pixels for image files whose name carries no origin).

| Field | Notes |
|---|---|
| `id` | UUID, stable for a given tile and cell order. |
| `geometry` | `Polygon`, one closed exterior ring, coordinates rounded to 0.1 px. |
| `properties.objectType` | `annotation` (default, editable in QuPath), `detection` or `cell`. |
| `properties.classification` | `{name, color}` from the label schema of the head. |
| `properties.measurements` | `area_px`, `area_um2`, `prob` (mean cell probability), `class_prob`, `touches_border` (present and 1 when the cell touches the tile edge). |

## `cells/<tile_id>.parquet`

`pd.read_parquet("<run_dir>/cells")` loads every tile as one table.

| Column | Type | Notes |
|---|---|---|
| `cell_id` | string | `<tile_id>_c<index>`; index = position in the tile's GeoJSON. |
| `tile_id` | string | |
| `prep_id` | string | |
| `slide_id` | string | |
| `x`, `y` | float32 | Centroid, level-0 slide pixels. |
| `area_px` | float32 | Level-0 pixels². |
| `area_um2` | float32 | Null when the slide resolution is unknown. |
| `class_id` | int64 | 1..K as in the label schema. |
| `class_name` | string | |
| `prob` | float32 | Mean cell probability inside the cell; null for the stain baseline. |
| `class_prob` | float32 | Share of the per-pixel type vote won by `class_id`; null without a type branch. |
| `touches_border` | bool | The cell reaches the tile edge and may be cut. |
| `method` | string | `head` (base-model segmenter) or `stain-baseline`. |
| `base_model` | string | Encoder that produced `embedding`. |
| `embedding` | fixed_size_list<float32>[D] | Base-model patch tokens averaged over the cell's area. Absent with `--no-cell-embeddings`. |

## `tile_stats.jsonl`

One JSON object per processed tile, appended as the run proceeds; a tile that was redone
appears again, and the last line wins (`mpn_al.segmenter.read_tile_stats`).

| Key | Notes |
|---|---|
| `tile_id`, `prep_id`, `method` | |
| `n_cells`, `n_<class_name>` | Counts, in total and per class. |
| `n_border_cells` | Cells touching the tile edge. |
| `foreground_fraction` | Share of the tile covered by cells. |
| `mean_entropy` | Mean binary entropy (bits, 0..1) of the cell probability over the tile. Head only. |
| `uncertain_fraction` | Share of pixels with cell probability in [0.3, 0.7]. Head only. |
| `mean_cell_prob` | Mean of the cells' `prob`. Head only. |
| `seconds` | Processing time. |

The three uncertainty numbers are raw inputs for the selector's `candidate_scores.parquet`.
