# Segmentation head: checkpoint and training inputs

## `head.pt`

Written by `scripts/train_segmentation_head.py --out <run_dir>`; intended location
`06_models/<round_id>/<run_id>/`. The run directory holds `head.pt` (best validation PQ),
`last.pt`, `history.json` and `run.json`.

A head checkpoint is a plain dictionary (loadable with `torch.load(..., weights_only=True)`):

| Key | Notes |
|---|---|
| `format` | `mpn_al.segmentation_head/1`. |
| `base_model` | The base-model spec the head was trained on (source, constructor arguments, normalisation, licence). |
| `encoder` | `embed_dim`, `patch_size`, `num_layers` of that encoder; checked when the head is loaded. |
| `layers` | The four encoder blocks feeding the decoder. |
| `seg_mpp` | Working resolution in µm/px, or null for "the tile's own resolution". |
| `decoder` | `num_types`, `base_width`, `norm`, `dropout`. |
| `labels` | The label schema (class ids, names, colours, aliases, ignore and ROI names). |
| `state_dict` | Decoder weights only. |
| `extra` | Epoch, metrics, training settings, ids of training and validation tiles. |

**The encoder's weights are never stored.** A head is useless without the base model it was
trained on, and it can be shared even when that model's licence forbids redistributing
weights. Loading a head fetches the encoder from its usual source.

## Label schema (`configs/labels/*.yaml`)

```yaml
name: megakaryocyte_binary
classes:                      # ids 1..K in order; 0 is background
  - id: 1
    name: megakaryocyte
    color: [230, 25, 75]
    aliases: [meg_norm, meg_atypical]   # annotation names that count as this class
ignore: [uncertain]           # polygons with these names are excluded from the loss
roi: [ROI]                    # polygons with these names delimit the exhaustively annotated area
```

A schema with one class trains a head without a type branch; with several classes the head
also predicts the class of every pixel and a cell takes the majority class.

## Training labels

`--annotations <dir>`: `<tile_id>.geojson` per tile, QuPath export in level-0 slide
coordinates (the layout of `04_annotations/<round>/by_tile`). The tile's origin is taken from
its name (`..._x<x>_y<y>_s<size>`); for an image named otherwise the coordinates are read as
pixels of that image. The class is read from `properties.classification.name`. Rules:

- a polygon of a schema class is one cell instance;
- a polygon with an `ignore` name, or a class the schema does not know, is excluded from the loss;
- if a tile has `roi` polygons, everything outside them is excluded;
- pixels without any annotation are background, or excluded with `--unlabeled ignore`.

`--masks <dir>`: ready-made arrays per tile (the layout of `05_targets/<dataset_id>/by_tile`),
in any of these forms:

| Form | Content |
|---|---|
| `<tile_id>/instances.npy` (+ `types.npy`, `ignore.npy`) | `[H, W]` arrays: instance ids (0 = background), class ids, boolean ignore mask. |
| `<tile_id>.npz` | The same arrays under the keys `instances`, `types`, `ignore`. |
| `<tile_id>.npy` | `[H, W]` instance ids, or `[H, W, 2]` = (instance ids, class ids). |
| `<tile_id>.png` / `.tif` | Integer instance-id image. |

Without `types`, every instance is class 1. The horizontal/vertical distance maps are derived
from the instance map during training and are not stored.
