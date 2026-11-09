# Model registry test tiles

`tiles/` contains 18 RGB, 1024 × 1024 PNG tiles for smoke inference. Their
filenames follow `<image_id>_x<level-0-x>_y<level-0-y>.png`. No reviewed
instance masks are available, so these images must not be used to claim
segmentation quality.

Pass the directory to the foundation-model dispatcher to run every PNG through
one selected model:

```text
python -m src.model_registry.run_foundation_model --model <model-id> \
  --input src/model_registry/test_data/tiles \
  --output data/03_predictions/foundation_model_runs
```

Each PNG is read as a complete tile; the coordinates in its filename are
recorded as level-0 provenance and are not used to crop the PNG again. The
dispatcher writes one unique run directory per tile. It loads the model once
and reuses it for the directory.

Coordinates in `TileRequest` are level-0 top-left coordinates. A 0.30 border
adds 30% of the canonical output size on every side (rounded to the nearest
pixel); only the model input contains context, and `Tile.central_crop()`
returns the canonical center field of view.
