# Model registry smoke fixture

`sample_tile.png` is intended to be a 1024 × 1024 RGB center crop from the
local bone-marrow NDPI slide. It is a smoke-inference fixture only: no reviewed
instance mask is available, so it must not be used to claim segmentation
quality. The source slide is local working data and must not be copied here.

The source NDPI is present in this checkout's ignored `data/` directory, but
the fixture image and `sample_tile.json` have not yet been generated because
the OpenSlide backend could not be run in the current environment. After
installing OpenSlide, call `extract_tile(TileRequest(...))` in center-crop mode
and save `tile.image` as `sample_tile.png`; record the returned metadata,
source SHA-256, and extracted-image SHA-256 in `sample_tile.json`. Keep those
files together so the fixture identity and extraction provenance remain
stable. Add `sample_mask.tif` only after a reviewer creates an instance mask.

Coordinates in `TileRequest` are level-0 top-left coordinates. A 0.30 border
adds 30% of the canonical output size on every side (rounded to the nearest
pixel); only the model input contains context, and `Tile.central_crop()` returns
the canonical center field of view.
