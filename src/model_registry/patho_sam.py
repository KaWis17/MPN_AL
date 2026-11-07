"""PathoSAM adapter.

PathoSAM releases do not expose a stable Python inference API across revisions.
This adapter deliberately requires the pinned release's callable to be supplied
by its Python package instead of guessing a semantic-segmentation entry point.
"""
from __future__ import annotations
import importlib
import time
from pathlib import Path
from typing import Any
from .base import ModelAdapter, TileRequest
from ._adapter_utils import checked_checkpoint, result, rgb_array

class PathoSAMAdapter(ModelAdapter):
    model_name = "pathosam"
    version = "1.0"
    def __init__(self, *, predictor: Any = None): self._predictor = predictor; self._checkpoint = None; self._device = "cpu"
    def load(self, checkpoint_path: Path | None = None, *, device: str = "cpu") -> None:
        self._checkpoint = checked_checkpoint(checkpoint_path, self.model_name); self._device = device
        try:
            from pathosam import load_model
        except ImportError as exc:
            raise RuntimeError("PathoSAM dependencies are unavailable. Install the pinned upstream release and its documented tissue/patch preprocessing.") from exc
        try: self._predictor = load_model(str(self._checkpoint), device=device)
        except Exception as exc: raise RuntimeError(f"Could not load PathoSAM checkpoint {self._checkpoint}: {exc}") from exc
        if not callable(self._predictor): raise RuntimeError("This PathoSAM release has no supported callable cell-instance predictor; semantic tissue outputs cannot satisfy the benchmark contract")
    def predict(self, tile: Any, request: TileRequest) -> Any:
        if self._predictor is None: raise RuntimeError("Call load() before predict()")
        return self._predictor(rgb_array(tile))
    def postprocess(self, raw_output: Any, request: TileRequest):
        start = time.perf_counter()
        if isinstance(raw_output, dict): raw_output = raw_output.get("instance_mask", raw_output.get("instances"))
        if raw_output is None: raise RuntimeError("PathoSAM did not return cell instances; refusing semantic tissue segmentation")
        return result(self.model_name, self.version, raw_output, request, str(self._checkpoint), self._device, start, {"postprocessing":"nearest-neighbor resize; center crop context"})
