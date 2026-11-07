"""CellViT++ adapter; expects the upstream CellViT inference package."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any
from .base import ModelAdapter, TileRequest
from ._adapter_utils import checked_checkpoint, result, rgb_array

class CellViTAdapter(ModelAdapter):
    model_name="cellvit"; version="1.0"
    def __init__(self, *, predictor: Any=None): self._predictor=predictor; self._checkpoint=None; self._device="cpu"
    def load(self, checkpoint_path: Path|None=None, *, device: str="cpu") -> None:
        self._checkpoint=checked_checkpoint(checkpoint_path,self.model_name); self._device=device
        try: from cellvit.inference import CellViTInference
        except ImportError as exc: raise RuntimeError("CellViT++ inference dependencies are missing. Install the pinned CellViT++ release and its torch/torchvision requirements.") from exc
        try: self._predictor=CellViTInference(str(self._checkpoint), device=device)
        except Exception as exc: raise RuntimeError(f"Could not initialize CellViT++: {exc}") from exc
    def predict(self,tile:Any,request:TileRequest)->Any:
        if self._predictor is None: raise RuntimeError("Call load() before predict()")
        image=rgb_array(tile)
        # CellViT patches are conventionally 1024 px at the model's training MPP.
        return self._predictor.predict_instances(image)
    def postprocess(self,raw_output:Any,request:TileRequest):
        start=time.perf_counter(); labels=raw_output.get("instance_map",raw_output.get("instances")) if isinstance(raw_output,dict) else raw_output
        if labels is None: raise RuntimeError("CellViT++ output contains no instance map")
        return result(self.model_name,self.version,labels,request,str(self._checkpoint),self._device,start,{"native_patch_size":1024,"mpp":"checkpoint dependent; use matching magnification","classes":"collapsed to class-agnostic instances","postprocessing":"nearest-neighbor resize; center crop context"})
