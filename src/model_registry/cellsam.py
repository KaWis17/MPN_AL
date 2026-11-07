"""CellSAM adapter using the cell-oriented automatic instance path."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any
from .base import ModelAdapter, TileRequest
from ._adapter_utils import checked_checkpoint, result, rgb_array

class CellSAMAdapter(ModelAdapter):
    model_name="cellsam"; version="1.0"
    def __init__(self, *, predictor: Any=None): self._predictor=predictor; self._checkpoint=None; self._device="cpu"
    def load(self,checkpoint_path:Path|None=None,*,device:str="cpu")->None:
        self._checkpoint=checked_checkpoint(checkpoint_path,self.model_name); self._device=device
        try: from cellSAM import cellSAM
        except ImportError as exc: raise RuntimeError("CellSAM dependencies are missing. Install the pinned CellSAM upstream package.") from exc
        try: self._predictor=cellSAM
        except Exception as exc: raise RuntimeError(f"Could not initialize CellSAM: {exc}") from exc
    def predict(self,tile:Any,request:TileRequest)->Any:
        if self._predictor is None: raise RuntimeError("Call load() before predict()")
        masks=self._predictor(rgb_array(tile), model_path=str(self._checkpoint))
        if isinstance(masks,dict): masks=masks.get("masks",masks.get("labels"))
        return masks
    def postprocess(self,raw_output:Any,request:TileRequest):
        start=time.perf_counter()
        return result(self.model_name,self.version,raw_output,request,str(self._checkpoint),self._device,start,{"path":"automatic cell segmentation","thresholds":"upstream defaults","postprocessing":"instance labels; nearest-neighbor resize; center crop context"})
