"""Cellpose-SAM adapter for instance segmentation."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any
from .base import ModelAdapter, TileRequest
from ._adapter_utils import checked_checkpoint, result, rgb_array

class CellposeSAMAdapter(ModelAdapter):
    model_name="cellpose_sam"; version="1.0"
    def __init__(self,*,diameter:float|None=None,flow_threshold:float=0.4,cellprob_threshold:float=0.0): self.diameter=diameter; self.flow_threshold=flow_threshold; self.cellprob_threshold=cellprob_threshold; self._model=None; self._checkpoint=None; self._device="cpu"
    def load(self,checkpoint_path:Path|None=None,*,device:str="cpu")->None:
        self._checkpoint=checked_checkpoint(checkpoint_path,self.model_name); self._device=device
        try: from cellpose import models
        except ImportError as exc: raise RuntimeError("Cellpose-SAM requires the cellpose package; install the pinned Cellpose-SAM environment.") from exc
        try: self._model=models.CellposeModel(pretrained_model=str(self._checkpoint), gpu=device.startswith("cuda"))
        except Exception as exc: raise RuntimeError(f"Could not load Cellpose-SAM checkpoint: {exc}") from exc
    def predict(self,tile:Any,request:TileRequest)->Any:
        if self._model is None: raise RuntimeError("Call load() before predict()")
        masks, *_ = self._model.eval(rgb_array(tile), diameter=self.diameter, flow_threshold=self.flow_threshold, cellprob_threshold=self.cellprob_threshold, channels=[0,0])
        return masks
    def postprocess(self,raw_output:Any,request:TileRequest):
        start=time.perf_counter()
        return result(self.model_name,self.version,raw_output,request,str(self._checkpoint),self._device,start,{"diameter":self.diameter,"flow_threshold":self.flow_threshold,"cellprob_threshold":self.cellprob_threshold,"channels":[0,0],"postprocessing":"instance labels; nearest-neighbor resize; center crop context"})
