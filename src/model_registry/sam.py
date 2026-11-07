"""SAM automatic-mask baseline adapter.

SAM does not promise cell instances. Results are accepted only when its masks
are returned as separate instances; merged semantic regions fail validation.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any
from .base import ModelAdapter, TileRequest
from ._adapter_utils import checked_checkpoint, result, rgb_array

class SAMAdapter(ModelAdapter):
    model_name="sam"; version="1.0"
    def __init__(self,*,model_type:str="vit_b",points_per_side:int=32,pred_iou_thresh:float=.88,stability_score_thresh:float=.95): self.model_type=model_type; self.points_per_side=points_per_side; self.pred_iou_thresh=pred_iou_thresh; self.stability_score_thresh=stability_score_thresh; self._generator=None; self._checkpoint=None; self._device="cpu"
    def load(self,checkpoint_path:Path|None=None,*,device:str="cpu")->None:
        self._checkpoint=checked_checkpoint(checkpoint_path,self.model_name); self._device=device
        try: from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError as exc: raise RuntimeError("SAM dependencies are missing. Install the pinned segment-anything release.") from exc
        try:
            model=sam_model_registry[self.model_type](checkpoint=str(self._checkpoint)); model.to(device=device)
            self._generator=SamAutomaticMaskGenerator(model,points_per_side=self.points_per_side,pred_iou_thresh=self.pred_iou_thresh,stability_score_thresh=self.stability_score_thresh)
        except Exception as exc: raise RuntimeError(f"Could not initialize SAM: {exc}") from exc
    def predict(self,tile:Any,request:TileRequest)->Any:
        if self._generator is None: raise RuntimeError("Call load() before predict()")
        masks=self._generator.generate(rgb_array(tile))
        if not masks: return __import__("numpy").zeros((request.output_size+2*round(request.output_size*request.border_ratio),)*2,dtype="int32")
        import numpy as np
        output=np.zeros_like(masks[0]["segmentation"],dtype=np.int32)
        # Large masks first; later smaller masks claim their unassigned pixels.
        for label,item in enumerate(sorted(masks,key=lambda m:-m["area"]),1): output[item["segmentation"] & (output==0)]=label
        return output
    def postprocess(self,raw_output:Any,request:TileRequest):
        start=time.perf_counter()
        return result(self.model_name,self.version,raw_output,request,str(self._checkpoint),self._device,start,{"prompt_strategy":"automatic mask generator","model_type":self.model_type,"points_per_side":self.points_per_side,"pred_iou_thresh":self.pred_iou_thresh,"stability_score_thresh":self.stability_score_thresh,"limitation":"general SAM masks can merge adjacent cells; validate cell-instance behavior","postprocessing":"mask instances; nearest-neighbor resize; center crop context"})
