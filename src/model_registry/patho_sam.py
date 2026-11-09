"""PathoSAM adapter using the library's built-in histopathology model by default."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from ._adapter_utils import checked_checkpoint, result, rgb_array
from .base import ModelAdapter, TileRequest


class PathoSAMAdapter(ModelAdapter):
    model_name = "pathosam"
    version = "1.0"

    def __init__(
        self,
        *,
        predictor: Callable[..., Any] | None = None,
        model_type: str = "vit_b_histopathology",
    ) -> None:
        self._predictor = predictor
        self._model_type = model_type
        self._checkpoint: Path | None = None
        self._device = "cpu"
        self._prediction_started: float | None = None

    def load(self, checkpoint_path: Path | None = None, *, device: str = "cpu") -> None:
        self._checkpoint = (
            checked_checkpoint(checkpoint_path, self.model_name)
            if checkpoint_path is not None
            else None
        )
        self._device = device
        if self._predictor is not None:
            return

        try:
            from patho_sam.automatic_segmentation import automatic_segmentation_wsi
        except ImportError as exc:
            raise RuntimeError(
                "PathoSAM dependencies are unavailable. Install the upstream "
                "patho-sam environment, including micro_sam and PyTorch."
            ) from exc
        self._predictor = automatic_segmentation_wsi

    def predict(self, tile: Any, _request: TileRequest) -> Any:
        del _request
        if self._predictor is None:
            raise RuntimeError("Call load() before predict()")
        self._prediction_started = time.perf_counter()
        try:
            return self._predictor(
                rgb_array(tile),
                model_type=self._model_type,
                checkpoint_path=str(self._checkpoint) if self._checkpoint else None,
                device=self._device,
                output_choice="instances",
                tile_shape=(384, 384),
                halo=(64, 64),
                batch_size=1,
            )
        except Exception as exc:
            raise RuntimeError(f"PathoSAM inference failed: {exc}") from exc

    def postprocess(self, raw_output: Any, request: TileRequest):
        started = self._prediction_started
        self._prediction_started = None
        if started is None:
            started = time.perf_counter()
        if isinstance(raw_output, dict):
            raw_output = raw_output.get("instance_mask", raw_output.get("instances"))
        if raw_output is None:
            raise RuntimeError("PathoSAM did not return cell instances")
        return result(
            self.model_name,
            self.version,
            raw_output,
            request,
            str(self._checkpoint) if self._checkpoint else None,
            self._device,
            started,
            {
                "model_type": self._model_type,
                "weights": "custom checkpoint" if self._checkpoint else "PathoSAM built-in weights",
                "postprocessing": "nearest-neighbor resize; center crop context",
            },
        )
