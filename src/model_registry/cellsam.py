"""CellSAM adapter using the cell-oriented automatic instance path."""
from __future__ import annotations
import os
import tempfile
import tarfile
import time
from pathlib import Path
from typing import Any
from .base import ModelAdapter, TileRequest
from ._adapter_utils import checked_checkpoint, result, rgb_array

CELLSAM_HF_REPO = "vanvalenlab/cellsam"
CELLSAM_HF_REVISION = "3a2e42cab75a0eb50b9d3c04889a8ea1cdd73904"
CELLSAM_ARCHIVE = "cellsam-models_v1.2.tar.gz"
CELLSAM_VARIANTS = {"cellsam_general", "cellsam_extra"}


def _download_cellsam_checkpoint(model_variant: str, cache_root: Path) -> Path:
    """Download the official gated v1.2 weights from Hugging Face."""
    if model_variant not in CELLSAM_VARIANTS:
        raise ValueError(
            f"unsupported CellSAM variant {model_variant!r}; choose one of "
            f"{', '.join(sorted(CELLSAM_VARIANTS))}"
        )

    destination = cache_root / "cellsam_v1.2" / f"{model_variant}.pt"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination.resolve()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "CellSAM's Hugging Face weight download needs huggingface-hub; "
            "install it in the CellSAM environment."
        ) from exc

    try:
        archive_path = hf_hub_download(
            repo_id=CELLSAM_HF_REPO,
            filename=CELLSAM_ARCHIVE,
            revision=CELLSAM_HF_REVISION,
            repo_type="model",
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not download official CellSAM v1.2 weights from Hugging Face. "
            "Accept the non-commercial access terms at "
            "https://huggingface.co/vanvalenlab/cellsam, then provide an "
            "authorized HF_TOKEN to the job (or run `hf auth login`)."
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_name = f"{model_variant}.pt"
    temporary_path: Path | None = None
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            matches = [
                member for member in archive.getmembers()
                if Path(member.name).name == expected_name and member.isfile()
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one {expected_name} checkpoint in the official "
                    f"CellSAM archive, found {len(matches)}."
                )
            source = archive.extractfile(matches[0])
            if source is None:
                raise RuntimeError(f"Could not read {expected_name} from CellSAM archive")
            with tempfile.NamedTemporaryFile(
                prefix=f".{expected_name}.", suffix=".part",
                dir=destination.parent, delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                while chunk := source.read(1024 * 1024):
                    temporary.write(chunk)
                temporary.flush()
                os.fsync(temporary.fileno())
        if temporary_path.stat().st_size == 0:
            raise RuntimeError(f"Downloaded CellSAM checkpoint is empty: {expected_name}")
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return destination.resolve()


class CellSAMAdapter(ModelAdapter):
    model_name="cellsam"; version="1.0"
    def __init__(
        self, *, predictor: Any = None, model_variant: str = "cellsam_general",
        checkpoint_root: Path | None = None,
    ):
        self._predictor=predictor; self._model=None; self._checkpoint=None
        self._device="cpu"; self.model_variant=model_variant; self._builtin_checkpoint=False
        self.checkpoint_root = checkpoint_root
    def load(self,checkpoint_path:Path|None=None,*,device:str="cpu")->None:
        self._builtin_checkpoint=checkpoint_path is None
        self._checkpoint=(
            checked_checkpoint(checkpoint_path, self.model_name)
            if checkpoint_path is not None
            else None
        )
        self._device=device
        if self._predictor is not None:
            return
        try:
            from cellSAM import get_local_model, segment_cellular_image
        except ImportError as exc:
            raise RuntimeError(
                "CellSAM dependencies are missing. Install the pinned official "
                "CellSAM package in an environment with PyTorch."
            ) from exc
        if self._builtin_checkpoint:
            root = Path(
                self.checkpoint_root
                or os.environ.get(
                    "MPN_AL_CHECKPOINT_ROOT",
                    str(Path(__file__).resolve().parents[2] / "data" / "06_models" / "fm_weights"),
                )
            ).expanduser()
            self._checkpoint = _download_cellsam_checkpoint(
                self.model_variant, root / "cellsam"
            )
        self._model = get_local_model(str(self._checkpoint))
        self._predictor = segment_cellular_image
    def predict(self,tile:Any,request:TileRequest)->Any:
        del request
        if self._predictor is None: raise RuntimeError("Call load() before predict()")
        image=rgb_array(tile)
        if self._model is not None:
            masks, *_ = self._predictor(image, self._model, device=self._device)
        else:
            masks=self._predictor(image, model_path=str(self._checkpoint) if self._checkpoint else None)
        if isinstance(masks,dict): masks=masks.get("masks",masks.get("labels"))
        if masks is None: raise RuntimeError("CellSAM returned no instance map")
        return masks
    def postprocess(self,raw_output:Any,request:TileRequest):
        start=time.perf_counter()
        return result(
            self.model_name,self.version,raw_output,request,
            str(self._checkpoint) if self._checkpoint else "CellSAM v1.2 cellsam_general",
            self._device,start,
            {"upstream":"vanvalenlab/cellSAM",
             "upstream_revision":"362a8e521bcd7313d5cb280bcd4cfc12977c1e9f",
             "upstream_version":"v1.2",
             "weights_repository":CELLSAM_HF_REPO,
             "weights_revision":CELLSAM_HF_REVISION,
             "model_variant":self.model_variant,
             "checkpoint_source":"Hugging Face official v1.2 archive" if self._builtin_checkpoint else "user override",
             "checkpoint_access":"Hugging Face repository access and license acceptance required",
             "cell_scope":"cell segmentation; marrow whole-cell masks not validated",
             "whole_cell_validated":False,
             "postprocessing":"upstream labels; nearest-neighbor resize; center crop context"},
        )
