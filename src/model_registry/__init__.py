"""Shared contracts for foundation-model cell-segmentation adapters."""

from .base import ModelAdapter, ModelResult, TileRequest
from .cellpose_sam import CellposeSAMAdapter
from .cellsam import CellSAMAdapter
from .cellvit import CellViTAdapter
from .patho_sam import PathoSAMAdapter
from .sam import SAMAdapter

__all__ = ["ModelAdapter", "ModelResult", "TileRequest", "PathoSAMAdapter",
           "CellViTAdapter", "CellSAMAdapter", "CellposeSAMAdapter", "SAMAdapter"]
