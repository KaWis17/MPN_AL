"""SSL base models: registry, loading, hidden representations and tile embeddings."""

from .base_model import BaseModelAccessError, HiddenRepresentations, SSLBaseModel, build_base_model
from .registry import BaseModelSpec, list_specs, load_registry_file, register, resolve_spec
from .tile_encoder import TileEmbedding, TileEncoder
from .windows import WINDOW_MODES, Window, WindowPlanner, crop_window, windows_table

__all__ = [
    "WINDOW_MODES",
    "BaseModelAccessError",
    "BaseModelSpec",
    "HiddenRepresentations",
    "SSLBaseModel",
    "TileEmbedding",
    "TileEncoder",
    "Window",
    "WindowPlanner",
    "build_base_model",
    "crop_window",
    "list_specs",
    "load_registry_file",
    "register",
    "resolve_spec",
    "windows_table",
]
