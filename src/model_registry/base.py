"""Dependency-light contracts shared by foundation-model adapters.

Adapters receive canonical RGB tiles and return instance labels on the same
field of view. Model-specific libraries belong in each adapter module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


SUPPORTED_MODELS = ("pathosam", "cellvit", "cellsam", "cellpose_sam", "sam")
SUPPORTED_INPUT_FORMATS = (".ndpi", ".png")


@dataclass(frozen=True)
class TileRequest:
    """Describe one canonical field of view from an image or WSI.

    ``x`` and ``y`` identify the top-left level-0 origin. ``None`` is allowed
    for fixture-driven center crops. A border ratio is per side (0.30 adds
    30% of ``output_size`` on each side); extraction/padding is implemented by
    the later tile I/O step.
    """

    input_path: Path | str
    image_id: str
    level: int = 0
    x: int | None = None
    y: int | None = None
    output_size: int = 1024
    border_ratio: float = 0.0
    padding_policy: str = "reflect"
    tile_id: str | None = None
    prep_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_path", Path(self.input_path))
        if not self.image_id.strip():
            raise ValueError("image_id must be a non-empty identifier")
        if self.level < 0:
            raise ValueError("level must be >= 0")
        if (self.x is None) != (self.y is None):
            raise ValueError("x and y must either both be set or both be None")
        if self.x is not None and (self.x < 0 or self.y is None or self.y < 0):
            raise ValueError("x and y must be non-negative level-0 coordinates")
        if self.output_size <= 0:
            raise ValueError("output_size must be positive")
        if self.border_ratio < 0:
            raise ValueError("border_ratio must be non-negative")
        if self.padding_policy not in {"reflect", "edge", "constant"}:
            raise ValueError("padding_policy must be reflect, edge, or constant")
        if self.input_path.suffix.lower() not in SUPPORTED_INPUT_FORMATS:
            raise ValueError(
                f"Unsupported input format {self.input_path.suffix!r}; "
                f"expected one of {SUPPORTED_INPUT_FORMATS}"
            )


@dataclass
class ModelResult:
    """Normalized prediction plus provenance for a single tile inference."""

    instance_mask: Any
    binary_mask: Any
    model_name: str
    adapter_version: str
    tile_id: str | None = None
    image_id: str | None = None
    level: int = 0
    x: int | None = None
    y: int | None = None
    output_size: int = 1024
    border_ratio: float = 0.0
    padding_policy: str = "reflect"
    confidence: Any | None = None
    scores: Any | None = None
    input_shape: tuple[int, ...] | None = None
    output_shape: tuple[int, ...] | None = None
    resize_scale: float | tuple[float, float] | None = None
    device: str = "cpu"
    checkpoint_id: str | None = None
    elapsed_seconds: float | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown stable model identifier: {self.model_name!r}")
        expected = (self.output_size, self.output_size)
        for name, value in (("instance_mask", self.instance_mask), ("binary_mask", self.binary_mask)):
            shape = getattr(value, "shape", None)
            if shape is None or tuple(shape[:2]) != expected:
                raise ValueError(f"{name} must have shape {expected}; got {shape}")
        if self.output_shape is None:
            self.output_shape = expected
        elif tuple(self.output_shape[:2]) != expected:
            raise ValueError(f"output_shape must begin with {expected}")
        if self.elapsed_seconds is not None and self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")


class ModelAdapter(ABC):
    """Base interface; adapters do not parse CLI arguments or choose paths."""

    model_name: str
    version: str

    @abstractmethod
    def load(self, checkpoint_path: Path | None = None, *, device: str = "cpu") -> None:
        """Load model dependencies and weights; fail clearly if unavailable."""

    @abstractmethod
    def predict(self, tile: Any, request: TileRequest) -> Any:
        """Run inference on an RGB tile and return model-native output."""

    @abstractmethod
    def postprocess(self, raw_output: Any, request: TileRequest) -> ModelResult:
        """Normalize model output to the requested instance and binary masks."""
