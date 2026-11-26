"""GeoJSON in the dialect QuPath imports and exports.

One ``FeatureCollection`` per tile, polygons in level-0 slide coordinates, the class under
``properties.classification``. Proposals are written as *annotation* objects by default,
because the pathologist has to be able to edit them; QuPath detections are read-only shapes.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .labels import LabelSchema
from .postprocess import Cell

OBJECT_TYPES = ("annotation", "detection", "cell")


def cells_to_feature_collection(
    cells: Sequence[Cell],
    labels: LabelSchema | None = None,
    *,
    tile_id: str = "",
    object_type: str = "annotation",
    mpp: float | None = None,
) -> dict[str, Any]:
    """Build the FeatureCollection for one tile. Feature ids are stable for a given tile and cell order."""
    if object_type not in OBJECT_TYPES:
        raise ValueError(f"object_type must be one of {OBJECT_TYPES}, got {object_type!r}")
    features = []
    for index, cell in enumerate(cells):
        ring = [[round(float(x), 1), round(float(y), 1)] for x, y in cell.polygon]
        ring.append(ring[0])  # GeoJSON rings are closed
        measurements: dict[str, float] = {"area_px": round(cell.area, 1)}
        if mpp is not None:
            measurements["area_um2"] = round(cell.area * mpp * mpp, 2)
        if cell.prob is not None:
            measurements["prob"] = round(cell.prob, 4)
        if cell.class_prob is not None:
            measurements["class_prob"] = round(cell.class_prob, 4)
        if cell.touches_border:
            measurements["touches_border"] = 1.0
        properties: dict[str, Any] = {"objectType": object_type, "measurements": measurements}
        if cell.class_name is not None:
            classification: dict[str, Any] = {"name": cell.class_name}
            if labels is not None and cell.class_id is not None:
                classification["color"] = list(labels.by_id(cell.class_id).color)
            properties["classification"] = classification
        features.append(
            {
                "type": "Feature",
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"mpn_al/{tile_id}/{index}")),
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": properties,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def write_geojson(path: str | Path, collection: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(collection, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)  # readers never see a half-written file
    return path


def read_features(path: str | Path) -> list[dict[str, Any]]:
    """Features of a GeoJSON file, whether it holds a FeatureCollection, a list or a single Feature."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(document, list):
        return document
    if document.get("type") == "FeatureCollection":
        return list(document.get("features", []))
    return [document]


def feature_class_name(feature: dict[str, Any]) -> str | None:
    """Class name of a QuPath feature (``properties.classification.name``), if it has one."""
    properties = feature.get("properties") or {}
    classification = properties.get("classification")
    if isinstance(classification, dict):
        return classification.get("name")
    if isinstance(classification, str):
        return classification
    return properties.get("class") or properties.get("name")


def feature_polygons(feature: dict[str, Any]) -> Iterable[list[np.ndarray]]:
    """Yield each polygon of a feature as ``[exterior, hole, ...]`` arrays of ``[n, 2]`` float64 points."""
    geometry = feature.get("geometry") or {}
    kind, coordinates = geometry.get("type"), geometry.get("coordinates") or []
    if kind == "Polygon":
        polygons = [coordinates]
    elif kind == "MultiPolygon":
        polygons = coordinates
    else:
        return
    for rings in polygons:
        parsed = [np.asarray(ring, dtype=np.float64)[:, :2] for ring in rings if len(ring) >= 3]
        if parsed:
            yield parsed
