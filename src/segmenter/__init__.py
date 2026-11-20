"""Cell segmentation with SSL base models: decoder head, instance decoding, GeoJSON, per-cell embeddings."""

from .cell_features import pool_cell_embeddings
from .decoder import DECODER_STRIDE, CellDecoder
from .geojson import cells_to_feature_collection, read_features, write_geojson
from .labels import LabelClass, LabelSchema
from .model import CellSegmenter, SegmentationMaps, load_segmenter, predict_maps, read_head, save_head
from .pipeline import PipelineConfig, RunWriter, SegmentationPipeline, TileResult, dense_tokens, read_tile_stats
from .postprocess import Cell, PostprocessConfig, flood_labels, hv_to_instances, instances_to_cells, tile_statistics
from .stain_baseline import StainBaselineConfig, segment_nuclei_stain

__all__ = [
    "DECODER_STRIDE",
    "Cell",
    "CellDecoder",
    "CellSegmenter",
    "LabelClass",
    "LabelSchema",
    "PipelineConfig",
    "PostprocessConfig",
    "RunWriter",
    "SegmentationMaps",
    "SegmentationPipeline",
    "StainBaselineConfig",
    "TileResult",
    "cells_to_feature_collection",
    "dense_tokens",
    "flood_labels",
    "hv_to_instances",
    "instances_to_cells",
    "load_segmenter",
    "pool_cell_embeddings",
    "predict_maps",
    "read_features",
    "read_head",
    "read_tile_stats",
    "save_head",
    "segment_nuclei_stain",
    "tile_statistics",
    "write_geojson",
]
