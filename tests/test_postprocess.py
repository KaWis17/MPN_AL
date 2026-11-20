"""Instance decoding, polygons, per-cell embeddings, GeoJSON and the metrics that score them."""

import json

import numpy as np
import pytest
from conftest import synthetic_cells

from mpn_al.evaluation import aggregate, instance_counts, panoptic_quality
from mpn_al.image_backend import TileRef
from mpn_al.segmenter import (
    LabelSchema,
    cells_to_feature_collection,
    flood_labels,
    hv_to_instances,
    instances_to_cells,
    pool_cell_embeddings,
    read_features,
    segment_nuclei_stain,
    tile_statistics,
    write_geojson,
)
from mpn_al.training import hv_maps, rasterize_annotations

TWO_CLASSES = LabelSchema.from_dict(
    {
        "name": "toy",
        "classes": [{"name": "megakaryocyte", "aliases": ["meg_norm"]}, {"name": "other"}],
        "ignore": ["uncertain"],
        "roi": ["ROI"],
    }
)


def touching_pair(size: int = 96) -> np.ndarray:
    """Two cells that share a straight border: one blob that only the HV maps can split."""
    yy, xx = np.mgrid[:size, :size]
    left = (xx - 36) ** 2 + (yy - 48) ** 2 < 20**2
    right = (xx - 60) ** 2 + (yy - 48) ** 2 < 20**2
    instances = np.zeros((size, size), dtype=np.int32)
    instances[left & (xx < 48)] = 1
    instances[right & (xx >= 48)] = 2
    return instances


def test_flood_labels_meets_on_the_ridge():
    markers = np.zeros((9, 40), dtype=np.int32)
    markers[:, 2], markers[:, 37] = 1, 2
    elevation = np.tile(np.exp(-0.5 * ((np.arange(40) - 14) / 3.0) ** 2), (9, 1)).astype(np.float32)  # ridge at x=14
    labels = flood_labels(markers, elevation, np.ones((9, 40), dtype=bool), np.linspace(0, 1, 20))
    assert (labels > 0).all()
    border = int((labels[4] == 1).sum())
    assert 13 <= border <= 16  # not the geometric middle (20): the ridge decides


def test_flood_labels_respects_the_mask():
    markers = np.zeros((5, 20), dtype=np.int32)
    markers[2, 1] = 7
    mask = np.ones((5, 20), dtype=bool)
    mask[:, 10] = False  # a wall
    labels = flood_labels(markers, np.zeros((5, 20), np.float32), mask, [0.5])
    assert (labels[:, :10] == 7).all() and (labels[:, 10:] == 0).all()


def test_hv_maps_separate_touching_cells():
    truth = touching_pair()
    assert hv_to_instances((truth > 0).astype(np.float32), np.zeros((2, 96, 96), np.float32)).max() == 1  # one blob
    decoded = hv_to_instances((truth > 0).astype(np.float32), hv_maps(truth), mpp=0.25)
    assert decoded.max() == 2
    assert panoptic_quality(truth, decoded)["pq"] > 0.9


def test_perfect_maps_decode_back_to_the_instances():
    _, truth = synthetic_cells(size=384, count=40, seed=3)
    decoded = hv_to_instances((truth > 0).astype(np.float32), hv_maps(truth), mpp=0.25)
    score = panoptic_quality(truth, decoded)
    assert score["dq"] > 0.97 and score["sq"] > 0.9, score


def test_small_objects_are_dropped_by_physical_size():
    prob = np.zeros((64, 64), np.float32)
    prob[10:13, 10:13] = 1.0  # 9 px = 0.56 um^2 at 0.25 um/px: below the 3 um^2 default
    prob[30:50, 30:50] = 1.0
    assert hv_to_instances(prob, np.zeros((2, 64, 64), np.float32), mpp=0.25).max() == 1
    assert hv_to_instances(prob, np.zeros((2, 64, 64), np.float32), mpp=1.0).max() == 2  # 9 um^2 at 1 um/px


def test_cells_carry_slide_coordinates_class_and_confidence():
    instances = np.zeros((64, 64), dtype=np.int32)
    instances[10:20, 30:50] = 1  # 10 rows x 20 columns
    instances[0:8, 0:8] = 2  # touches the tile border
    prob = np.where(instances > 0, 0.9, 0.1).astype(np.float32)
    type_prob = np.zeros((3, 64, 64), np.float32)
    type_prob[1][instances == 1] = 0.8  # megakaryocyte wins on cell 1
    type_prob[2][instances == 1] = 0.2
    type_prob[2][instances == 2] = 1.0
    cells = instances_to_cells(
        instances, scale=(2.0, 2.0), origin=(1000.0, 5000.0), prob=prob, type_prob=type_prob, labels=TWO_CLASSES
    )
    first, second = cells
    assert (first.class_name, second.class_name) == ("megakaryocyte", "other")
    assert first.class_prob == pytest.approx(0.8) and first.prob == pytest.approx(0.9)
    assert first.area == pytest.approx(10 * 20 * 4)  # level-0 px^2
    assert first.centroid == pytest.approx((1000 + 40 * 2, 5000 + 15 * 2))
    xs, ys = first.polygon[:, 0], first.polygon[:, 1]
    assert (xs.min(), xs.max()) == pytest.approx((1000 + 30.5 * 2, 1000 + 49.5 * 2))  # pixel centres, scaled
    assert (ys.min(), ys.max()) == pytest.approx((5000 + 10.5 * 2, 5000 + 19.5 * 2))
    assert (first.touches_border, second.touches_border) == (False, True)

    stats = tile_statistics(cells, instances, prob)
    assert stats["n_cells"] == 2 and stats["n_megakaryocyte"] == 1 and stats["n_border_cells"] == 1
    assert 0 < stats["mean_entropy"] < 1 and stats["uncertain_fraction"] == 0


def test_slivers_without_an_outline_leave_the_cell_list_and_the_mask_together():
    from mpn_al.segmenter.postprocess import drop_instances_without_cells

    instances = np.zeros((40, 40), dtype=np.int32)
    instances[5:15, 5:15] = 1
    instances[20, 3:30] = 2  # one pixel wide: its contour is a line, not a polygon
    instances[25:35, 20:32] = 3
    cells = instances_to_cells(instances)
    assert [cell.label for cell in cells] == [1, 3]
    cleaned = drop_instances_without_cells(instances, cells)
    assert set(np.unique(cleaned)) == {0, 1, 3} and (cleaned[20] == 0).all()
    assert drop_instances_without_cells(cleaned, cells) is cleaned  # nothing to drop: no copy
    # A blob so small that simplification would flatten it keeps its unsimplified outline.
    tiny = np.zeros((10, 10), dtype=np.int32)
    tiny[4:6, 4:6] = 1
    assert len(instances_to_cells(tiny, simplify=5.0)) == 1


def test_stain_baseline_finds_the_nuclei_of_a_synthetic_tile():
    image, truth = synthetic_cells(size=256, count=14, seed=0)
    found = segment_nuclei_stain(image, mpp=0.25)
    score = panoptic_quality(truth, found)
    assert score["dq"] > 0.9 and score["sq"] > 0.8, score
    assert segment_nuclei_stain(np.full((64, 64, 3), 245, np.uint8), mpp=0.25).max() == 0  # empty glass


def test_cell_embedding_is_the_overlap_weighted_mean_of_its_tokens():
    tokens = np.arange(16, dtype=np.float32).reshape(1, 4, 4)  # one feature: the token's own index
    tokens = np.concatenate([tokens, np.ones((1, 4, 4), np.float32)])
    instances = np.zeros((64, 64), dtype=np.int32)
    instances[0:16, 0:32] = 5  # exactly tokens 0 and 1
    instances[16:32, 48:64] = 9  # exactly token 7
    instances[40:48, 0:16] = 2  # half of token 8 only
    embeddings = pool_cell_embeddings(instances, tokens, [5, 9, 2, 77])
    assert embeddings.shape == (4, 2)
    assert embeddings[:, 0] == pytest.approx([0.5, 7.0, 8.0, 0.0])
    assert embeddings[:, 1] == pytest.approx([1.0, 1.0, 1.0, 0.0])  # label 77 is absent: zero vector


def test_geojson_round_trip_through_qupath_format(tmp_path):
    _, truth = synthetic_cells(size=256, count=12, seed=5)
    tile = TileRef("tl_S1_x4096_y8192_s256", x=4096, y=8192, width=256, height=256, mpp=0.25)
    schema = LabelSchema.single_class("megakaryocyte")
    cells = instances_to_cells(truth, origin=(tile.x, tile.y), labels=schema, simplify=0.5)
    collection = cells_to_feature_collection(cells, schema, tile_id=tile.tile_id, mpp=tile.mpp)
    path = write_geojson(tmp_path / "by_tile" / f"{tile.tile_id}.geojson", collection)

    document = json.loads(path.read_text())
    feature = document["features"][0]
    assert document["type"] == "FeatureCollection" and len(document["features"]) == len(cells)
    assert feature["geometry"]["type"] == "Polygon"
    assert feature["geometry"]["coordinates"][0][0] == feature["geometry"]["coordinates"][0][-1]  # closed ring
    assert feature["properties"]["objectType"] == "annotation"
    assert feature["properties"]["classification"] == {"name": "megakaryocyte", "color": [0, 200, 255]}
    assert feature["properties"]["measurements"]["area_um2"] == pytest.approx(cells[0].area * 0.0625, abs=0.01)
    assert all(4096 <= x <= 4352 and 8192 <= y <= 8448 for x, y in feature["geometry"]["coordinates"][0])
    assert collection == cells_to_feature_collection(cells, schema, tile_id=tile.tile_id, mpp=tile.mpp)  # stable ids

    instances, types, ignore = rasterize_annotations(read_features(path), tile, schema)
    score = panoptic_quality(truth, instances)
    assert score["dq"] == 1.0 and score["sq"] > 0.93, score
    assert set(np.unique(types)) == {0, 1} and not ignore.any()


def test_annotation_roles_ignore_roi_alias_and_unknown():
    def box(name, x0, y0, x1, y1):
        ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {"classification": {"name": name}}}

    tile = TileRef("t", x=100, y=200, width=64, height=64)
    features = [
        box("ROI", 100, 200, 148, 264),  # left 48 columns were annotated
        box("meg_norm", 104, 204, 120, 220),  # alias of megakaryocyte
        box("other", 124, 204, 132, 212),
        box("uncertain", 104, 230, 114, 240),
        box("never_heard_of", 124, 230, 134, 240),
        box("other", 900, 900, 910, 910),  # outside the tile: no instance
    ]
    instances, types, ignore = rasterize_annotations(features, tile, TWO_CLASSES)
    assert instances.max() == 2
    assert types[10, 10] == 1 and types[6, 26] == 2
    assert ignore[:, 50:].all()  # outside the ROI
    assert ignore[32, 6] and ignore[32, 26]  # "uncertain" and the unknown class
    assert not ignore[10, 10] and not ignore[50, 40]  # annotated cell, and background inside the ROI
    skipped = rasterize_annotations(features, tile, TWO_CLASSES, unknown="skip")[2]
    assert not skipped[32, 26]
    everything = rasterize_annotations(features[1:3], tile, TWO_CLASSES, unlabeled="ignore")[2]
    assert everything[50, 40] and not everything[10, 10]


def test_panoptic_quality_counts():
    truth = np.zeros((40, 40), dtype=np.int32)
    truth[2:12, 2:12], truth[20:30, 20:30], truth[2:8, 30:36] = 1, 2, 3
    assert panoptic_quality(truth, truth) == pytest.approx(
        {"pq": 1, "dq": 1, "sq": 1, "precision": 1, "recall": 1, "dice": 1, "tp": 3, "fp": 0, "fn": 0}
    )
    predicted = np.zeros_like(truth)
    predicted[2:12, 2:12] = 10  # exact
    predicted[20:30, 22:32] = 4  # shifted by 2: IoU = 80 / 120
    predicted[32:38, 2:8] = 5  # false positive; true cell 3 is missed
    score = panoptic_quality(truth, predicted)
    assert (score["tp"], score["fp"], score["fn"]) == (2, 1, 1)
    assert score["sq"] == pytest.approx((1 + 80 / 120) / 2) and score["dq"] == pytest.approx(2 / 3)
    assert score["pq"] == pytest.approx(score["dq"] * score["sq"])

    ignore = np.zeros_like(truth, dtype=bool)
    ignore[30:, :10] = True  # nobody annotated where the false positive lies
    assert instance_counts(truth, predicted, ignore=ignore)["fp"] == 0
    pooled = aggregate([instance_counts(truth, truth), instance_counts(truth, np.zeros_like(truth))])
    assert pooled["recall"] == 0.5 and pooled["precision"] == 1.0
