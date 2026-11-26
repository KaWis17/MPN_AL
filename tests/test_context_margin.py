"""Reading a margin around tiles: cells are kept by centroid, so abutting tiles share no cell."""

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_cells
from PIL import Image

from mpn_al.image_backend import ArrayTileSource, TileRef, WsiTileSource
from mpn_al.segmenter import PipelineConfig, SegmentationPipeline
from mpn_al.segmenter.cli import main as main_segment

MARGIN = 32
PAD = 64


@pytest.fixture()
def slide():
    return synthetic_cells(size=256, count=40, radius=(6, 12))[0]


@pytest.fixture()
def source(tmp_path, slide):
    padded = np.pad(slide, ((PAD, PAD), (PAD, PAD), (0, 0)), constant_values=255)

    class FakeSlide:
        def __init__(self, path):
            pass

        def read_region(self, location, level, size):
            x, y = location[0] + PAD, location[1] + PAD
            rgb = padded[y : y + size[1], x : x + size[0]]
            return Image.fromarray(np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)]), "RGBA")

    rows = [(f"S_x{x}_y{y}_s128", x, y) for y in (0, 128) for x in (0, 128)]
    pd.DataFrame(
        {"tile_id": [r[0] for r in rows], "slide_id": "S", "x": [r[1] for r in rows], "y": [r[2] for r in rows],
         "width": 128, "height": 128, "mpp": 0.25}
    ).to_parquet(tmp_path / "tiles.parquet")
    pd.DataFrame({"slide_id": ["S"], "path": ["S.ndpi"]}).to_parquet(tmp_path / "slides.parquet")
    return WsiTileSource(tmp_path / "tiles.parquet", slides_parquet=tmp_path / "slides.parquet", opener=FakeSlide)


def test_margin_read_surrounds_the_tile(source, slide):
    tile = source.tiles[3]  # bottom-right quadrant: its margin runs off the slide (white)
    image = source.read_with_margin(tile, MARGIN)
    assert image.shape == (128 + 2 * MARGIN, 128 + 2 * MARGIN, 3)
    assert np.array_equal(image[MARGIN:-MARGIN, MARGIN:-MARGIN], source.read(tile))
    assert np.array_equal(image[:MARGIN, MARGIN:-MARGIN], slide[128 - MARGIN : 128, 128:256])  # neighbour above
    assert (image[-MARGIN:] == 255).all()  # beyond the slide
    assert np.array_equal(source.read_with_margin(tile, 0), source.read(tile))


def test_cells_are_assigned_to_exactly_one_tile(source):
    pipeline = SegmentationPipeline(config=PipelineConfig(context_margin=MARGIN, cell_embeddings=False))
    results = [pipeline.run_tile(source.read_with_margin(tile, MARGIN), tile) for tile in source.tiles]
    for tile, result in zip(source.tiles, results):
        assert result.tile is tile and result.cells
        for cell in result.cells:
            assert tile.x <= cell.centroid[0] < tile.x + tile.width and tile.y <= cell.centroid[1] < tile.y + tile.height
        assert result.instances.shape == (128, 128)  # cropped to the tile at working resolution
        assert set(np.unique(result.instances)) - {0} <= {cell.label for cell in result.cells}
        assert result.stats["n_cells"] == len(result.cells)
    centroids = {(round(c.centroid[0], 3), round(c.centroid[1], 3)) for r in results for c in r.cells}
    assert len(centroids) == sum(len(r.cells) for r in results)  # no cell reported by two tiles

    whole = SegmentationPipeline(config=PipelineConfig(cell_embeddings=False)).run_tile(slide_image := np.concatenate(  # the same region segmented in one piece
        [np.concatenate([source.read(source.tiles[0]), source.read(source.tiles[1])], axis=1),
         np.concatenate([source.read(source.tiles[2]), source.read(source.tiles[3])], axis=1)], axis=0),
        TileRef("whole", 0, 0, 256, 256, mpp=0.25))
    assert slide_image.shape[:2] == (256, 256)
    total = sum(len(r.cells) for r in results)
    assert abs(total - len(whole.cells)) <= 0.15 * len(whole.cells)
    assert any(cell.touches_border for r in results for cell in r.cells)  # cells crossing a tile edge are whole


def test_wrong_image_size_is_reported(source):
    pipeline = SegmentationPipeline(config=PipelineConfig(context_margin=MARGIN))
    with pytest.raises(ValueError, match="plus a 32 px margin"):
        pipeline.run_tile(source.read(source.tiles[0]), source.tiles[0])


def test_image_sources_cannot_read_a_margin(tmp_path, slide):
    with pytest.raises(NotImplementedError):
        ArrayTileSource({"a": slide}).read_with_margin(TileRef("a", width=256, height=256), 8)
    Image.fromarray(slide).save(tmp_path / "S_x0_y0_s256.png")
    with pytest.raises(SystemExit):
        main_segment(["--baseline", "stain", "--images", str(tmp_path), "--mpp", "0.25", "--out", str(tmp_path / "o"),
                      "--context-margin", "16"])


def test_cli_runs_with_a_margin_on_wsi_tiles(source, tmp_path, monkeypatch):
    from mpn_al.segmenter import cli

    monkeypatch.setattr(cli, "source_from_arguments", lambda args: (source, source.tiles))
    out = tmp_path / "out"
    code = main_segment(["--baseline", "stain", "--images", str(tmp_path), "--out", str(out), "--context-margin", str(MARGIN),
                         "--no-cell-embeddings"])
    assert code == 0
    stats = pd.read_json(out / "tile_stats.jsonl", lines=True)
    assert len(stats) == 4 and (stats["n_cells"] > 0).all()
