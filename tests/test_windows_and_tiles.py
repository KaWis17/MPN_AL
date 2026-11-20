"""Tile readers and the tile -> model-window adapter."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mpn_al.embeddings import TileEncoder, WindowPlanner, crop_window, windows_table
from mpn_al.embeddings.tables import VECTOR, ParquetStreamWriter
from mpn_al.embeddings.tile_encoder import TILE_COLUMNS
from mpn_al.image_backend import (
    ArrayTileSource,
    ImageFolderSource,
    TileRef,
    WsiTileSource,
    parse_tile_id,
    select_tiles,
)

TILE = TileRef("tl_S00042_HE_01_x147456_y66048_s1024", x=147456, y=66048, mpp=0.25)


def test_tile_id_round_trip():
    assert parse_tile_id(TILE.tile_id) == {"slide_id": "S00042_HE_01", "x": 147456, "y": 66048, "size": 1024}
    assert parse_tile_id("PGI_image_67") is None


def test_grid_windows_follow_the_data_layout_formula():
    """224 px at 0.5 um/px on a 0.25 um/px slide is 448 px at level 0: 3x3 windows cover 1024 px."""
    windows = WindowPlanner(mode="grid", patch_size=14, native_size=224, mpp_model=0.5).plan(TILE)
    assert len(windows) == 9
    assert sorted({w.x - TILE.x for w in windows}) == [0, 288, 576]
    for window in windows:
        assert (window.width_level0, window.width_model, window.mpp_model, window.role) == (448, 224, 0.5, "embed")
        assert TILE.x <= window.x and window.x + window.width_level0 <= TILE.x + TILE.width
        assert TILE.y <= window.y and window.y + window.height_level0 <= TILE.y + TILE.height
    assert windows[0].window_id == "win_S00042_HE_01_x147456_y66048_s1024_dx0_dy0_s448_224_mpp0p5"
    assert len({w.window_id for w in windows}) == 9


def test_other_window_modes():
    planner = dict(patch_size=14, native_size=224, mpp_model=0.5)
    (full,) = WindowPlanner(mode="full", **planner).plan(TILE)
    assert (full.width_level0, full.width_model) == (1024, 518)  # 512 px rounded to a multiple of 14
    assert full.mpp_model == pytest.approx(0.25 * 1024 / 518)
    (center,) = WindowPlanner(mode="center", **planner).plan(TILE)
    assert (center.x - TILE.x, center.y - TILE.y, center.width_level0) == (288, 288, 448)
    (thumbnail,) = WindowPlanner(mode="thumbnail", **planner).plan(TILE)
    assert (thumbnail.width_level0, thumbnail.width_model) == (1024, 224)
    # A model that reads at the slide's own resolution needs more windows.
    assert len(WindowPlanner(mode="grid", patch_size=14, native_size=224, mpp_model=0.25).plan(TILE)) == 25
    # Overlap can only add windows.
    assert len(WindowPlanner(mode="grid", overlap=0.5, **planner).plan(TILE)) == 16


def test_missing_resolution_is_an_error_unless_a_default_is_given():
    tile = TileRef("plain_image", width=600, height=400)
    with pytest.raises(ValueError, match="has no mpp"):
        WindowPlanner(mode="grid").plan(tile)
    windows = WindowPlanner(mode="grid", default_mpp=0.25, patch_size=16).plan(tile)
    assert all(w.x + w.width_level0 <= 600 and w.y + w.height_level0 <= 400 for w in windows)
    assert WindowPlanner(mode="thumbnail").plan(tile)[0].mpp_model is None


def test_crop_window_resamples_the_right_region():
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    image[288:736, 576:1024] = 200  # exactly the footprint of the window at dx=576, dy=288
    windows = WindowPlanner(mode="grid", patch_size=14, native_size=224, mpp_model=0.5).plan(TILE)
    target = next(w for w in windows if (w.x - TILE.x, w.y - TILE.y) == (576, 288))
    crop = crop_window(image, TILE, target)
    assert crop.shape == (224, 224, 3) and crop.min() == 200
    with pytest.raises(ValueError, match="but the tile is"):
        crop_window(image[:512], TILE, target)


def test_windows_table_matches_the_schema():
    windows = WindowPlanner(mode="grid", patch_size=14).plan(TILE)
    table = windows_table(windows, prep_id="prep_x")
    documented = ["window_id", "tile_id", "prep_id", "role", "x", "y", "size_level0", "size_model", "mpp_model"]
    assert list(table.columns)[: len(documented)] == documented
    assert table["window_id"].is_unique and (table["tile_id"] == TILE.tile_id).all()


def test_image_folder_source_reads_origin_from_the_file_name(tmp_path):
    rng = np.random.default_rng(0)
    for name in ("tl_S1_x2048_y4096_s64", "loose_image"):
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(tmp_path / f"{name}.png")
    (tmp_path / "notes.txt").write_text("not an image")
    source = ImageFolderSource(tmp_path, mpp=0.25)
    by_id = {t.tile_id: t for t in source.tiles}
    assert set(by_id) == {"tl_S1_x2048_y4096_s64", "loose_image"}
    assert (by_id["tl_S1_x2048_y4096_s64"].x, by_id["tl_S1_x2048_y4096_s64"].y) == (2048, 4096)
    assert by_id["tl_S1_x2048_y4096_s64"].slide_id == "S1"
    assert (by_id["loose_image"].x, by_id["loose_image"].width, by_id["loose_image"].mpp) == (0, 64, 0.25)
    assert source.read(by_id["loose_image"]).shape == (64, 64, 3)
    with pytest.raises(FileNotFoundError):
        ImageFolderSource(tmp_path / "missing")


def test_wsi_source_reads_level0_regions_listed_in_tiles_parquet(tmp_path):
    slide = np.random.default_rng(1).integers(0, 255, (300, 400, 3), dtype=np.uint8)

    class FakeSlide:
        def __init__(self, path):
            self.path = path

        def read_region(self, location, level, size):
            assert level == 0
            x, y = location
            rgba = np.dstack([slide[y : y + size[1], x : x + size[0]], np.full((size[1], size[0]), 255, np.uint8)])
            return Image.fromarray(rgba, "RGBA")

    pd.DataFrame(
        {
            "tile_id": ["tl_S1_x100_y50_s64", "tl_S1_x0_y0_s64", "tl_S1_x200_y200_s64"],
            "prep_id": "prep_a",
            "slide_id": "S1",
            "patient_id": "P1",
            "x": [100, 0, 200],
            "y": [50, 0, 200],
            "width": 64,
            "height": 64,
            "mpp": 0.23,
            "qc_pass": [True, False, True],
            "split": ["train", "train", "test"],
        }
    ).to_parquet(tmp_path / "tiles.parquet")
    pd.DataFrame({"slide_id": ["S1"], "path": ["wsi/P1/S1.ndpi"]}).to_parquet(tmp_path / "slides.parquet")

    source = WsiTileSource(tmp_path / "tiles.parquet", slides_parquet=tmp_path / "slides.parquet", opener=FakeSlide)
    assert [t.tile_id for t in source.tiles] == ["tl_S1_x100_y50_s64", "tl_S1_x200_y200_s64"]  # qc_pass only
    tile = source.tiles[0]
    assert (tile.prep_id, tile.patient_id, tile.mpp) == ("prep_a", "P1", pytest.approx(0.23))
    assert tile.path == str(tmp_path / "wsi" / "P1" / "S1.ndpi")
    assert np.array_equal(source.read(tile), slide[50:114, 100:164])
    train = WsiTileSource(tmp_path / "tiles.parquet", split="train", qc_pass_only=False, opener=FakeSlide)
    assert len(train) == 2
    with pytest.raises(FileNotFoundError, match="no WSI file known"):
        train.read(train.tiles[0])


def test_select_tiles_shards_partition_the_list():
    tiles = [TileRef(f"t{i}") for i in range(10)]
    shards = [select_tiles(tiles, shard_index=i, num_shards=3) for i in range(3)]
    assert sorted(t.tile_id for shard in shards for t in shard) == sorted(t.tile_id for t in tiles)
    assert [t.tile_id for t in select_tiles(tiles, tile_ids=["t3", "t7", "zz"], limit=1)] == ["t3"]


def test_tile_embeddings_are_the_mean_of_window_embeddings_and_survive_parquet(tmp_path, tiny_model):
    rng = np.random.default_rng(0)
    images = {
        "tl_S1_x0_y0_s512": rng.integers(0, 255, (512, 512, 3), dtype=np.uint8),
        "tl_S1_x512_y0_s512": rng.integers(0, 255, (512, 512, 3), dtype=np.uint8),
        "odd_size": rng.integers(0, 255, (448, 448, 3), dtype=np.uint8),
    }
    source = ArrayTileSource(images, mpp=0.5)
    encoder = TileEncoder(tiny_model, window_mode="grid", batch_size=3)  # batches straddle tiles
    results = list(encoder.encode(source))
    assert [r.tile.tile_id for r in results] == list(images)
    assert [len(r.windows) for r in results] == [9, 9, 4]
    for result in results:
        assert result.window_embeddings.shape == (len(result.windows), 192)
        assert np.allclose(result.embedding, result.window_embeddings.mean(axis=0), atol=1e-6)
        single = encoder.encode_tile(images[result.tile.tile_id], result.tile)
        assert np.allclose(single.embedding, result.embedding, atol=1e-4)

    path = tmp_path / "rounds" / "round_00" / "embeddings.parquet"
    with ParquetStreamWriter(path, TILE_COLUMNS, flush_rows=2) as writer:
        for result in results:
            writer.add(result.tile_row("tiny", "cls", "grid"))
    table = pd.read_parquet(path)
    assert list(table["tile_id"]) == list(images)
    assert np.allclose(np.stack(table["embedding"].to_numpy()), np.stack([r.embedding for r in results]))
    assert table["slide_id"].tolist() == ["S1", "S1", None]


def test_interrupted_table_is_not_promoted_to_its_final_name(tmp_path):
    path = tmp_path / "embeddings.parquet"
    with pytest.raises(RuntimeError):
        with ParquetStreamWriter(path, {"tile_id": "string", "embedding": VECTOR}, flush_rows=1) as writer:
            writer.add({"tile_id": "a", "embedding": np.ones(4)})
            raise RuntimeError("job killed")
    assert not path.exists() and path.with_name("embeddings.parquet.partial").exists()
