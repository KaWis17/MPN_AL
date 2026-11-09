from __future__ import annotations

from PIL import Image
import pytest

from model_registry.base import TileRequest
from model_registry.tile_io import border_pixels, extract_tile


def test_border_pixels_are_per_side():
    assert border_pixels(1024, 0) == 0
    assert border_pixels(1024, 0.30) == 307


def test_png_rgb_no_border_and_center_crop(tmp_path):
    path = tmp_path / "rgb.png"
    Image.new("RGBA", (1100, 1050), (12, 34, 56, 255)).save(path)
    tile = extract_tile(TileRequest(path, "fixture", output_size=1024))
    assert tile.image.mode == "RGB"
    assert tile.image.size == (1024, 1024)
    assert tile.central_crop().size == (1024, 1024)
    assert tile.image.getpixel((0, 0)) == (12, 34, 56)


def test_border_extraction_pads_at_image_edges(tmp_path):
    path = tmp_path / "small.png"
    Image.new("RGB", (1024, 1024), (20, 40, 60)).save(path)
    tile = extract_tile(TileRequest(path, "edge", x=0, y=0, output_size=1024, border_ratio=0.30))
    assert tile.image.size == (1638, 1638)
    assert tile.central_crop().size == (1024, 1024)
    assert tile.metadata.requested_box != tile.metadata.readable_box
    assert tile.image.getpixel((0, 0)) == (20, 40, 60)


def test_rejects_unsupported_png_level(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    with pytest.raises(ValueError, match="level=0"):
        extract_tile(TileRequest(path, "image", level=1, output_size=2))


def test_metadata_is_deterministic(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(path)
    request = TileRequest(path, "image", x=2, y=2, output_size=4, border_ratio=0.25)
    first = extract_tile(request)
    second = extract_tile(request)
    assert first.metadata.to_dict() == second.metadata.to_dict()
    assert first.image.tobytes() == second.image.tobytes()
