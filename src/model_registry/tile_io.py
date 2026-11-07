"""Load PNG/NDPI images and extract canonical, border-aware tile fields of view."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .base import TileRequest


@dataclass
class TileMetadata:
    source_path: str
    image_id: str
    level: int
    x: int
    y: int
    requested_box: tuple[int, int, int, int]
    readable_box: tuple[int, int, int, int]
    level_downsample: float
    output_size: int
    border_ratio: float
    border_pixels: int
    input_shape: tuple[int, int, int]
    mpp: float | None = None
    padding_policy: str = "reflect"
    padding_fallback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Tile:
    """RGB inference image and its provenance. Image size includes context."""

    image: Image.Image
    metadata: TileMetadata

    def central_crop(self) -> Image.Image:
        size = self.metadata.output_size
        border = self.metadata.border_pixels
        return self.image.crop((border, border, border + size, border + size))


def border_pixels(output_size: int, ratio: float) -> int:
    """Return rounded per-side context width (30% means 30% on each side)."""
    if output_size <= 0 or ratio < 0:
        raise ValueError("output_size must be positive and border_ratio non-negative")
    return int(round(output_size * ratio))


def _load_source(path: Path):
    if path.suffix.lower() == ".png":
        image = Image.open(path).convert("RGB")
        return image, image.size, 1.0, None, lambda box, size: image.crop(box).convert("RGB")
    try:
        import openslide
    except ImportError as exc:
        raise RuntimeError(
            "Reading NDPI requires OpenSlide and its native libraries; install "
            "openslide-python plus the OpenSlide system package."
        ) from exc
    slide = openslide.OpenSlide(str(path))
    mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    mpp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
    mpp = float(mpp_x) if mpp_x and (not mpp_y or float(mpp_x) == float(mpp_y)) else None
    downsamples = slide.level_downsamples

    def read(box, size):
        left, top, right, bottom = box
        # region coordinates use level-0 units; output width/height use level pixels.
        return slide.read_region((left, top), size[2], (size[0], size[1])).convert("RGB")

    return slide, slide.level_dimensions[0], downsamples, mpp, read


def _pad_image(image: Image.Image, left: int, top: int, right: int, bottom: int,
               policy: str) -> tuple[Image.Image, str | None]:
    """Pad an already-read image. Reflection uses mirrored edge strips."""
    if not any((left, top, right, bottom)):
        return image, None
    fallback = None
    if policy == "constant":
        out = Image.new("RGB", (image.width + left + right, image.height + top + bottom))
        out.paste(image, (left, top))
        return out, None
    # Pillow has no general reflect padding primitive. Construct an extended
    # reflected canvas with periodic mirrored copies, including tiny images.
    if image.width < 2 or image.height < 2:
        policy, fallback = "edge", "reflect padding requires at least 2 pixels per dimension; used edge padding"
    if policy == "edge":
        extended = Image.new("RGB", (image.width + left + right, image.height + top + bottom))
        extended.paste(image, (left, top))
        if left:
            extended.paste(image.crop((0, 0, 1, image.height)).resize((left, image.height)), (0, top))
        if right:
            extended.paste(image.crop((image.width - 1, 0, image.width, image.height)).resize((right, image.height)), (left + image.width, top))
        if top:
            extended.paste(extended.crop((0, top, extended.width, top + 1)).resize((extended.width, top)), (0, 0))
        if bottom:
            extended.paste(extended.crop((0, top + image.height - 1, extended.width, top + image.height)).resize((extended.width, bottom)), (0, top + image.height))
        return extended, fallback

    # Tile mirrored copies, then crop to the requested padded canvas.
    tile_w, tile_h = image.width * 2 - 2, image.height * 2 - 2
    if tile_w <= 0 or tile_h <= 0:
        return _pad_image(image, left, top, right, bottom, "edge")
    pattern = Image.new("RGB", (tile_w, tile_h))
    pattern.paste(image, (0, 0))
    pattern.paste(ImageOps.mirror(image.crop((1, 0, image.width, image.height))), (image.width - 1, 0))
    pattern.paste(ImageOps.flip(image.crop((0, 1, image.width, image.height))), (0, image.height - 1))
    pattern.paste(ImageOps.flip(ImageOps.mirror(image.crop((1, 1, image.width, image.height)))), (image.width - 1, image.height - 1))
    out = Image.new("RGB", (image.width + left + right, image.height + top + bottom))
    # Repeat the small mirrored pattern and crop so source pixel (0, 0) lands
    # at (left, top). Work is in image blocks, not per-pixel Python loops.
    canvas_w, canvas_h = out.size
    for py in range(-((top // tile_h) + 1) * tile_h, canvas_h, tile_h):
        for px in range(-((left // tile_w) + 1) * tile_w, canvas_w, tile_w):
            out.paste(pattern, (px + left, py + top))
    return out, fallback


def extract_tile(request: TileRequest) -> Tile:
    """Extract an RGB FOV. x/y are level-0 top-left coordinates; None centers."""
    if not request.input_path.is_file():
        raise FileNotFoundError(request.input_path)
    source, level0_dimensions, downsamples, mpp, reader = _load_source(request.input_path)
    if request.input_path.suffix.lower() == ".png":
        dimensions = level0_dimensions
        downsample = 1.0
        if request.level != 0:
            raise ValueError("PNG inputs only support level=0")
        mpp = None
    else:
        if request.level >= len(downsamples):
            raise ValueError(f"level {request.level} is unavailable; slide has {len(downsamples)} levels")
        dimensions = source.level_dimensions[request.level]
        downsample = float(downsamples[request.level])
    size = request.output_size
    extra = border_pixels(size, request.border_ratio)
    wanted = size + 2 * extra
    if request.x is None:
        cx, cy = dimensions[0] // 2, dimensions[1] // 2
        left, top = cx - wanted // 2, cy - wanted // 2
        canonical_x, canonical_y = round((left + extra) * downsample), round((top + extra) * downsample)
    else:
        canonical_x, canonical_y = request.x, request.y
        left, top = round(canonical_x / downsample) - extra, round(canonical_y / downsample) - extra
    right, bottom = left + wanted, top + wanted
    clipped = (max(0, left), max(0, top), min(dimensions[0], right), min(dimensions[1], bottom))
    read_w, read_h = clipped[2] - clipped[0], clipped[3] - clipped[1]
    if read_w <= 0 or read_h <= 0:
        raise ValueError("requested tile does not intersect the image")
    if request.input_path.suffix.lower() == ".png":
        raw = reader(clipped, (read_w, read_h))
    else:
        raw = source.read_region((round(clipped[0] * downsample), round(clipped[1] * downsample)), request.level, (read_w, read_h)).convert("RGB")
    pads = (clipped[0] - left, clipped[1] - top, right - clipped[2], bottom - clipped[3])
    padded, fallback = _pad_image(raw, *pads, request.padding_policy)
    if padded.size != (wanted, wanted):
        raise RuntimeError(f"internal extraction size error: expected {(wanted, wanted)}, got {padded.size}")
    metadata = TileMetadata(
        source_path=str(request.input_path.resolve()), image_id=request.image_id,
        level=request.level, x=canonical_x, y=canonical_y,
        requested_box=(round(left * downsample), round(top * downsample), round(right * downsample), round(bottom * downsample)),
        readable_box=(round(clipped[0] * downsample), round(clipped[1] * downsample), round(clipped[2] * downsample), round(clipped[3] * downsample)),
        level_downsample=downsample, output_size=size, border_ratio=request.border_ratio,
        border_pixels=extra, input_shape=(wanted, wanted, 3), mpp=mpp,
        padding_policy=request.padding_policy, padding_fallback=fallback,
    )
    return Tile(padded, metadata)
