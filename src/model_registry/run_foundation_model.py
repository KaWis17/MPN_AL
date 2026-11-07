"""Run one foundation-model adapter on a single PNG or NDPI tile.

Example::

    python -m src.model_registry.run_foundation_model --model pathosam \
        --input tile.png --output data/03_predictions/foundation_model_runs

The dispatcher owns I/O and provenance only; model behavior stays in adapters.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import inspect
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .base import SUPPORTED_MODELS, ModelResult, TileRequest
from .tile_io import extract_tile


_ADAPTERS = {
    "pathosam": (".patho_sam", "PathoSAMAdapter"),
    "cellvit": (".cellvit", "CellViTAdapter"),
    "cellsam": (".cellsam", "CellSAMAdapter"),
    "cellpose_sam": (".cellpose_sam", "CellposeSAMAdapter"),
    "sam": (".sam", "SAMAdapter"),
}
_ALIASES = {
    "patho-sam": "pathosam", "cellvit++": "cellvit", "cell-sam": "cellsam",
    "cellpose-sam": "cellpose_sam", "segment-anything": "sam",
}


def _model_name(value: str) -> str:
    name = _ALIASES.get(value.strip().lower(), value.strip().lower())
    if name not in SUPPORTED_MODELS:
        raise argparse.ArgumentTypeError(
            f"unsupported model {value!r}; choose one of {', '.join(SUPPORTED_MODELS)}"
        )
    return name


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=_model_name, help="stable model identifier")
    parser.add_argument("--list-models", action="store_true", help="list stable model IDs and exit")
    parser.add_argument("--input", type=Path, help="source .png fixture or .ndpi WSI")
    parser.add_argument("--output", type=Path,
                        help="run directory, or parent directory under which a unique run is created")
    parser.add_argument("--run-id", help="explicit run directory name (never overwrites)")
    parser.add_argument("--x", type=int, help="level-0 x coordinate of canonical tile top-left")
    parser.add_argument("--y", type=int, help="level-0 y coordinate of canonical tile top-left")
    parser.add_argument("--level", type=int, default=0, help="pyramid level (coordinates remain level 0)")
    parser.add_argument("--border", choices=("none", "0.30"), default="none",
                        help="context added on each side of the canonical tile")
    parser.add_argument("--padding", choices=("reflect", "edge", "constant"), default="reflect")
    parser.add_argument("--checkpoint", type=Path, help="explicit model checkpoint; never downloaded")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tile-id", help="optional stable tile_id; defaults to source and coordinates")
    parser.add_argument("--image-id", help="optional image/slide ID; defaults to input stem")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                        help="model-specific override, repeatable; values are parsed as JSON when possible")
    parser.add_argument("--dry-run", action="store_true",
                        help="extract and validate the tile, then write a clearly marked zero-mask contract run")
    return parser


def _parse_overrides(values: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in values:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"invalid --param {item!r}; expected KEY=VALUE")
        key = key.strip()
        if key in parsed:
            raise ValueError(f"model override {key!r} was provided more than once")
        try:
            parsed[key] = json.loads(value)
        except json.JSONDecodeError:
            parsed[key] = value
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("--device cuda requested, but CUDA is unavailable")
            except ImportError as exc:
                raise RuntimeError("--device cuda requested, but PyTorch is not installed") from exc
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _run_id(model: str, source_digest: str, request: TileRequest, overrides: dict[str, Any], seed: int) -> str:
    identity = {"model": model, "sha256": source_digest, "image_id": request.image_id,
                "x": request.x, "y": request.y, "level": request.level,
                "border": request.border_ratio, "seed": seed, "overrides": overrides}
    suffix = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    return f"{model}_{suffix}"


def _save_mask(path: Path, array: Any, *, instance: bool) -> None:
    data = np.asarray(array)
    if data.ndim != 2:
        raise ValueError(f"postprocessed mask must be 2D, got shape {data.shape}")
    if instance:
        if not np.issubdtype(data.dtype, np.integer) or np.any(data < 0):
            raise ValueError("instance_mask must contain non-negative integer labels")
        Image.fromarray(data.astype(np.int32), mode="I").save(path, format="TIFF")
    else:
        Image.fromarray((data > 0).astype(np.uint8) * 255, mode="L").save(path, format="PNG")


def run(args: argparse.Namespace) -> Path:
    if not args.model or not args.input:
        raise ValueError("--model and --input are required (unless --list-models is used)")
    if (args.x is None) != (args.y is None):
        raise ValueError("--x and --y must be supplied together")
    args.input = args.input.expanduser().resolve()
    if args.input.suffix.lower() == ".ndpi" and args.x is None:
        raise ValueError("NDPI input requires explicit level-0 --x and --y coordinates")
    overrides = _parse_overrides(args.param)
    request = TileRequest(input_path=args.input, image_id=args.image_id or args.input.stem,
        level=args.level, x=args.x, y=args.y, border_ratio=0 if args.border == "none" else 0.30,
        padding_policy=args.padding, tile_id=args.tile_id)
    tile = extract_tile(request)
    device = "cpu" if args.dry_run else _device(args.device)
    source_digest = _sha256(args.input)
    run_id = args.run_id or _run_id(args.model, source_digest, request, overrides, args.seed)
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("--run-id must be a single safe directory name")
    if args.output is not None and args.run_id is None:
        destination = args.output.expanduser()
    else:
        parent = (args.output or Path("data/03_predictions/foundation_model_runs")).expanduser()
        destination = parent / run_id
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"run directory already exists; refusing to overwrite: {destination}")

    started = time.perf_counter()
    result_obj: ModelResult | None = None
    raw = None
    if args.dry_run:
        labels = np.zeros((request.output_size, request.output_size), dtype=np.int32)
        result_obj = ModelResult(labels, labels.astype(np.uint8), args.model, "dry-run",
            tile_id=request.tile_id, image_id=request.image_id, level=request.level,
            x=tile.metadata.x, y=tile.metadata.y, output_size=request.output_size,
            border_ratio=request.border_ratio, padding_policy=request.padding_policy,
            input_shape=tile.metadata.input_shape, output_shape=(request.output_size, request.output_size),
            device=device, elapsed_seconds=0.0, warnings=["dry-run: no model inference performed"],
            metadata={"dry_run": True})
    else:
        module_name, class_name = _ADAPTERS[args.model]
        adapter_cls = getattr(__import__(module_name, globals(), locals(), [class_name], 1), class_name)
        accepted = inspect.signature(adapter_cls).parameters
        unknown = sorted(set(overrides) - set(accepted))
        if unknown:
            raise ValueError(f"unsupported override(s) for {args.model}: {', '.join(unknown)}")
        adapter = adapter_cls(**overrides)
        # Seed the supported random sources before model loading/inference.
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        try:
            import torch
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
        except ImportError:
            pass
        adapter.load(args.checkpoint, device=device)
        raw = adapter.predict(tile.image, request)
        result_obj = adapter.postprocess(raw, request)
        if result_obj.model_name != args.model:
            raise ValueError(f"adapter returned model_name={result_obj.model_name!r}; expected {args.model!r}")

    assert result_obj is not None
    if tuple(np.asarray(result_obj.instance_mask).shape) != (request.output_size, request.output_size):
        raise ValueError("adapter instance mask does not match the canonical output shape")
    if tuple(np.asarray(result_obj.binary_mask).shape) != (request.output_size, request.output_size):
        raise ValueError("adapter binary mask does not match the canonical output shape")
    destination.mkdir(parents=True, exist_ok=False)
    # Persist the actual model input (including context) for reproducible review.
    tile.image.save(destination / "input.png", format="PNG")
    _save_mask(destination / "instance_mask.tif", result_obj.instance_mask, instance=True)
    _save_mask(destination / "binary_mask.png", result_obj.binary_mask, instance=False)

    result_obj.tile_id = request.tile_id or f"{request.image_id}_L{request.level}_x{tile.metadata.x}_y{tile.metadata.y}"
    metadata = {
        "schema_version": 1, "run_id": run_id, "model": args.model,
        "adapter_version": result_obj.adapter_version, "dry_run": bool(args.dry_run),
        "source": {"path": str(args.input), "sha256": source_digest,
                   "format": args.input.suffix.lower()},
        "tile": {**tile.metadata.to_dict(), "tile_id": result_obj.tile_id},
        "request": {"level": request.level, "x": tile.metadata.x, "y": tile.metadata.y,
                    "output_size": request.output_size, "border_ratio": request.border_ratio,
                    "border_semantics": "per_side", "padding_policy": request.padding_policy},
        "inference": {"device": device, "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
                      "checkpoint_sha256": _sha256(args.checkpoint.resolve()) if args.checkpoint and args.checkpoint.is_file() else None,
                      "seed": args.seed, "model_overrides": overrides,
                      "elapsed_seconds": result_obj.elapsed_seconds,
                      "wall_seconds": time.perf_counter() - started,
                      "input_shape": result_obj.input_shape, "output_shape": result_obj.output_shape,
                      "resize_scale": result_obj.resize_scale, "warnings": result_obj.warnings,
                      "adapter_metadata": result_obj.metadata},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "pillow": _version("Pillow"),
                        "torch": _version("torch")},
        "outputs": {"input.png": _sha256(destination / "input.png"),
                    "instance_mask.tif": _sha256(destination / "instance_mask.tif"),
                    "binary_mask.png": _sha256(destination / "binary_mask.png")},
    }
    (destination / "run.json").write_text(json.dumps(_jsonable(metadata), indent=2) + "\n", encoding="utf-8")
    (destination / "README.md").write_text(
        f"# {run_id}\n\n- Model: `{args.model}`\n- Tile: `{result_obj.tile_id}`\n"
        f"- Border: `{args.border}` (per side)\n- Dry run: `{bool(args.dry_run)}`\n"
        f"- Runtime: `{result_obj.elapsed_seconds or 0:.3f} s`\n\nSee `run.json` for provenance and checksums.\n",
        encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.list_models:
        print("\n".join(SUPPORTED_MODELS))
        return 0
    try:
        path = run(args)
    except (ValueError, RuntimeError, FileNotFoundError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
