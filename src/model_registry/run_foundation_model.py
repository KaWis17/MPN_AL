"""Run one foundation-model adapter on a PNG/NDPI tile or a PNG tile directory.

Example::

    python -m src.model_registry.run_foundation_model --model cellsam \
        --input tile.png

    python -m src.model_registry.run_foundation_model --model pathosam \
        --input src/model_registry/test_data/tiles \

Results are written under ``segment_result/<model>/result/`` by default.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .base import SUPPORTED_MODELS, ModelResult, TileRequest
from ._scale import prepare_model_input, restore_model_result
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
_DEFAULT_CHECKPOINT_ROOT = Path(__file__).resolve().parents[2] / "data" / "06_models" / "fm_weights"
_CHECKPOINT_SUFFIXES = {".bin", ".ckpt", ".model", ".npy", ".pt", ".pth", ".safetensors"}


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
    parser.add_argument("--input", type=Path,
                        help="source .png/.ndpi file or a directory of named PNG tiles")
    parser.add_argument("--input-list", type=Path,
                        help="text file containing one PNG/NDPI input path per line")
    parser.add_argument("--output", type=Path,
                        help="run directory for one tile; batch input uses it as the parent directory")
    parser.add_argument("--output-base", type=Path,
                        help="parent directory for batch tile outputs")
    parser.add_argument("--run-id", help="explicit run directory name (never overwrites)")
    parser.add_argument("--x", type=int, help="level-0 x coordinate of canonical tile top-left")
    parser.add_argument("--y", type=int, help="level-0 y coordinate of canonical tile top-left")
    parser.add_argument("--level", type=int, default=0, help="pyramid level (coordinates remain level 0)")
    parser.add_argument("--border", choices=("none", "0.30"), default="none",
                        help="context added on each side of the canonical tile")
    parser.add_argument("--padding", choices=("reflect", "edge", "constant"), default="reflect")
    parser.add_argument("--checkpoint", type=Path,
                        help="use a specific compatible local model checkpoint")
    parser.add_argument("--checkpoint-root", type=Path,
                        help="cache root for auto-downloaded SAM weights (default: "
                             "MPN_AL_CHECKPOINT_ROOT or data/06_models/fm_weights)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--mpp", type=float,
                        help="source microns-per-pixel (required for physical scale normalization of PNG tiles)")
    parser.add_argument("--target-mpp", type=float,
                        help="override model reference microns-per-pixel; defaults to a verified model setting when available")
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


def _named_tile_coordinates(path: Path) -> tuple[str, int, int] | None:
    match = re.fullmatch(r"(.+)_x(\d+)_y(\d+)", path.stem)
    if match is None:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def _resolve_checkpoint(model: str, checkpoint: Path | None,
                        checkpoint_root: Path | None = None) -> Path:
    if checkpoint is not None:
        resolved = checkpoint.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"{model} checkpoint does not exist: {resolved}")
        return resolved

    configured_root = checkpoint_root or (
        Path(os.environ["MPN_AL_CHECKPOINT_ROOT"])
        if os.environ.get("MPN_AL_CHECKPOINT_ROOT")
        else _DEFAULT_CHECKPOINT_ROOT
    )
    model_dir = configured_root.expanduser().resolve() / model
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"No checkpoint directory for {model}: {model_dir}. "
            "Place the model weights there, set --checkpoint-root or "
            "MPN_AL_CHECKPOINT_ROOT, or pass --checkpoint."
        )
    candidates = sorted(
        path for path in model_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _CHECKPOINT_SUFFIXES
    )
    if not candidates:
        raise FileNotFoundError(
            f"No supported checkpoint file found under {model_dir}; "
            f"expected one of {', '.join(sorted(_CHECKPOINT_SUFFIXES))}. "
            "Pass --checkpoint to select a specific file."
        )
    if len(candidates) > 1:
        choices = ", ".join(str(path) for path in candidates)
        raise ValueError(
            f"Multiple checkpoint files found for {model} under {model_dir}: "
            f"{choices}. Pass --checkpoint to select one."
        )
    return candidates[0].resolve()


def _create_adapter(args: argparse.Namespace, device: str,
                    overrides: dict[str, Any]) -> Any:
    module_name, class_name = _ADAPTERS[args.model]
    adapter_cls = getattr(importlib.import_module(module_name, package=__package__), class_name)
    accepted = inspect.signature(adapter_cls).parameters
    unknown = sorted(set(overrides) - set(accepted))
    if unknown:
        raise ValueError(f"unsupported override(s) for {args.model}: {', '.join(unknown)}")
    if args.model == "sam" and getattr(args, "checkpoint_root", None) is not None:
        overrides = {**overrides, "checkpoint_root": args.checkpoint_root}
    adapter = adapter_cls(**overrides)
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
    checkpoint_root = getattr(args, "checkpoint_root", None)
    if args.checkpoint is not None:
        args.checkpoint = _resolve_checkpoint(args.model, args.checkpoint, checkpoint_root)
    adapter.load(args.checkpoint, device=device)
    return adapter


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


def run(args: argparse.Namespace) -> Path | list[Path]:
    if not args.model:
        raise ValueError("--model is required (unless --list-models is used)")
    if args.input_list is not None and args.input is not None:
        raise ValueError("--input and --input-list cannot be used together")
    if args.input_list is not None:
        input_list = args.input_list.expanduser().resolve()
        if not input_list.is_file():
            raise FileNotFoundError(f"input list does not exist: {input_list}")
        input_paths = [
            Path(line.strip()).expanduser()
            for line in input_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not input_paths:
            raise ValueError(f"input list contains no tile paths: {input_list}")
        missing = [str(path) for path in input_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"input tile does not exist: {missing[0]}")
        if args.output_base is None:
            raise ValueError("--output-base is required with --input-list")
        if args.output is not None:
            raise ValueError("--output cannot be used with --input-list")
    elif args.input is not None:
        input_paths = None
    else:
        raise ValueError("--input or --input-list is required (unless --list-models is used)")
    if (args.x is None) != (args.y is None):
        raise ValueError("--x and --y must be supplied together")
    if args.target_mpp is not None and (not np.isfinite(args.target_mpp) or args.target_mpp <= 0):
        raise ValueError("--target-mpp must be a finite positive value")
    if args.input_list is not None:
        overrides = _parse_overrides(args.param)
        device = "cpu" if args.dry_run else _device(args.device)
        adapter = None if args.dry_run else _create_adapter(args, device, overrides)
        output_base = args.output_base.expanduser().resolve()
        runs = []
        tile_names: dict[str, Path] = {}
        for input_path in input_paths:
            input_path = input_path.resolve()
            if input_path.stem in tile_names:
                raise ValueError(
                    f"duplicate tile name {input_path.stem!r} in input list: "
                    f"{tile_names[input_path.stem]} and {input_path}"
                )
            tile_names[input_path.stem] = input_path
            tile_args = argparse.Namespace(**vars(args))
            tile_args.input = input_path
            coordinates = _named_tile_coordinates(input_path)
            if coordinates is None:
                tile_args.image_id = args.image_id or input_path.stem
                tile_args.x = tile_args.y = None
            else:
                image_id, tile_args.x, tile_args.y = coordinates
                tile_args.image_id = args.image_id or image_id
            tile_args.tile_id = input_path.stem
            tile_args.output = output_base / input_path.stem
            runs.append(_run_file(tile_args, batch_output=True, adapter=adapter))
        return runs
    args.input = args.input.expanduser().resolve()
    if args.input.is_dir():
        if args.x is not None:
            raise ValueError("--x and --y cannot be used with a PNG tile directory; use filename coordinates")
        if args.run_id is not None:
            raise ValueError("--run-id cannot be used with a PNG tile directory; each tile gets its own run ID")
        if args.tile_id is not None:
            raise ValueError("--tile-id cannot be used with a PNG tile directory; each tile uses its filename")
        if args.level != 0:
            raise ValueError("PNG tile directories only support level=0")
        input_paths = sorted(
            path for path in args.input.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        )
        if not input_paths:
            raise ValueError(f"PNG tile directory contains no PNG files: {args.input}")
        overrides = _parse_overrides(args.param)
        device = "cpu" if args.dry_run else _device(args.device)
        adapter = None if args.dry_run else _create_adapter(args, device, overrides)
        runs = []
        for input_path in input_paths:
            tile_args = argparse.Namespace(**vars(args))
            tile_args.input = input_path
            tile_args.output = (
                (args.output.expanduser().resolve()
                 if args.output is not None
                 else Path("segment_result") / args.model / "result")
                / input_path.stem
            )
            coordinates = _named_tile_coordinates(input_path)
            if coordinates is None:
                tile_args.image_id = args.image_id or input_path.stem
                tile_args.x = tile_args.y = None
            else:
                image_id, tile_args.x, tile_args.y = coordinates
                tile_args.image_id = args.image_id or image_id
            tile_args.tile_id = input_path.stem
            runs.append(_run_file(tile_args, batch_output=True, adapter=adapter))
        return runs
    return _run_file(args)


def _run_file(args: argparse.Namespace, *, batch_output: bool = False,
              adapter: Any = None) -> Path:
    if args.input.suffix.lower() == ".ndpi" and args.x is None:
        raise ValueError("NDPI input requires explicit level-0 --x and --y coordinates")
    overrides = _parse_overrides(args.param)
    request = TileRequest(input_path=args.input, image_id=args.image_id or args.input.stem,
        level=args.level, x=args.x, y=args.y, border_ratio=0 if args.border == "none" else 0.30,
        mpp=args.mpp,
        padding_policy=args.padding, tile_id=args.tile_id)
    extraction_request = request
    if batch_output:
        extraction_request = TileRequest(
            input_path=request.input_path, image_id=request.image_id, level=request.level,
            output_size=request.output_size, border_ratio=request.border_ratio,
            mpp=request.mpp, padding_policy=request.padding_policy, tile_id=request.tile_id,
        )
    tile = extract_tile(extraction_request)
    source_mpp = tile.metadata.mpp
    reference_mpp = args.target_mpp
    if batch_output and request.x is not None and request.y is not None:
        border = tile.metadata.border_pixels
        tile.metadata.x, tile.metadata.y = request.x, request.y
        tile.metadata.requested_box = (
            request.x - border, request.y - border,
            request.x + request.output_size + border,
            request.y + request.output_size + border,
        )
        tile.metadata.readable_box = (
            request.x, request.y,
            request.x + request.output_size,
            request.y + request.output_size,
        )
    device = "cpu" if args.dry_run else _device(args.device)
    source_digest = _sha256(args.input)
    run_id = args.run_id or _run_id(args.model, source_digest, request, overrides, args.seed)
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("--run-id must be a single safe directory name")
    if batch_output:
        if args.output is not None:
            destination = args.output.expanduser()
        else:
            parent = (args.output or Path("segment_result") / args.model / "result").expanduser()
            destination = parent / run_id
    elif args.output is not None and args.run_id is None:
        destination = args.output.expanduser()
    else:
        parent = (args.output or Path("segment_result") / args.model / "result").expanduser()
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
            source_mpp=source_mpp, reference_mpp=reference_mpp,
            metadata={"dry_run": True})
    else:
        if adapter is None:
            adapter = _create_adapter(args, device, overrides)
        if args.target_mpp is None:
            reference_mpp = getattr(adapter, "reference_mpp", None)
        model_tile, model_request, scale, scale_warning = prepare_model_input(
            tile.image, request, source_mpp=source_mpp, reference_mpp=reference_mpp,
        )
        inference_started = time.perf_counter()
        raw = adapter.predict(model_tile, model_request)
        result_obj = adapter.postprocess(raw, model_request)
        result_obj = restore_model_result(
            result_obj, request, source_mpp=source_mpp,
            reference_mpp=reference_mpp, scale=scale,
            input_shape=tile.metadata.input_shape,
        )
        result_obj.elapsed_seconds = time.perf_counter() - inference_started
        if scale_warning:
            result_obj.warnings.append(scale_warning)
        if result_obj.metadata.get("whole_cell_validated") is False:
            result_obj.warnings.append(
                "whole_cell_unvalidated: model output has not been validated "
                "against reviewed marrow whole-cell annotations"
            )
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
    instance_metadata_path = None
    if result_obj.instance_classes or result_obj.class_probabilities:
        class_ids = set(result_obj.instance_classes or {}) | set(result_obj.class_probabilities or {})
        instance_metadata_path = destination / "instances.json"
        instance_metadata_path.write_text(
            json.dumps(_jsonable({
                "schema_version": 1,
                "taxonomy": None,
                "model": args.model,
                "instances": [
                    {
                        "instance_id": int(instance_id),
                        "predicted_class": (result_obj.instance_classes or {}).get(instance_id),
                        "class_probabilities": (result_obj.class_probabilities or {}).get(instance_id),
                    }
                    for instance_id in sorted(class_ids)
                ],
            }), indent=2) + "\n",
            encoding="utf-8",
        )

    result_obj.tile_id = request.tile_id or f"{request.image_id}_L{request.level}_x{tile.metadata.x}_y{tile.metadata.y}"
    checkpoint_path = args.checkpoint
    if checkpoint_path is None and result_obj.checkpoint_id:
        possible_path = Path(result_obj.checkpoint_id).expanduser()
        if possible_path.is_file():
            checkpoint_path = possible_path.resolve()
    metadata = {
        "schema_version": 1, "run_id": run_id, "model": args.model,
        "adapter_version": result_obj.adapter_version, "dry_run": bool(args.dry_run),
        "source": {"path": str(args.input), "sha256": source_digest,
                   "format": args.input.suffix.lower()},
        "tile": {**tile.metadata.to_dict(), "tile_id": result_obj.tile_id},
        "request": {"level": request.level, "x": tile.metadata.x, "y": tile.metadata.y,
                    "output_size": request.output_size, "border_ratio": request.border_ratio,
                    "border_semantics": "per_side", "padding_policy": request.padding_policy},
        "inference": {"device": device,
                      "checkpoint": str(checkpoint_path) if checkpoint_path else result_obj.checkpoint_id,
                      "checkpoint_sha256": _sha256(checkpoint_path) if checkpoint_path and checkpoint_path.is_file() else None,
                      "seed": args.seed, "model_overrides": overrides,
                      "source_mpp": result_obj.source_mpp,
                      "reference_mpp": result_obj.reference_mpp,
                      "scale_factor": result_obj.scale_factor,
                      "elapsed_seconds": result_obj.elapsed_seconds,
                      "wall_seconds": time.perf_counter() - started,
                      "input_shape": result_obj.input_shape, "output_shape": result_obj.output_shape,
                      "resize_scale": result_obj.resize_scale, "warnings": result_obj.warnings,
                      "adapter_metadata": result_obj.metadata},
        "validation": {
            "whole_cell_target": "marrow whole-cell instances",
            "status": (
                "not_validated"
                if result_obj.metadata.get("whole_cell_validated") is False
                else "not_assessed"
            ),
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "pillow": _version("Pillow"),
                        "torch": _version("torch"),
                        "model_packages": {
                            "patho-sam": _version("patho-sam"),
                            "cellvit": _version("cellvit"),
                            "cellSAM": _version("cellSAM"),
                            "cellpose": _version("cellpose"),
                            "segment-anything": _version("segment-anything"),
                        }},
        "outputs": {"input.png": _sha256(destination / "input.png"),
                    "instance_mask.tif": _sha256(destination / "instance_mask.tif"),
                    "binary_mask.png": _sha256(destination / "binary_mask.png")},
    }
    if instance_metadata_path is not None:
        metadata["outputs"]["instances.json"] = _sha256(instance_metadata_path)
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
        paths = run(args)
    except (ValueError, RuntimeError, FileNotFoundError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if isinstance(paths, list):
        print("\n".join(str(path) for path in paths))
    else:
        print(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
