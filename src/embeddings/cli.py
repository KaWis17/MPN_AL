"""Command-line entry points for base models: list/check them and extract tile embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Sequence

import torch

from ..image_backend.cli_args import add_source_arguments, source_from_arguments
from .base_model import SSLBaseModel, build_base_model
from .provenance import run_provenance, write_json
from .registry import POOLINGS, list_specs, load_registry_file, resolve_spec
from .tables import ParquetStreamWriter, shard_path
from .tile_encoder import TILE_COLUMNS, WINDOW_COLUMNS, TileEncoder
from .windows import WINDOW_MODES, windows_table

logger = logging.getLogger("mpn_al.embeddings")


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def add_model_arguments(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    """Options choosing the base model and where it runs."""
    group = parser.add_argument_group("base model")
    group.add_argument(
        "--base-model",
        required=required,
        help="registry name (see list_base_models.py), or 'timm:<name>' / 'hf:<org/repo>'",
    )
    group.add_argument("--checkpoint", type=Path, help="local encoder weights to load instead of the pretrained ones")
    group.add_argument("--registry-file", type=Path, help="YAML file with additional base-model entries")
    group.add_argument("--device", default=default_device(), help="cuda, cuda:0, cpu (default: cuda if available)")
    group.add_argument("--no-amp", action="store_true", help="disable float16 autocast on CUDA")
    group.add_argument(
        "--random-init",
        action="store_true",
        help="do not load pretrained weights: a random encoder, only for testing that a pipeline runs",
    )


def model_from_arguments(args: argparse.Namespace) -> SSLBaseModel:
    if args.registry_file is not None:
        load_registry_file(args.registry_file)
    if args.random_init:
        logger.warning("--random-init: the encoder has RANDOM weights; outputs are meaningless beyond their shapes")
    model = build_base_model(
        args.base_model, pretrained=not args.random_init, checkpoint_path=args.checkpoint, device=args.device
    )
    facts = model.describe()
    logger.info(
        "base model %s: %s, embed_dim=%d, patch=%d, %d blocks, %.0fM parameters, on %s",
        facts["name"], facts["source"], facts["embed_dim"], facts["patch_size"], facts["num_layers"],
        facts["parameters"] / 1e6, args.device,
    )
    return model


# ---- list / check ---------------------------------------------------------------------------


def main_models(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List the registered SSL base models, or load one and verify it end to end."
    )
    parser.add_argument("--registry-file", type=Path, help="YAML file with additional base-model entries")
    parser.add_argument("--format", choices=("table", "markdown", "json"), default="table")
    parser.add_argument(
        "--check",
        metavar="MODEL",
        help="download/load this model, print the facts of the loaded network and run test forwards",
    )
    parser.add_argument("--checkpoint", type=Path, help="with --check: local encoder weights")
    parser.add_argument("--random-init", action="store_true", help="with --check: build the architecture without weights")
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.registry_file is not None:
        load_registry_file(args.registry_file)

    if args.check:
        return _check_model(args)

    specs = list_specs()
    if args.format == "json":
        print(json.dumps([_spec_row(spec) | {"notes": spec.notes, "url": spec.url} for spec in specs], indent=2))
        return 0
    rows = [_spec_row(spec) for spec in specs]
    headers = list(rows[0])
    if args.format == "markdown":
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join("---" for _ in headers) + "|")
        for row in rows:
            print("| " + " | ".join(str(row[h]) for h in headers) + " |")
        return 0
    widths = [max(len(h), *(len(str(row[h])) for row in rows)) for h in headers]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(row[h]).ljust(w) for h, w in zip(headers, widths)))
    print("\nAd-hoc references also work: --base-model timm:<timm name or hf-hub:org/repo>, hf:<org/repo>")
    return 0


def _spec_row(spec) -> dict[str, object]:
    return {
        "name": spec.name,
        "architecture": spec.architecture or "-",
        "dim": spec.embed_dim or "-",
        "ssl": spec.ssl_method or "-",
        "input": f"{spec.native_size}px@{spec.native_mpp}um",
        "pooling": spec.tile_pooling,
        "backend": spec.backend,
        "gated": "yes" if spec.gated else "no",
        "license": spec.license or "-",
    }


def _check_model(args: argparse.Namespace) -> int:
    spec = resolve_spec(args.check)
    print(f"loading {spec.name} from {spec.source} ...")
    model = build_base_model(
        spec, pretrained=not args.random_init, checkpoint_path=args.checkpoint, device=args.device
    )
    print(json.dumps(model.describe(), indent=2))
    native = model.fit_size(spec.native_size)
    for side in (native, model.fit_size(native * 2)):
        images = torch.rand(1, 3, side, side, device=args.device)
        started = time.perf_counter()
        with torch.no_grad():
            output = model(images, layers=model.default_layers())
        elapsed = time.perf_counter() - started
        print(
            f"input {side}x{side}: patch tokens {tuple(output.patch_tokens.shape)}, "
            f"cls {None if output.cls_token is None else tuple(output.cls_token.shape)}, "
            f"hidden layers {sorted(output.hidden_states)}, "
            f"tile embedding ({spec.tile_pooling}) {tuple(output.pooled(spec.tile_pooling).shape)}, {elapsed:.2f}s"
        )
    print("ok")
    return 0


# ---- extract --------------------------------------------------------------------------------


def main_extract(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Embed tiles with an SSL base model and write embeddings.parquet (one row per tile)."
    )
    add_model_arguments(parser)
    add_source_arguments(parser)
    windows = parser.add_argument_group("windows (how a tile is fed to the model)")
    windows.add_argument("--window-mode", choices=WINDOW_MODES, default="grid")
    windows.add_argument("--mpp-model", type=float, help="resolution the model reads at (default: the model's native mpp)")
    windows.add_argument("--overlap", type=float, default=0.0, help="grid mode: minimum fractional overlap of windows")
    windows.add_argument("--pooling", choices=POOLINGS, help="default: the pooling recommended by the model card")
    output = parser.add_argument_group("output")
    output.add_argument("--out", type=Path, required=True, help="tile embeddings, e.g. .../rounds/<round_id>/embeddings.parquet")
    output.add_argument("--windows-out", type=Path, help="also write one embedding per window to this Parquet file")
    output.add_argument("--windows-table", type=Path, help="also write the window geometry (model_windows.parquet rows)")
    output.add_argument(
        "--skip-existing",
        action="store_true",
        help="exit at once if this task's output file exists, so that resubmitting a job array redoes only the missing shards",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="windows per forward pass")
    parser.add_argument("--num-workers", type=int, default=0, help="processes reading and cropping tiles")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    source, tiles = source_from_arguments(args)
    # With --num-shards every task writes <name>.shard-III-of-NNN.<ext>; merge them with scripts/merge_shards.py.
    out = shard_path(args.out, args.shard_index, args.num_shards)
    windows_out = shard_path(args.windows_out, args.shard_index, args.num_shards) if args.windows_out else None
    windows_path = shard_path(args.windows_table, args.shard_index, args.num_shards) if args.windows_table else None
    if not tiles:
        logger.info("shard %d of %d has no tiles: nothing to do", args.shard_index, args.num_shards)
        return 0
    if args.skip_existing and out.exists():
        logger.info("%s already exists: nothing to do", out)
        return 0
    model = model_from_arguments(args)
    encoder = TileEncoder(
        model,
        window_mode=args.window_mode,
        mpp_model=args.mpp_model,
        overlap=args.overlap,
        default_mpp=args.mpp,
        pooling=args.pooling,
        batch_size=args.batch_size,
        amp=not args.no_amp,
        num_workers=args.num_workers,
    )
    logger.info("%d tiles, window mode %s, pooling %s", len(tiles), args.window_mode, encoder.pooling)

    from tqdm import tqdm

    all_windows = []
    window_writer = ParquetStreamWriter(windows_out, WINDOW_COLUMNS) if windows_out else None
    started = time.perf_counter()
    with ParquetStreamWriter(out, TILE_COLUMNS) as tile_writer:
        for result in tqdm(encoder.encode(source, tiles), total=len(tiles), unit="tile"):
            tile_writer.add(result.tile_row(model.name, encoder.pooling, args.window_mode))
            if window_writer is not None:
                for row in result.window_rows(model.name, encoder.pooling):
                    window_writer.add(row)
            if windows_path is not None:
                all_windows.extend(result.windows)
    if window_writer is not None:
        window_writer.close()
    if windows_path is not None:
        windows_path.parent.mkdir(parents=True, exist_ok=True)
        windows_table(all_windows, prep_id=tiles[0].prep_id).to_parquet(windows_path, index=False)

    elapsed = time.perf_counter() - started
    write_json(
        out.with_suffix(".run.json"),
        run_provenance(
            vars(args),
            base_model=model.describe(),
            windows={"mode": args.window_mode, "mpp_model": encoder.planner.mpp_model, "pooling": encoder.pooling},
            tiles=len(tiles),
            seconds=round(elapsed, 1),
        ),
    )
    logger.info("wrote %d tile embeddings to %s in %.1fs", len(tiles), out, elapsed)
    return 0
