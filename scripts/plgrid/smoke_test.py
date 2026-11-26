#!/usr/bin/env python
"""Smoke test of the whole base-model chain on the machine it runs on (built for a PLGrid GPU node).

    python scripts/plgrid/smoke_test.py --out "$SCRATCH/mpn_al_runs/smoke" --models lunit-dino-s16 phikon-v2

It needs no data: a synthetic data set is generated first. For every model it then checks

  load            the weights load (offline, from the cache) and match the registry
  forward         encoder passes at several input sizes, float32 and float16 autocast (speed, memory,
                  and whether float16 changes the embedding)
  tile embeddings TileEncoder on the demo tiles (windows per second)
  train head      a few training steps of the segmentation head, with float16 autocast on a GPU
  segment         segmentation of whole tiles; mask, cell table and GeoJSON must agree

and writes smoke_report.json and smoke_report.md into --out. A failing stage does not stop the others;
the exit code is 1 if any stage failed. On the cluster this runs inside `smoke_gpu.sbatch`
(`bash scripts/plgrid/remote.sh smoke`); here it also runs on a CPU, which is how the tests use it.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import json
import logging
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402, F401  (makes `mpn_al` importable from a plain checkout)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mpn_al.embeddings import TileEncoder, build_base_model  # noqa: E402
from mpn_al.image_backend import ImageFolderSource  # noqa: E402
from mpn_al.segmenter import (  # noqa: E402
    CellDecoder,
    CellSegmenter,
    LabelSchema,
    PipelineConfig,
    RunWriter,
    SegmentationPipeline,
)
from mpn_al.training import TrainConfig, load_training_tiles, train_head  # noqa: E402
from mpn_al.training.synthetic import write_demo_dataset  # noqa: E402

logger = logging.getLogger("mpn_al.smoke")


class Report:
    """Stages with their status, timing and details."""

    def __init__(self) -> None:
        self.stages: list[dict] = []
        self.environment: dict = {}

    @contextlib.contextmanager
    def stage(self, name: str):
        """``with report.stage("x") as details:`` - an exception marks the stage failed; ``details["warnings"]`` marks it warn."""
        entry: dict = {"stage": name, "status": "ok", "seconds": 0.0, "details": {}}
        started = time.perf_counter()
        logger.info("== %s", name)
        try:
            yield entry["details"]
        except Exception as exc:  # noqa: BLE001 - one broken stage must not hide the others
            entry["status"] = "fail"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            logger.error("stage %r failed:\n%s", name, traceback.format_exc())
        else:
            if entry["details"].get("warnings"):
                entry["status"] = "warn"
        entry["seconds"] = round(time.perf_counter() - started, 2)
        self.stages.append(entry)
        logger.info("-- %s: %s (%.1fs)", name, entry["status"], entry["seconds"])

    @property
    def failed(self) -> list[str]:
        return [stage["stage"] for stage in self.stages if stage["status"] == "fail"]


def environment() -> dict:
    """Versions, hardware and scheduler context of this run."""
    info: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "node": os.environ.get("SLURMD_NODENAME") or platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cpus": int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "hf_home": os.environ.get("HF_HOME"),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
    }
    for package in ("timm", "transformers", "huggingface_hub", "numpy", "scipy", "pandas", "pyarrow", "cv2"):
        try:
            info[package] = importlib.import_module(package).__version__
        except Exception:  # noqa: BLE001 - an optional package may be absent
            info[package] = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        info.update(
            gpu=properties.name,
            gpu_memory_gb=round(properties.total_memory / 2**30, 1),
            gpu_capability=f"{properties.major}.{properties.minor}",
            cudnn=torch.backends.cudnn.version(),
        )
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            result = subprocess.run(
                [smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30, check=False,
            )
            info["nvidia_smi"] = (result.stdout or result.stderr).strip()
        except (OSError, subprocess.SubprocessError):
            info["nvidia_smi"] = None
    return info


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_gb(device: torch.device) -> float | None:
    return round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if device.type == "cuda" else None


def milliseconds(function, device: torch.device, repeat: int = 3) -> float:
    function()  # warm-up: lazy initialisation, cuDNN and allocator behaviour
    synchronize(device)
    started = time.perf_counter()
    for _ in range(repeat):
        function()
    synchronize(device)
    return round((time.perf_counter() - started) / repeat * 1000, 1)


def forward_rows(model, device: torch.device, sizes: list[int]) -> tuple[list[dict], list[str]]:
    """Time the encoder at several input sizes; on a GPU also compare float16 autocast with float32."""
    rows, warnings = [], []
    for requested in sizes:
        side = model.fit_size(requested)
        images = torch.rand(1, 3, side, side, device=device)
        row: dict = {"input": side, "tokens": (side // model.patch_size) ** 2}
        reference = model.embed(images)
        if not torch.isfinite(reference).all():
            raise RuntimeError(f"non-finite embedding at {side}x{side}")
        reset_peak(device)
        row["fp32_ms"] = milliseconds(lambda: model.embed(images), device)
        row["fp32_peak_gb"] = peak_gb(device)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                half = model.embed(images)
                reset_peak(device)
                row["fp16_ms"] = milliseconds(lambda: model.embed(images), device)
            row["fp16_peak_gb"] = peak_gb(device)
            cosine = float(F.cosine_similarity(reference.float(), half.float(), dim=-1).min())
            row["fp16_cosine"] = round(cosine, 5)
            if not np.isfinite(cosine) or cosine < 0.98:
                warnings.append(f"float16 autocast changes the embedding at {side}x{side} (cosine {cosine:.4f})")
        rows.append(row)
    return rows, warnings


def check_model(report: Report, args: argparse.Namespace, device: torch.device, name: str, demo: Path) -> None:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    model = None
    with report.stage(f"{name} / load") as details:
        started = time.perf_counter()
        model = build_base_model(name, pretrained=not args.random_init, device=device)
        details.update(model.describe())
        details["load_seconds"] = round(time.perf_counter() - started, 1)
    if model is None:
        return

    with report.stage(f"{name} / forward") as details:
        rows, warnings = forward_rows(model, device, args.forward_sizes)
        details["forward"] = rows
        if warnings:
            details["warnings"] = warnings

    source = ImageFolderSource(demo / "images", mpp=args.mpp)
    with report.stage(f"{name} / tile embeddings") as details:
        encoder = TileEncoder(model, window_mode="grid", default_mpp=args.mpp, batch_size=args.embed_batch, amp=True)
        reset_peak(device)
        synchronize(device)
        started = time.perf_counter()
        results = list(encoder.encode(source))
        synchronize(device)
        seconds = time.perf_counter() - started
        dimension = model.output_dim()
        bad = [r.tile.tile_id for r in results if r.embedding.shape != (dimension,) or not np.isfinite(r.embedding).all()]
        if bad:
            raise RuntimeError(f"embeddings of the wrong shape or not finite for {bad}")
        windows = sum(len(r.windows) for r in results)
        details.update(
            tiles=len(results), windows=windows, seconds=round(seconds, 2),
            windows_per_second=round(windows / seconds, 1), embedding_dim=dimension, peak_gb=peak_gb(device),
        )

    schema = LabelSchema.single_class("cell")
    decoder = CellDecoder(model.embed_dim, base_width=args.base_width)
    segmenter = CellSegmenter(model, decoder, schema).to(device)

    with report.stage(f"{name} / train head") as details:
        if args.steps <= 0:
            details["skipped"] = "--steps 0"
        else:
            tiles = load_training_tiles(demo / "images", schema=schema, masks=demo / "masks", mpp=args.mpp)
            config = TrainConfig(
                epochs=args.epochs, steps_per_epoch=args.steps, batch_size=args.train_batch, crop_size=args.crop_size,
                lr=1e-3, warmup_steps=5, amp=True, validate_every=args.epochs, max_size=args.max_size,
            )
            reset_peak(device)
            synchronize(device)
            started = time.perf_counter()
            history = train_head(segmenter, tiles[:-1], tiles[-1:], config, args.out / slug / "head", work_mpp=args.mpp)
            synchronize(device)
            seconds = time.perf_counter() - started
            losses = [record["loss"] for record in history]
            if not all(np.isfinite(losses)):
                raise RuntimeError(f"non-finite training loss: {losses}")
            details.update(
                steps=args.epochs * args.steps, seconds=round(seconds, 1), loss_first=round(losses[0], 3),
                loss_last=round(losses[-1], 3), val_pq=round(history[-1].get("val_pq", float("nan")), 3),
                trainable_parameters=sum(p.numel() for p in decoder.parameters()), peak_gb=peak_gb(device),
            )
            if losses[-1] >= losses[0]:
                details["warnings"] = ["the training loss did not decrease"]

    with report.stage(f"{name} / segment") as details:
        config = PipelineConfig(default_mpp=args.mpp, amp=True, max_size=args.max_size)
        pipeline = SegmentationPipeline(segmenter=segmenter, config=config)
        writer = RunWriter(args.out / slug / "predictions", labels=schema, base_model=name)
        tiles = source.tiles[: max(1, args.segment_tiles)]
        pipeline.run_tile(source.read(tiles[0]), tiles[0])  # warm-up, not timed
        rows = []
        for tile in tiles:
            image = source.read(tile)
            reset_peak(device)
            synchronize(device)
            started = time.perf_counter()
            result = pipeline.run_tile(image, tile)
            synchronize(device)
            seconds = time.perf_counter() - started
            writer.write(result, seconds)
            in_mask = int((np.unique(result.instances) > 0).sum())
            in_geojson = len(json.loads(writer.geojson_path(tile.tile_id).read_text())["features"])
            in_table = len(pd.read_parquet(writer.cells_path(tile.tile_id)))
            if not (in_mask == len(result.cells) == in_geojson == in_table):
                raise RuntimeError(
                    f"{tile.tile_id}: {in_mask} instances in the mask, {len(result.cells)} cells, "
                    f"{in_geojson} polygons, {in_table} table rows"
                )
            rows.append({"tile": tile.tile_id, "cells": len(result.cells), "seconds": round(seconds, 2), "peak_gb": peak_gb(device)})
        details["tiles"] = rows
        details["median_seconds_per_tile"] = round(statistics.median(row["seconds"] for row in rows), 2)
        if args.mpp < 0.5:  # the same head at half the resolution: timing only, the head was not trained for it
            half = SegmentationPipeline(segmenter=CellSegmenter(model, decoder, schema, seg_mpp=0.5).to(device), config=config)
            synchronize(device)
            started = time.perf_counter()
            half.run_tile(source.read(tiles[0]), tiles[0])
            synchronize(device)
            details["seconds_per_tile_at_0.5_um"] = round(time.perf_counter() - started, 2)

    del segmenter, decoder, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> Report:
    report = Report()
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    with report.stage("environment") as details:
        report.environment = environment()
        details["device"] = str(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"--device {args.device} was requested but PyTorch sees no CUDA device (torch built for CUDA "
                f"{torch.version.cuda}; nvidia-smi: {report.environment.get('nvidia_smi')}). On a login node there is no GPU; "
                "on a GPU node an old driver cannot run a newer CUDA build of PyTorch: install a build for an older CUDA "
                "(MPN_AL_TORCH_INDEX in scripts/plgrid/remote_setup.sh)"
            )
    if report.failed:
        return report
    demo = args.out / "demo"
    with report.stage("data") as details:
        tile_ids = write_demo_dataset(demo, tiles=args.tiles, size=args.tile_size, count=args.cells, mpp=args.mpp)
        details.update(tiles=len(tile_ids), tile_size=args.tile_size, cells_per_tile=args.cells)
    if report.failed:
        return report
    for name in args.models:
        check_model(report, args, device, name, demo)
    return report


def render_details(details: dict) -> list[str]:
    lines: list[str] = []
    scalars = {key: value for key, value in details.items() if not isinstance(value, (list, dict))}
    if scalars:
        lines.append("; ".join(f"{key}={value}" for key, value in scalars.items()))
    for key, value in details.items():
        if key == "warnings":
            lines += [f"- WARNING: {warning}" for warning in value]
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            headers = list(value[0])
            lines += ["", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
            lines += ["| " + " | ".join(str(row.get(h, "")) for h in headers) + " |" for row in value]
    return lines


def markdown(payload: dict) -> str:
    lines = [f"# Smoke test, {payload['created_utc']}", "", f"Result: **{'PASS' if payload['ok'] else 'FAIL'}**", ""]
    lines += ["## Environment", ""] + [f"- {key}: {value}" for key, value in payload["environment"].items() if value is not None]
    lines += ["", "## Stages", "", "| stage | status | seconds |", "|---|---|---|"]
    lines += [f"| {s['stage']} | {s['status']} | {s['seconds']} |" for s in payload["stages"]]
    for stage in payload["stages"]:
        body = render_details(stage["details"])
        if body or stage.get("error"):
            lines += ["", f"### {stage['stage']}", ""]
            if stage.get("error"):
                lines.append(f"**{stage['error']}**")
            lines += body
    return "\n".join(lines) + "\n"


def write_report(report: Report, args: argparse.Namespace) -> dict:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ok": not report.failed,
        "arguments": vars(args),
        "environment": report.environment,
        "stages": report.stages,
    }
    (args.out / "smoke_report.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    (args.out / "smoke_report.md").write_text(markdown(payload), encoding="utf-8")
    return payload


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="directory for the demo data, outputs and the report")
    parser.add_argument("--models", nargs="+", default=["lunit-dino-s16", "phikon-v2"], help="registry names (weights must be in the cache)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--random-init", action="store_true", help="random weights instead of pretrained ones (tests only)")
    data = parser.add_argument_group("synthetic data")
    data.add_argument("--tiles", type=int, default=4)
    data.add_argument("--tile-size", type=int, default=1024)
    data.add_argument("--cells", type=int, default=100, help="cells per tile")
    data.add_argument("--mpp", type=float, default=0.25, help="resolution assigned to the tiles (um/px)")
    work = parser.add_argument_group("workload")
    work.add_argument("--forward-sizes", type=int, nargs="+", default=[224, 448, 1024])
    work.add_argument("--embed-batch", type=int, default=32, help="windows per forward pass when embedding tiles")
    work.add_argument("--epochs", type=int, default=3)
    work.add_argument("--steps", type=int, default=20, help="training steps per epoch; 0 skips training")
    work.add_argument("--train-batch", type=int, default=8)
    work.add_argument("--crop-size", type=int, default=256)
    work.add_argument("--base-width", type=int, default=32)
    work.add_argument("--max-size", type=int, default=1024, help="largest side the segmenter sees in one pass")
    work.add_argument("--segment-tiles", type=int, default=2)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    report = run(args)
    payload = write_report(report, args)
    print((args.out / "smoke_report.md").read_text(encoding="utf-8"))
    if report.failed:
        logger.error("FAILED stages: %s", ", ".join(report.failed))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
