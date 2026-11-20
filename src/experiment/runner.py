"""Run one cell of the experiment grid: train a head on a subset of the training tiles, evaluate by patient.

Rules that keep the comparison honest:

* the split comes from a fixed file (``tile_id, patient_id, split``) built from the patient split;
* training uses the ``train`` tiles only and, for the final epoch's score, ``validation`` tiles; the
  ``test`` tiles are evaluated only with ``--final``, once the configuration is frozen;
* subsets are nested per seed, so a larger share contains the smaller one;
* every tile's counts are stored, so intervals over patients can be recomputed later.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from ..embeddings.base_model import SSLBaseModel, build_base_model
from ..embeddings.provenance import run_provenance, write_json
from ..evaluation import (
    aggregate,
    bootstrap_ci,
    correction_effort,
    instance_class_ids,
    instance_counts,
    mean_pq,
    per_class_counts,
    sum_counts,
)
from ..segmenter import CellDecoder, CellSegmenter, LabelSchema
from ..segmenter.model import predict_working_maps
from ..segmenter.postprocess import PostprocessConfig, hv_to_instances
from ..training import TrainConfig, load_training_tiles, train_head
from ..training.dataset import TrainingTile
from .grid import RunConfig, load_grid

logger = logging.getLogger("mpn_al.experiment")


def read_split(path: str | Path) -> pd.DataFrame:
    table = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)  # tiles.parquet works too
    missing = {"tile_id", "patient_id", "split"} - set(table.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    unknown = set(table["split"]) - {"train", "validation", "test"}
    if unknown:
        raise ValueError(f"unknown split names {sorted(unknown)} (use train / validation / test)")
    leaks = table.groupby("patient_id")["split"].nunique()
    if (leaks > 1).any():
        raise ValueError(f"patients in more than one split: {sorted(leaks[leaks > 1].index)}")
    return table


def nested_subset(tile_ids: Sequence[str], fraction: float, seed: int) -> list[str]:
    """The first ``round(fraction * n)`` tiles of a seeded shuffle of the sorted ids."""
    ordered = sorted(tile_ids)
    order = np.random.default_rng(seed).permutation(len(ordered))
    keep = max(1, round(fraction * len(ordered)))
    return [ordered[i] for i in order[:keep]]


def predicted_classes(instances: np.ndarray, type_prob: np.ndarray | None) -> np.ndarray:
    """Class of every predicted instance (index = id): the class with most probability mass under it."""
    count = int(instances.max())
    classes = np.zeros(count + 1, dtype=np.int64)
    if count == 0:
        return classes
    if type_prob is None or type_prob.shape[0] < 2:
        classes[1:] = 1
        return classes
    mass = np.stack(
        [np.bincount(instances.ravel(), weights=channel.ravel(), minlength=count + 1) for channel in type_prob[1:]]
    )
    classes[1:] = mass.argmax(axis=0)[1:] + 1
    return classes


def evaluate_tiles(
    segmenter: CellSegmenter,
    tiles: Sequence[TrainingTile],
    *,
    work_mpp: float | None,
    amp: bool = True,
    max_size: int = 1024,
) -> list[dict]:
    """Segment whole tiles and return, per tile, the counts behind PQ (overall and per class) and the correction effort."""
    num_classes = segmenter.labels.num_classes
    rows = []
    for tile in tiles:
        maps = predict_working_maps(segmenter, tile.image, amp=amp, max_size=max_size)
        predicted = hv_to_instances(maps.prob, maps.hv, mpp=work_mpp, config=PostprocessConfig())
        true_classes = instance_class_ids(tile.instances, tile.types)
        pred_classes = predicted_classes(predicted, maps.type_prob)
        rows.append(
            {
                "tile_id": tile.tile_id,
                "counts": instance_counts(tile.instances, predicted, ignore=tile.ignore),
                "classes": {
                    str(c): v
                    for c, v in per_class_counts(
                        tile.instances, true_classes, predicted, pred_classes, num_classes, ignore=tile.ignore
                    ).items()
                },
                "effort": correction_effort(tile.instances, predicted, ignore=tile.ignore),
            }
        )
    return rows


def summarize(rows: list[dict], patients: dict[str, str], *, resamples: int = 1000) -> dict:
    """Overall metrics, per-class PQ, mPQ, interval over patients and total correction effort for one split."""
    if not rows:
        return {}
    per_patient: dict[str, list[dict]] = {}
    for row in rows:
        per_patient.setdefault(patients[row["tile_id"]], []).append(row["counts"])
    classes = sorted({c for row in rows for c in row["classes"]})
    per_class = {int(c): sum_counts(row["classes"][c] for row in rows) for c in classes}
    return {
        "tiles": len(rows),
        "patients": len(per_patient),
        **aggregate([row["counts"] for row in rows]),
        "pq_ci": bootstrap_ci(per_patient, "pq", resamples=resamples),
        **mean_pq(per_class),
        "effort": {k: sum(row["effort"][k] for row in rows) for k in rows[0]["effort"]},
    }


def run_one(
    config: RunConfig,
    *,
    base: SSLBaseModel,
    schema: LabelSchema,
    split: pd.DataFrame,
    load,
    out_dir: Path,
    eval_splits: Sequence[str],
    device: torch.device,
    resamples: int = 1000,
) -> dict:
    """Train and evaluate one run. ``load(seg_mpp)`` returns all labelled tiles at that working resolution."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_split = dict(zip(split["tile_id"], split["split"]))
    patients = dict(zip(split["tile_id"], split["patient_id"]))
    tiles = [t for t in load(config.seg_mpp) if t.tile_id in by_split]
    train_ids = nested_subset([t.tile_id for t in tiles if by_split[t.tile_id] == "train"], config.fraction, config.seed)
    train = [t for t in tiles if t.tile_id in set(train_ids)]
    validation = [t for t in tiles if by_split[t.tile_id] == "validation"]
    if not train:
        raise ValueError("no training tiles in the split file")

    torch.manual_seed(config.seed)
    decoder = CellDecoder(
        base.embed_dim, num_types=schema.num_classes if schema.num_classes > 1 else 0, base_width=config.base_width
    )
    segmenter = CellSegmenter(base, decoder, schema, seg_mpp=config.seg_mpp).to(device)
    train_config = TrainConfig(
        epochs=config.epochs, steps_per_epoch=config.steps_per_epoch, batch_size=config.batch_size,
        crop_size=config.crop_size, lr=config.lr, seed=config.seed, validate_every=config.epochs,
        amp=device.type == "cuda",
    )
    work_mpp = config.seg_mpp
    started = time.perf_counter()
    history = train_head(segmenter, train, validation, train_config, out_dir / "head", work_mpp=work_mpp)
    train_seconds = time.perf_counter() - started

    summaries, counts_path = {}, out_dir / "tile_counts.jsonl"
    with open(counts_path, "w", encoding="utf-8") as handle:
        for name in eval_splits:
            selected = [t for t in tiles if by_split[t.tile_id] == name]
            started = time.perf_counter()
            rows = evaluate_tiles(segmenter, selected, work_mpp=work_mpp, amp=device.type == "cuda")
            seconds = time.perf_counter() - started
            for row in rows:
                handle.write(json.dumps({"split": name, "patient_id": patients[row["tile_id"]], **row}) + "\n")
            summaries[name] = summarize(rows, patients, resamples=resamples)
            summaries[name]["seconds_per_tile"] = seconds / max(len(rows), 1)
    result = {
        "run_id": config.run_id,
        "config": asdict(config),
        "train_tiles": train_ids,
        "train_seconds": round(train_seconds, 1),
        "final_loss": history[-1]["loss"],
        "metrics": summaries,
        "base_model": base.describe(),
    }
    write_json(out_dir / "result.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one cell of an experiment grid (a Slurm array task index selects it).")
    parser.add_argument("--grid", type=Path, required=True, help="grid YAML, e.g. configs/experiments/pilot.yaml")
    parser.add_argument("--task-index", type=int, help="run number (default: SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--list", action="store_true", help="print the runs of the grid and exit")
    parser.add_argument("--images", type=Path, help="directory of tile images named <tile_id>.<ext>")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--masks", type=Path)
    parser.add_argument("--labels", type=Path, help="label schema YAML (default: one class)")
    parser.add_argument("--mpp", type=float, help="resolution of the images in um/px")
    parser.add_argument("--split-file", type=Path, help="CSV with tile_id, patient_id, split")
    parser.add_argument("--out", type=Path, help="experiment directory; each run writes <out>/<run_id>/")
    parser.add_argument("--final", action="store_true", help="also evaluate the test split (only once the configuration is frozen)")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--random-init", action="store_true", help="random encoder weights (tests only)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resamples", type=int, default=1000, help="bootstrap resamples over patients")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    name, runs = load_grid(args.grid)
    if args.list:
        for index, run in enumerate(runs):
            print(f"{index:4d}  {run.run_id}")
        print(f"{len(runs)} runs; submit an array with --array=0-{len(runs) - 1}")
        return 0
    for required in ("images", "split_file", "out"):
        if getattr(args, required) is None:
            parser.error(f"--{required.replace('_', '-')} is required")
    if (args.masks is None) == (args.annotations is None):
        parser.error("give exactly one of --masks and --annotations")
    index = args.task_index
    if index is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if index is None or not 0 <= index < len(runs):
        parser.error(f"--task-index must be in [0, {len(runs)}) (or run as a Slurm array task)")
    config = runs[index]
    out_dir = args.out / name / config.run_id
    if args.skip_existing and (out_dir / "result.json").exists():
        logger.info("%s already done", config.run_id)
        return 0

    schema = LabelSchema.from_yaml(args.labels) if args.labels else LabelSchema.single_class("cell")
    split = read_split(args.split_file)
    eval_splits = ["validation", "test"] if args.final else ["validation"]
    cache: dict = {}

    def load(seg_mpp):
        if seg_mpp not in cache:
            cache.clear()
            cache[seg_mpp] = load_training_tiles(
                args.images, schema=schema, masks=args.masks, annotations=args.annotations, mpp=args.mpp,
                seg_mpp=seg_mpp, tile_ids=list(split["tile_id"]),
            )
        return cache[seg_mpp]

    device = torch.device(args.device)
    base = build_base_model(config.model, pretrained=not args.random_init, device=device)
    logger.info("run %d of %d: %s (evaluating %s)", index, len(runs), config.run_id, eval_splits)
    result = run_one(config, base=base, schema=schema, split=split, load=load, out_dir=out_dir,
                     eval_splits=eval_splits, device=device, resamples=args.resamples)
    write_json(out_dir / "provenance.json", run_provenance(vars(args), grid=name))
    for split_name, metrics in result["metrics"].items():
        logger.info("%s: PQ %.3f (95%% CI %.3f-%.3f over %d patients), mPQ %.3f",
                    split_name, metrics["pq"], metrics["pq_ci"]["low"], metrics["pq_ci"]["high"],
                    metrics["patients"], metrics["mpq"])
    return 0
