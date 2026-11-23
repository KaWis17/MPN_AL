"""Aggregate foundation-model runs and compare instance masks to a reference.

Reference label ``0`` is background and ``-1`` is ignored. Instance matching is
one-to-one maximum-IoU matching at threshold 0.5. PQ follows the standard
sum-IoU/(TP + .5 FP + .5 FN) definition. Accuracy is omitted (NA) without a
whole-cell reference; an optional megakaryocyte reference enables target-cell
scoring independently. Descriptive statistics remain available without either.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

MODEL_ORDER = ("pathosam", "cellvit", "cellsam", "cellpose_sam", "sam")
MATCH_IOU = 0.5


def _maximum_assignment(weights: np.ndarray) -> list[tuple[int, int]]:
    """Pure-Python Hungarian assignment for a rectangular weight matrix."""
    n, m = weights.shape
    if not n or not m:
        return []
    size = max(n, m)
    padded = np.zeros((size, size), dtype=float)
    padded[:n, :m] = weights
    # Hungarian algorithm for minimum cost, using negated weights.
    cost = -padded
    u = [0.0] * (size + 1); v = [0.0] * (size + 1)
    p = [0] * (size + 1); way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i; j0 = 0
        minv = [float("inf")] * (size + 1); used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]; delta = float("inf"); j1 = 0
            for j in range(1, size + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]: minv[j] = cur; way[j] = j0
                    if minv[j] < delta: delta = minv[j]; j1 = j
            for j in range(size + 1):
                if used[j]: u[p[j]] += delta; v[j] -= delta
                else: minv[j] -= delta
            j0 = j1
            if p[j0] == 0: break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0: break
    return [(p[j] - 1, j - 1) for j in range(1, size + 1) if p[j] and p[j] <= n and j <= m]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_labels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        labels = np.asarray(image)
    if labels.ndim == 3:
        if labels.shape[2] == 1:
            labels = labels[..., 0]
        else:
            raise ValueError(f"expected a 2D label mask at {path}, got {labels.shape}")
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"expected an integer 2D label mask at {path}")
    if np.any(labels < -1):
        raise ValueError(f"labels below -1 are invalid in {path}")
    return labels.astype(np.int64, copy=False)


def _components(binary: np.ndarray) -> list[int]:
    """8-connected component sizes, without scipy/opencv dependencies."""
    seen = np.zeros(binary.shape, dtype=bool)
    sizes: list[int] = []
    h, w = binary.shape
    for y, x in zip(*np.nonzero(binary & ~seen)):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        size = 0
        while stack:
            cy, cx = stack.pop()
            size += 1
            for ny in range(max(0, cy - 1), min(h, cy + 2)):
                for nx in range(max(0, cx - 1), min(w, cx + 2)):
                    if binary[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        sizes.append(size)
    return sizes


def _instance_counts(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    p = pred[valid]
    r = ref[valid]
    p_ids, p_counts = np.unique(p[p > 0], return_counts=True)
    r_ids, r_counts = np.unique(r[r > 0], return_counts=True)
    npred, nref = len(p_ids), len(r_ids)
    if npred and nref:
        # Contingency table gives each pair's intersection; calculate IoU only
        # for overlapping pairs, then solve exact maximum-weight assignment.
        pairs = np.stack((p, r), axis=1)
        both = (pairs[:, 0] > 0) & (pairs[:, 1] > 0)
        overlap = {}
        if np.any(both):
            unique, counts = np.unique(pairs[both], axis=0, return_counts=True)
            overlap = {(int(a), int(b)): int(n) for (a, b), n in zip(unique, counts)}
        p_area = dict(zip(map(int, p_ids), map(int, p_counts)))
        r_area = dict(zip(map(int, r_ids), map(int, r_counts)))
        matrix = np.zeros((npred, nref), dtype=float)
        p_index = {int(value): i for i, value in enumerate(p_ids)}
        r_index = {int(value): i for i, value in enumerate(r_ids)}
        for (pi, ri), inter in overlap.items():
            matrix[p_index[pi], r_index[ri]] = inter / (p_area[pi] + r_area[ri] - inter)
        assignment = _maximum_assignment(matrix)
        matched = [float(matrix[i, j]) for i, j in assignment if matrix[i, j] >= MATCH_IOU]
    else:
        matched = []
    tp = len(matched)
    fp, fn = npred - tp, nref - tp
    precision = tp / npred if npred else (1.0 if nref == 0 else 0.0)
    recall = tp / nref if nref else (1.0 if npred == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    sq = float(sum(matched) / tp) if tp else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if tp + fp + fn else 1.0
    return {"instance_tp": tp, "instance_fp": fp, "instance_fn": fn,
            "instance_precision": precision, "instance_recall": recall,
            "instance_f1": f1, "pq": sq * rq, "sq": sq, "rq": rq}


def _megakaryocyte_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float | int | str]:
    valid = ref != -1
    predicted_ids, predicted_areas = np.unique(pred[valid & (pred > 0)], return_counts=True)
    target_ids, target_areas = np.unique(ref[valid & (ref > 0)], return_counts=True)
    if not len(target_ids):
        return {
            "megakaryocyte_gt_count": 0,
            "megakaryocyte_detected_count": 0,
            "megakaryocyte_detection_recall": "NA",
            "megakaryocyte_mean_iou": "NA",
        }

    overlaps = np.zeros((len(predicted_ids), len(target_ids)), dtype=np.int64)
    if len(predicted_ids):
        pred_index = {int(value): i for i, value in enumerate(predicted_ids)}
        target_index = {int(value): i for i, value in enumerate(target_ids)}
        pairs = np.stack((pred[valid], ref[valid]), axis=1)
        both = (pairs[:, 0] > 0) & (pairs[:, 1] > 0)
        if np.any(both):
            unique, counts = np.unique(pairs[both], axis=0, return_counts=True)
            for (pred_id, target_id), count in zip(unique, counts):
                overlaps[pred_index[int(pred_id)], target_index[int(target_id)]] = count

    pred_areas = dict(zip(map(int, predicted_ids), map(int, predicted_areas)))
    target_areas_by_id = dict(zip(map(int, target_ids), map(int, target_areas)))
    ious = np.zeros_like(overlaps, dtype=float)
    for i, pred_id in enumerate(predicted_ids):
        for j, target_id in enumerate(target_ids):
            intersection = overlaps[i, j]
            if intersection:
                union = pred_areas[int(pred_id)] + target_areas_by_id[int(target_id)] - intersection
                ious[i, j] = intersection / union

    assignment = _maximum_assignment(ious)
    assigned_ious = {
        target_index: float(ious[pred_index, target_index])
        for pred_index, target_index in assignment
    }
    detected = sum(iou >= MATCH_IOU for iou in assigned_ious.values())
    return {
        "megakaryocyte_gt_count": len(target_ids),
        "megakaryocyte_detected_count": detected,
        "megakaryocyte_detection_recall": detected / len(target_ids),
        "megakaryocyte_mean_iou": sum(assigned_ious.values()) / len(target_ids),
    }


def evaluate(pred: np.ndarray, ref: np.ndarray | None,
             megakaryocyte_ref: np.ndarray | None = None) -> dict[str, Any]:
    if pred.ndim != 2 or np.any(pred < 0):
        raise ValueError("prediction must be a 2D non-negative instance-label mask")
    if megakaryocyte_ref is not None:
        if megakaryocyte_ref.ndim != 2 or np.any(megakaryocyte_ref < -1):
            raise ValueError("megakaryocyte reference must be a 2D integer label mask with labels >= -1")
        if megakaryocyte_ref.shape != pred.shape:
            raise ValueError(
                f"megakaryocyte reference shape {megakaryocyte_ref.shape} does not match prediction {pred.shape}"
            )
    foreground = pred > 0
    sizes = _components(foreground)
    result: dict[str, Any] = {
        "pred_cell_count": int(len(np.unique(pred[foreground]))),
        "foreground_fraction": float(foreground.mean()),
        "component_count_8conn": len(sizes),
        "component_area_mean_px": float(np.mean(sizes)) if sizes else 0.0,
        "component_area_median_px": float(np.median(sizes)) if sizes else 0.0,
    }
    if ref is None:
        result.update({name: "NA" for name in (
            "binary_dice", "binary_iou", "binary_precision", "binary_recall", "binary_f1",
            "instance_tp", "instance_fp", "instance_fn", "instance_precision",
            "instance_recall", "instance_f1", "pq", "sq", "rq")})
    else:
        if ref.shape != pred.shape:
            raise ValueError(f"reference shape {ref.shape} does not match prediction {pred.shape}")
        valid = ref != -1
        pb, rb = (pred > 0) & valid, (ref > 0) & valid
        tp, fp, fn = int(np.count_nonzero(pb & rb)), int(np.count_nonzero(pb & ~rb)), int(np.count_nonzero(~pb & rb))
        denom = 2 * tp + fp + fn
        precision = tp / (tp + fp) if tp + fp else (1.0 if tp + fn == 0 else 0.0)
        recall = tp / (tp + fn) if tp + fn else (1.0 if tp + fp == 0 else 0.0)
        result.update({"binary_dice": 2 * tp / denom if denom else 1.0,
                       "binary_iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
                       "binary_precision": precision, "binary_recall": recall,
                       "binary_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0})
        result.update(_instance_counts(pred, ref, valid))
    if megakaryocyte_ref is not None:
        result.update(_megakaryocyte_metrics(pred, megakaryocyte_ref))
    else:
        result.update({name: "NA" for name in (
            "megakaryocyte_gt_count", "megakaryocyte_detected_count",
            "megakaryocyte_detection_recall", "megakaryocyte_mean_iou")})
    return result


def discover_runs(root: Path) -> list[dict[str, Any]]:
    manifests = sorted(root.rglob("run.json")) if root.is_dir() else ([root / "run.json"] if root.is_dir() else [])
    runs = []
    for manifest in manifests:
        try:
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            model = metadata["model"]
            run_id = metadata["run_id"]
            tile_id = metadata["tile"]["tile_id"]
            outputs = metadata["outputs"]
            mask_path = manifest.parent / "instance_mask.tif"
            input_path = manifest.parent / "input.png"
            if model not in MODEL_ORDER or not isinstance(run_id, str) or not tile_id:
                raise ValueError("invalid model, run_id, or tile_id")
            if not mask_path.is_file() or not input_path.is_file():
                raise ValueError("run is incomplete: input.png and instance_mask.tif are required")
            if outputs.get("instance_mask.tif") != sha256(mask_path):
                raise ValueError("instance_mask.tif checksum does not match run.json")
            if outputs.get("input.png") != sha256(input_path):
                raise ValueError("input.png checksum does not match run.json")
            labels = _load_labels(mask_path)
            tile = metadata["tile"]
            expected_size = int(tile["output_size"])
            if labels.shape != (expected_size, expected_size):
                raise ValueError(f"prediction shape {labels.shape} does not match tile output_size={expected_size}")
            inference = metadata.get("inference", {})
            runs.append({"manifest": manifest, "metadata": metadata, "labels": labels,
                         "input_path": input_path, "mask_path": mask_path,
                         "tile_id": tile_id, "model": model, "run_id": run_id,
                         "border_ratio": metadata.get("request", {}).get("border_ratio"),
                         "checkpoint": inference.get("checkpoint_sha256") or inference.get("checkpoint"),
                         "runtime_seconds": inference.get("elapsed_seconds"),
                         "memory_peak_bytes": inference.get("peak_memory_bytes")})
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid completed run {manifest.parent}: {exc}") from exc
    if not runs:
        raise ValueError(f"no run.json manifests found under {root}")
    return runs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_parquet(out: Path, rows: list[dict[str, Any]]) -> bool:
    """Write the same run-level table when pandas and a parquet engine exist."""
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(out / "metrics.parquet", index=False)
        return True
    except (ImportError, ValueError, OSError):
        return False


def _plots(out: Path, rows: list[dict[str, Any]], reference_available: bool,
           megakaryocyte_reference_available: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if reference_available:
        models = [model for model in MODEL_ORDER if any(row["model"] == model for row in rows)]
        metrics = ("binary_dice", "instance_f1", "pq")
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
        for ax, metric in zip(axes, metrics):
            vals = [float(np.mean([float(r[metric]) for r in rows if r["model"] == m])) for m in models]
            ax.bar(models, vals, color=["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677"][:len(models)])
            ax.set_title(metric.replace("_", " ")); ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=35)
        fig.tight_layout(); fig.savefig(out / "metric_by_model.png", dpi=150); plt.close(fig)
    if megakaryocyte_reference_available and any(
        isinstance(row["megakaryocyte_mean_iou"], (int, float)) for row in rows
    ):
        models = [model for model in MODEL_ORDER if any(row["model"] == model for row in rows)]
        metrics = ("megakaryocyte_mean_iou", "megakaryocyte_detection_recall")
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
        for ax, metric in zip(axes, metrics):
            vals = [float(np.mean([
                float(row[metric]) for row in rows
                if row["model"] == model and isinstance(row[metric], (int, float))
            ])) for model in models]
            ax.bar(models, vals, color=["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677"][:len(models)])
            ax.set_title(metric.replace("_", " ")); ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=35)
        fig.tight_layout(); fig.savefig(out / "megakaryocyte_metrics.png", dpi=150); plt.close(fig)
    # Paired border deltas for each tile/model where both settings exist.
    grouped: dict[tuple[str, str], dict[float, list[dict[str, Any]]]] = {}
    for row in rows:
        if row.get("binary_dice") == "NA":
            continue
        grouped.setdefault((row["tile_id"], row["model"]), {}).setdefault(float(row["border_ratio"] or 0), []).append(row)
    delta_rows = []
    for (tile_id, model), borders in grouped.items():
        if 0.0 in borders and 0.0 < max(borders):
            b = max(borders)
            delta_rows.append((f"{model}\n{tile_id[:12]}", np.mean([float(r["binary_dice"]) for r in borders[b]]) - np.mean([float(r["binary_dice"]) for r in borders[0.0]])))
    if delta_rows:
        fig, ax = plt.subplots(figsize=(max(6, len(delta_rows) * 1.2), 4))
        ax.bar([x[0] for x in delta_rows], [x[1] for x in delta_rows], color="#4477AA")
        ax.axhline(0, color="black", linewidth=.8); ax.set_ylabel("Dice delta (border − none)")
        ax.tick_params(axis="x", rotation=35); fig.tight_layout(); fig.savefig(out / "border_deltas.png", dpi=150); plt.close(fig)
    if rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        for model in MODEL_ORDER:
            vals = [r for r in rows if r["model"] == model and isinstance(r["runtime_seconds"], (int, float))]
            if vals:
                ax.scatter([r["runtime_seconds"] for r in vals], [r["pred_cell_count"] for r in vals], label=model)
        ax.set_xlabel("Inference time (s)"); ax.set_ylabel("Predicted cell count"); ax.legend()
        fig.tight_layout(); fig.savefig(out / "runtime_cell_count.png", dpi=150); plt.close(fig)


def _qualitative(out: Path, runs: list[dict[str, Any]]) -> None:
    for tile_id in sorted({r["tile_id"] for r in runs}):
        selected = sorted((r for r in runs if r["tile_id"] == tile_id), key=lambda r: (MODEL_ORDER.index(r["model"]), float(r["border_ratio"] or 0), r["run_id"]))
        thumbs = []
        for run in selected:
            image = Image.open(run["input_path"]).convert("RGB")
            labels = run["labels"]
            # Display the canonical central region if saved input has context.
            size = int(run["metadata"]["tile"]["output_size"])
            border = int(run["metadata"]["tile"].get("border_pixels", 0))
            if image.size != (size, size):
                image = image.crop((border, border, border + size, border + size))
            rgb = np.asarray(image.resize((384, 384))).copy()
            lab = np.asarray(Image.fromarray(labels.astype(np.int32), mode="I").resize((384, 384), Image.Resampling.NEAREST))
            edges = np.zeros(lab.shape, bool)
            edges[1:, :] |= lab[1:, :] != lab[:-1, :]; edges[:, 1:] |= lab[:, 1:] != lab[:, :-1]
            rgb[edges & (lab > 0)] = (255, 40, 40)
            thumb = Image.fromarray(rgb)
            draw = ImageDraw.Draw(thumb); draw.rectangle((0, 0, 383, 28), fill="white")
            draw.text((5, 5), f"{run['model']} border={run['border_ratio']}", fill="black")
            thumbs.append(thumb)
        columns = 3; rows = math.ceil(len(thumbs) / columns)
        panel = Image.new("RGB", (columns * 384, rows * 384), "white")
        for index, thumb in enumerate(thumbs):
            panel.paste(thumb, ((index % columns) * 384, (index // columns) * 384))
        safe = hashlib.sha256(tile_id.encode()).hexdigest()[:12]
        panel.save(out / f"qualitative_{safe}.png")


def compare(runs_root: Path, output: Path, reference: Path | None = None, *,
            megakaryocyte_reference: Path | None = None, tile_id: str | None = None,
            force: bool = False) -> Path:
    runs = discover_runs(runs_root.expanduser().resolve())
    if tile_id is not None:
        runs = [run for run in runs if run["tile_id"] == tile_id]
        if not runs:
            raise ValueError(f"no completed runs found for tile_id {tile_id!r}")
    ref = _load_labels(reference.expanduser().resolve()) if reference else None
    megakaryocyte_ref = (
        _load_labels(megakaryocyte_reference.expanduser().resolve())
        if megakaryocyte_reference else None
    )
    if ref is not None or megakaryocyte_ref is not None:
        tile_ids = {r["tile_id"] for r in runs}
        if len(tile_ids) > 1:
            raise ValueError("reference masks can only be used when all runs belong to one tile")
        shapes = {tuple(r["labels"].shape) for r in runs}
        for name, reference_labels in (
            ("reference", ref), ("megakaryocyte reference", megakaryocyte_ref)
        ):
            if reference_labels is not None and shapes != {tuple(reference_labels.shape)}:
                raise ValueError(
                    f"{name} shape {reference_labels.shape} does not match all run shapes {sorted(shapes)}"
                )
    destination = output.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=force)
    rows = []
    for run in sorted(runs, key=lambda r: (r["tile_id"], MODEL_ORDER.index(r["model"]), float(r["border_ratio"] or 0), r["run_id"])):
        metrics = evaluate(run["labels"], ref, megakaryocyte_ref)
        rows.append({"tile_id": run["tile_id"], "model": run["model"], "border_ratio": run["border_ratio"],
                     "checkpoint": run["checkpoint"], "run_id": run["run_id"],
                     "runtime_seconds": run["runtime_seconds"] if run["runtime_seconds"] is not None else "NA",
                     "memory_peak_bytes": run["memory_peak_bytes"] if run["memory_peak_bytes"] is not None else "NA",
                     **metrics})
    _write_csv(destination / "metrics.csv", rows)
    parquet_written = _write_parquet(destination, rows)
    # Descriptive table is independently useful and exists with/without GT.
    summary_cols = ("tile_id", "model", "border_ratio", "run_id", "runtime_seconds", "memory_peak_bytes",
                    "pred_cell_count", "foreground_fraction", "component_count_8conn",
                    "component_area_mean_px", "component_area_median_px",
                    "megakaryocyte_gt_count", "megakaryocyte_detected_count",
                    "megakaryocyte_detection_recall", "megakaryocyte_mean_iou")
    _write_csv(destination / "summary.csv", [{key: row[key] for key in summary_cols} for row in rows])
    _plots(destination, rows, ref is not None, megakaryocyte_ref is not None)
    _qualitative(destination, runs)
    manifest = {"schema_version": 1,
        "reference": ({"path": str(reference.resolve()), "sha256": sha256(reference.resolve())} if reference else None),
        "megakaryocyte_reference": (
            {"path": str(megakaryocyte_reference.resolve()), "sha256": sha256(megakaryocyte_reference.resolve())}
            if megakaryocyte_reference else None
        ),
        "reference_mask_policy": "integer labels; 0=background; -1=ignored pixels" if reference else None,
        "megakaryocyte_reference_mask_policy": (
            "integer labels; positive IDs are reviewed megakaryocytes, 0=other/background, -1=ignored pixels"
            if megakaryocyte_reference else None
        ),
        "instance_matching": {"criterion": "one-to-one maximum-IoU assignment", "iou_threshold": MATCH_IOU,
                              "panoptic_quality": "sum matched IoU / (TP + 0.5 FP + 0.5 FN)"},
        "binary_metrics": "foreground intersection over valid (reference != -1) pixels; empty-empty scores 1",
        "connected_components": "8-connected components of the predicted binary foreground; touching separate labels may merge",
        "border_truncated_instances": "compared as provided; no automatic ignore or truncation repair",
        "run_ids": [r["run_id"] for r in runs],
        "runs": [{"run_id": r["run_id"], "model": r["model"], "tile_id": r["tile_id"],
                  "run_json": str(r["manifest"].resolve()), "run_json_sha256": sha256(r["manifest"]),
                  "prediction_sha256": sha256(r["mask_path"]), "input_sha256": sha256(r["input_path"])} for r in runs],
        "accuracy_metrics_available": ref is not None,
        "megakaryocyte_metrics_available": megakaryocyte_ref is not None,
        "parquet": "written" if parquet_written else "not written; install pandas and pyarrow or fastparquet to enable Parquet export"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path, help="one run directory or a root containing run.json files")
    parser.add_argument("--tile-id", help="compare only this tile from a root containing multiple tiles")
    parser.add_argument("--reference", type=Path, help="reviewed integer instance mask TIFF; -1 marks ignored pixels")
    parser.add_argument("--megakaryocyte-reference", type=Path,
                        help="reviewed integer mask containing only megakaryocyte instances; 0=other, -1=ignored")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "results",
        help="comparison output directory (default: src/model_registry/results)",
    )
    parser.add_argument("--force", action="store_true", help="allow writing into an existing output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        path = compare(
            args.runs, args.output, args.reference,
            megakaryocyte_reference=args.megakaryocyte_reference,
            tile_id=args.tile_id, force=args.force,
        )
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
