"""Collect the results of a grid into tables: per-cell rows, learning curves, paired comparisons.

``python scripts/summarize_experiment.py <experiment dir> --split validation`` writes ``runs.csv``,
``curves.csv``, ``comparison.csv`` and ``report.md`` next to the results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

KEYS = ["model", "seg_mpp", "fraction"]


def load_runs(root: str | Path, split: str = "validation") -> pd.DataFrame:
    """One row per finished run that evaluated ``split``."""
    rows = []
    for path in sorted(Path(root).rglob("result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result["metrics"].get(split)
        if not metrics:
            continue
        config = result["config"]
        rows.append(
            {
                "run_id": result["run_id"], "model": config["model"], "seg_mpp": config["seg_mpp"],
                "fraction": config["fraction"], "seed": config["seed"], "train_tiles": len(result["train_tiles"]),
                "pq": metrics["pq"], "dq": metrics["dq"], "sq": metrics["sq"], "mpq": metrics["mpq"],
                "pq_low": metrics["pq_ci"]["low"], "pq_high": metrics["pq_ci"]["high"], "patients": metrics["patients"],
                "pixels_to_fix_per_tile": metrics["effort"]["pixels_to_fix"] / metrics["tiles"],
                "objects_to_fix_per_tile": sum(metrics["effort"][k] for k in ("objects_to_add", "objects_to_delete", "objects_to_adjust")) / metrics["tiles"],
                "seconds_per_tile": metrics["seconds_per_tile"], "train_seconds": result["train_seconds"],
            }
        )
    if not rows:
        raise FileNotFoundError(f"no result.json with split {split!r} under {root}")
    table = pd.DataFrame(rows)
    table["seg_mpp"] = table["seg_mpp"].astype(float).fillna(-1.0)  # native resolution; keeps groupby from dropping it
    return table


def curves(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds for every model x resolution x training-set share."""
    grouped = runs.groupby(KEYS)
    out = grouped.agg(
        seeds=("seed", "nunique"), train_tiles=("train_tiles", "mean"), pq=("pq", "mean"), pq_sd=("pq", "std"),
        mpq=("mpq", "mean"), dq=("dq", "mean"), sq=("sq", "mean"),
        pixels_to_fix_per_tile=("pixels_to_fix_per_tile", "mean"), seconds_per_tile=("seconds_per_tile", "mean"),
        train_seconds=("train_seconds", "mean"),
    ).reset_index()
    return out.sort_values(KEYS).reset_index(drop=True)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_tests(runs: pd.DataFrame, reference: str, metric: str = "pq") -> pd.DataFrame:
    """Every other model against ``reference``, paired on (resolution, share, seed), Wilcoxon signed-rank + Holm.

    With few pairs the smallest possible p-value is large (6 pairs: 0.031): a non-significant result then
    means "not enough runs", not "no difference".
    """
    from scipy.stats import wilcoxon

    key = ["seg_mpp", "fraction", "seed"]
    base = runs[runs["model"] == reference].set_index(key)[metric]
    if base.empty:
        raise ValueError(f"reference model {reference!r} has no runs")
    rows = []
    for model in sorted(set(runs["model"]) - {reference}):
        other = runs[runs["model"] == model].set_index(key)[metric]
        joined = pd.concat([other, base], axis=1, keys=["other", "reference"]).dropna()
        diff = joined["other"] - joined["reference"]
        p = float(wilcoxon(diff).pvalue) if len(diff) >= 2 and (diff != 0).any() else float("nan")
        rows.append({"model": model, "reference": reference, "pairs": len(diff), "mean_difference": float(diff.mean()) if len(diff) else float("nan"), "p": p})
    table = pd.DataFrame(rows)
    valid = table["p"].notna()
    table["p_holm"] = np.nan
    if valid.any():
        table.loc[valid, "p_holm"] = holm(table.loc[valid, "p"].tolist())
    return table


def markdown(curve: pd.DataFrame, comparison: pd.DataFrame | None, split: str) -> str:
    shown = curve.assign(seg_mpp=curve["seg_mpp"].map(lambda v: "native" if v < 0 else f"{v:g}"))
    lines = [f"# Experiment summary ({split} split)", "", "Mean over seeds; `pq_sd` is the standard deviation over seeds.", ""]
    columns = list(shown.columns)
    lines += ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    if comparison is not None:
        lines += ["", "## Paired comparison with the reference model", ""]
        lines += [f"| {' | '.join(comparison.columns)} |", "|" + "---|" * len(comparison.columns)]
        for _, row in comparison.iterrows():
            lines.append("| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise the results of an experiment directory.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument("--reference", help="model the others are compared with (paired Wilcoxon, Holm)")
    parser.add_argument("--metric", default="pq")
    args = parser.parse_args(argv)
    runs = load_runs(args.root, args.split)
    curve = curves(runs)
    comparison = paired_tests(runs, args.reference, args.metric) if args.reference else None
    runs.to_csv(args.root / "runs.csv", index=False)
    curve.to_csv(args.root / "curves.csv", index=False)
    if comparison is not None:
        comparison.to_csv(args.root / "comparison.csv", index=False)
    (args.root / "report.md").write_text(markdown(curve, comparison, args.split), encoding="utf-8")
    print(f"{len(runs)} runs, {runs['model'].nunique()} models; wrote runs.csv, curves.csv, report.md to {args.root}")
    return 0
