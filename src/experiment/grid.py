"""Experiment grid: one run = one base model x working resolution x share of the training tiles x seed.

A grid file (``configs/experiments/*.yaml``) lists the values of every axis; ``expand_grid`` returns the
runs in a fixed order, so a Slurm array task index always means the same run.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RunConfig:
    model: str
    seg_mpp: float | None  # working resolution of the head in um/px; None = the tiles' own resolution
    fraction: float  # share of the training tiles used (nested: a smaller share is a subset of a larger one)
    seed: int
    base_width: int = 32
    epochs: int = 20
    steps_per_epoch: int = 50
    batch_size: int = 8
    crop_size: int = 256
    lr: float = 3e-4

    @property
    def run_id(self) -> str:
        slug = re.sub(r"[^A-Za-z0-9.-]+", "-", self.model).strip("-")
        resolution = "native" if self.seg_mpp is None else f"{self.seg_mpp:g}um"
        return f"{slug}__{resolution}__f{round(self.fraction * 100):03d}__s{self.seed}"


def expand_grid(document: dict) -> list[RunConfig]:
    """All runs of a grid document, models slowest-varying (so that a model is loaded once per block)."""
    for key in ("models", "fractions", "seeds"):
        if not document.get(key):
            raise ValueError(f"grid needs a non-empty {key!r} list")
    train = dict(document.get("train") or {})
    allowed = {"base_width", "epochs", "steps_per_epoch", "batch_size", "crop_size", "lr"}
    if set(train) - allowed:
        raise ValueError(f"unknown training keys: {sorted(set(train) - allowed)}")
    for fraction in document["fractions"]:
        if not 0 < fraction <= 1:
            raise ValueError(f"fractions must be in (0, 1], got {fraction}")
    runs = [
        RunConfig(model=m, seg_mpp=r, fraction=float(f), seed=int(s), **train)
        for m, r, f, s in itertools.product(
            document["models"], document.get("seg_mpp") or [None], document["fractions"], document["seeds"]
        )
    ]
    ids = [run.run_id for run in runs]
    if len(set(ids)) != len(ids):
        raise ValueError("the grid contains duplicate runs")
    return runs


def load_grid(path: str | Path) -> tuple[str, list[RunConfig]]:
    with open(path, encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    return str(document.get("name", Path(path).stem)), expand_grid(document)
