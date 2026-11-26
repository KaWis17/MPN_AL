"""Which tiles an experiment shows, and the per-tile predictions it keeps so that figures can be drawn later.

Choosing tiles *after* seeing results invites cherry-picking, so the choice depends only on the data: the
split file, the tiles' own cell counts and sizes, and a seed from the grid file. Every run of a grid therefore
shows the same tiles. Three groups, each labelled in ``viz/index.csv``:

``dense``     the tiles with most reference cells per megapixel (the hardest for instance separation);
``atypical``  tiles whose size differs from the most common one in the split (other scale or scanner);
``random``    the rest, drawn at random with the seed.

One tile per patient is taken before any patient appears twice. The test split is only ever considered when it
is evaluated (``--final``), like its metrics.

Each saved tile is ``<run>/viz/<split>/<tile_id>.npz`` with arrays at the tile's working resolution:
``instances`` (predicted), ``prob`` (uint8, probability * 255), ``hv`` (int8, value * 127), ``token_pca``
(uint8 ``[gh, gw, 3]``, see ``evaluation.visual_compare.token_pca_grid``), ``instance_classes`` and the
``reference`` instances and ``ignore`` mask the scores were computed against, so a figure needs neither the
encoder nor the annotations, and recomputing the counts from the file reproduces the run's numbers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import yaml

from ..evaluation.visual_compare import token_pca_grid
from ..segmenter.model import SegmentationMaps
from ..training.dataset import TrainingTile

GROUPS = ("dense", "atypical", "random")


@dataclass(frozen=True)
class VizConfig:
    per_split: int = 0  # tiles saved per evaluated split; 0 = none
    seed: int = 0

    def __post_init__(self) -> None:
        if self.per_split < 0:
            raise ValueError(f"visualize.per_split must be >= 0, got {self.per_split}")


def load_visualize(path: str | Path) -> VizConfig:
    """The ``visualize:`` section of a grid file (absent = nothing is saved)."""
    with open(path, encoding="utf-8") as handle:
        section = (yaml.safe_load(handle) or {}).get("visualize") or {}
    unknown = set(section) - {"per_split", "seed"}
    if unknown:
        raise ValueError(f"unknown visualize keys: {sorted(unknown)}")
    return VizConfig(per_split=int(section.get("per_split", 0)), seed=int(section.get("seed", 0)))


def cell_density(tile: TrainingTile) -> float:
    """Reference cells per megapixel of the tile at working resolution."""
    count = len(np.unique(tile.instances[tile.instances > 0]))
    return count / (tile.image.shape[0] * tile.image.shape[1] / 1e6)


def select_viz_tiles(
    tiles: Sequence[TrainingTile], patients: dict[str, str], per_split: int, seed: int = 0
) -> list[dict]:
    """Tiles of ONE split to save, as dicts ``tile_id, patient_id, group, density, height, width``.

    Depends only on the tiles' reference annotations and sizes, never on a model's output.
    """
    if per_split <= 0 or not tiles:
        return []
    by_id = {t.tile_id: t for t in sorted(tiles, key=lambda t: t.tile_id)}
    count = min(per_split, len(by_id))
    density = {tile_id: cell_density(t) for tile_id, t in by_id.items()}
    shapes = Counter(t.image.shape[:2] for t in by_id.values())
    modal = shapes.most_common(1)[0][0]

    chosen: list[dict] = []
    used_patients: set[str] = set()

    def take(group: str, candidates: list[str], quota: int) -> None:
        taken = {row["tile_id"] for row in chosen}
        picked = 0
        for fresh_patients_only in (True, False):  # first one tile per patient, then fill up
            for tile_id in candidates:
                if picked >= quota:
                    return
                if tile_id in taken or (fresh_patients_only and patients[tile_id] in used_patients):
                    continue
                tile = by_id[tile_id]
                chosen.append(
                    {
                        "tile_id": tile_id,
                        "patient_id": patients[tile_id],
                        "group": group,
                        "density": round(density[tile_id], 3),
                        "height": int(tile.image.shape[0]),
                        "width": int(tile.image.shape[1]),
                    }
                )
                taken.add(tile_id)
                used_patients.add(patients[tile_id])
                picked += 1

    quota = count // 4
    take("dense", sorted(by_id, key=lambda i: (-density[i], i)), quota)
    odd = [i for i, t in by_id.items() if t.image.shape[:2] != modal]
    take("atypical", sorted(odd, key=lambda i: (shapes[by_id[i].image.shape[:2]], i)), quota)
    ids = list(by_id)
    shuffled = [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))]
    take("random", shuffled, count - len(chosen))
    return chosen


def save_prediction(
    path: str | Path,
    *,
    tile: TrainingTile,
    maps: SegmentationMaps,
    instances: np.ndarray,
    instance_classes: np.ndarray,
) -> None:
    """Write one tile's prediction and the reference it was scored against (see the module docstring)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def ids(array: np.ndarray) -> np.ndarray:
        return array.astype(np.uint16 if array.max(initial=0) < 65535 else np.int32)

    np.savez_compressed(
        path,
        instances=ids(instances),
        prob=np.round(np.clip(maps.prob, 0, 1) * 255).astype(np.uint8),
        hv=np.round(np.clip(maps.hv, -1, 1) * 127).astype(np.int8),
        token_pca=token_pca_grid(maps.tokens),
        instance_classes=instance_classes.astype(np.uint8),
        reference=ids(tile.instances),
        reference_types=tile.types.astype(np.uint8),
        ignore=tile.ignore.astype(bool),
    )


def make_capture(
    viz_dir: str | Path, split: str, chosen: Sequence[dict], work_mpp: float | None
) -> tuple[Callable, list[dict]]:
    """A ``capture(tile, maps, instances, instance_classes)`` hook for ``evaluate_tiles`` and the index rows it fills."""
    info = {row["tile_id"]: row for row in chosen}
    rows: list[dict] = []

    def capture(tile: TrainingTile, maps: SegmentationMaps, instances: np.ndarray, instance_classes: np.ndarray) -> None:
        if tile.tile_id not in info:
            return
        save_prediction(
            Path(viz_dir) / split / f"{tile.tile_id}.npz",
            tile=tile, maps=maps, instances=instances, instance_classes=instance_classes,
        )
        rows.append({"split": split, "work_mpp": work_mpp, **info[tile.tile_id]})

    return capture, rows


def write_index(viz_dir: str | Path, rows: Sequence[dict]) -> None:
    if rows:
        Path(viz_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(Path(viz_dir) / "index.csv", index=False)
