"""Figures of an experiment's results on images, drawn from the predictions its runs saved (``experiment.viz``).

Nothing here needs a GPU, an encoder or the annotations: a run's ``viz/<split>/<tile_id>.npz`` holds the prediction
and the reference it was scored against, and the tile's image is read from the image folder and resampled to the
working resolution of the prediction. Scores in captions are recomputed from the saved arrays with the same matching as
the experiment's metrics, and :func:`check_against_counts` verifies them against ``tile_counts.jsonl``.

Figures per tile:

``gallery``       one row per model, same training share and seed: outlines | errors | probability | token PCA;
``zoom``          the same on the densest window of the tile, enlarged without interpolation;
``curve``         one row per model, one column per training share: how the errors change with more tiles;
``seeds``         one row per model: outlines of every seed overlaid, and where the seeds disagree;
``disagreement``  how many models call each pixel a cell.

Errors use the matching of the PQ metric (IoU > 0.5): true positive (predicted cell with a match), false positive
(predicted cell without), false negative (reference cell nobody found). Colours are the Okabe-Ito colour-blind-safe set.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd

from ..evaluation.instance_metrics import _drop_mostly_ignored, aggregate, instance_counts, pairwise_iou
from ..evaluation.visual_compare import _fit, _label, _stack, densest_box, heatmap, overlay_instances
from ..segmenter.model import resize_image

TP, FP, FN = (0, 158, 115), (213, 94, 0), (0, 114, 178)  # Okabe-Ito: bluish green, vermillion, blue
MODEL_OUTLINE, REFERENCE_OUTLINE = (0, 255, 0), (255, 255, 0)
SEED_COLOURS = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
KINDS = ("gallery", "zoom", "curve", "seeds", "disagreement")


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    model: str
    seg_mpp: float | None
    fraction: float
    seed: int
    path: Path

    @property
    def viz_dir(self) -> Path:
        return self.path / "viz"


# ---- reading runs -----------------------------------------------------------------------------


def discover_runs(root: str | Path) -> list[RunInfo]:
    """Runs under ``root`` (an experiment directory or one level above) that kept predictions."""
    runs = []
    for result in sorted(Path(root).rglob("result.json")):
        if not (result.parent / "viz" / "index.csv").exists():
            continue
        config = json.loads(result.read_text(encoding="utf-8"))["config"]
        runs.append(
            RunInfo(
                run_id=result.parent.name, model=config["model"], seg_mpp=config["seg_mpp"],
                fraction=float(config["fraction"]), seed=int(config["seed"]), path=result.parent,
            )
        )
    return runs


def read_index(run: RunInfo, split: str) -> pd.DataFrame:
    table = pd.read_csv(run.viz_dir / "index.csv")
    return table[table["split"] == split].reset_index(drop=True)


def load_prediction(run: RunInfo, split: str, tile_id: str) -> dict[str, np.ndarray]:
    """A saved prediction as arrays at the tile's working resolution (token PCA resampled to it)."""
    with np.load(run.viz_dir / split / f"{tile_id}.npz") as saved:
        instances = saved["instances"].astype(np.int32)
        height, width = instances.shape
        return {
            "instances": instances,
            "prob": saved["prob"].astype(np.float32) / 255.0,
            "token_rgb": cv2.resize(saved["token_pca"], (width, height), interpolation=cv2.INTER_CUBIC),
            "reference": saved["reference"].astype(np.int32),
            "ignore": saved["ignore"].astype(bool),
        }


def load_image(image_path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """RGB image resampled to the working resolution ``shape = (height, width)`` of a prediction."""
    from PIL import Image

    with Image.open(image_path) as handle:
        return resize_image(np.asarray(handle.convert("RGB")), shape[0], shape[1])


# ---- scores and errors ------------------------------------------------------------------------


def score_prediction(data: dict[str, np.ndarray]) -> dict[str, float]:
    """PQ, DQ, SQ and the TP/FP/FN counts of a saved prediction against its reference."""
    counts = instance_counts(data["reference"], data["instances"], ignore=data["ignore"])
    return {**aggregate([counts]), "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"]}


def error_masks(data: dict[str, np.ndarray], threshold: float = 0.5) -> dict[str, np.ndarray]:
    """Instance maps ``tp`` / ``fp`` (predicted cells) and ``fn`` (reference cells), matched as in the PQ metric."""
    reference, predicted, ignore = data["reference"], data["instances"], data["ignore"]
    predicted = _drop_mostly_ignored(predicted, ignore)
    reference = np.where(ignore, 0, reference)
    predicted = np.where(ignore, 0, predicted)
    reference_ids = np.unique(reference[reference > 0])
    predicted_ids = np.unique(predicted[predicted > 0])
    iou = pairwise_iou(reference, predicted)
    matched = iou > threshold

    def pick(array: np.ndarray, ids: np.ndarray, keep: np.ndarray) -> np.ndarray:
        lookup = np.zeros(int(array.max(initial=0)) + 1, dtype=bool)
        lookup[ids[keep]] = True
        return np.where(lookup[array], array, 0)

    if len(reference_ids) == 0 or len(predicted_ids) == 0:
        hit_predicted = np.zeros(len(predicted_ids), bool)
        hit_reference = np.zeros(len(reference_ids), bool)
    else:
        hit_predicted, hit_reference = matched.any(axis=0), matched.any(axis=1)
    return {
        "tp": pick(predicted, predicted_ids, hit_predicted),
        "fp": pick(predicted, predicted_ids, ~hit_predicted),
        "fn": pick(reference, reference_ids, ~hit_reference),
    }


def error_overlay(image: np.ndarray, data: dict[str, np.ndarray]) -> np.ndarray:
    masks = error_masks(data)
    out = overlay_instances(image, masks["tp"], TP)
    out = overlay_instances(out, masks["fp"], FP)
    return overlay_instances(out, masks["fn"], FN)


def check_against_counts(run: RunInfo, split: str, tile_id: str, scores: dict[str, float]) -> bool:
    """True when the scores recomputed from a saved prediction equal the run's own ``tile_counts.jsonl`` row."""
    for line in (run.path / "tile_counts.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["split"] == split and row["tile_id"] == tile_id:
            return all(float(row["counts"][key]) == float(scores[key]) for key in ("tp", "fp", "fn"))
    return False


def scores_table(runs: Sequence[RunInfo], split: str) -> pd.DataFrame:
    """One row per (run, saved tile): scores, the tile's selection group and whether it matches ``tile_counts.jsonl``."""
    rows = []
    for run in runs:
        for tile in read_index(run, split).itertuples():
            scores = score_prediction(load_prediction(run, split, tile.tile_id))
            rows.append(
                {
                    "run_id": run.run_id, "model": run.model, "seg_mpp": run.seg_mpp, "fraction": run.fraction,
                    "seed": run.seed, "split": split, "tile_id": tile.tile_id, "patient_id": tile.patient_id,
                    "group": tile.group, **scores, "matches_counts": check_against_counts(run, split, tile.tile_id, scores),
                }
            )
    return pd.DataFrame(rows)


# ---- drawing ----------------------------------------------------------------------------------


def _crop(array: np.ndarray, box: tuple[slice, slice] | None) -> np.ndarray:
    return array if box is None else array[box]


def _panel(rgb: np.ndarray, box, size: int, text: str) -> np.ndarray:
    """A labelled panel: crops to ``box`` and enlarges with nearest-neighbour, or scales the whole image to ``size``."""
    rgb = _crop(rgb, box)
    if box is None:
        return _label(_fit(rgb, size), text)
    scale = size / max(rgb.shape[:2])
    shown = cv2.resize(rgb, (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))), interpolation=cv2.INTER_NEAREST)
    return _label(shown, text)


def legend_panel(size: tuple[int, int]) -> np.ndarray:
    """White panel of ``size = (height, width)`` explaining the outline colours."""
    panel = np.full((*size, 3), 255, np.uint8)
    entries = [(REFERENCE_OUTLINE, "reference cell"), (MODEL_OUTLINE, "predicted cell"), (TP, "TP: matched (IoU > 0.5)"),
               (FP, "FP: predicted, no match"), (FN, "FN: reference, missed")]
    for index, (colour, text) in enumerate(entries):
        y = 28 + index * 26
        cv2.rectangle(panel, (10, y - 12), (30, y + 4), colour, -1)
        cv2.putText(panel, text, (38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return panel


def _cell_count(instances: np.ndarray) -> int:
    return int(len(np.unique(instances[instances > 0])))


def _score_text(label: str, scores: dict[str, float]) -> str:
    return f"{label}  PQ {scores['pq']:.2f} DQ {scores['dq']:.2f} SQ {scores['sq']:.2f}"


def zoom_box(reference: np.ndarray, size: int) -> tuple[slice, slice] | None:
    """The densest window of the reference, or ``None`` when the whole image is not larger than ``size``."""
    if size <= 0 or max(reference.shape) <= size:
        return None
    return densest_box(reference, size)


def render_gallery(
    image: np.ndarray, entries: Sequence[tuple[str, dict]], *, box=None, panel: int = 320
) -> np.ndarray:
    """First row: image | reference | legend. Then per model: outlines | errors | probability | token PCA."""
    reference = entries[0][1]["reference"]
    first = [_panel(image, box, panel, "input"), _panel(overlay_instances(image, reference, REFERENCE_OUTLINE), box, panel, "reference")]
    first.append(legend_panel(first[0].shape[:2]))
    rows = [_stack(first, 1)]
    for label, data in entries:
        scores = score_prediction(data)
        rows.append(
            _stack(
                [
                    _panel(overlay_instances(image, data["instances"], MODEL_OUTLINE), box, panel, f"{label}: {_cell_count(data['instances'])} cells"),
                    _panel(error_overlay(image, data), box, panel, _score_text("errors", scores)),
                    _panel(heatmap(data["prob"]), box, panel, "cell probability"),
                    _panel(data["token_rgb"], box, panel, "tokens PCA"),
                ],
                1,
            )
        )
    return _stack(rows, 0)


def render_curve(
    image: np.ndarray, curves: dict[str, Sequence[tuple[float, dict]]], *, box=None, panel: int = 300
) -> np.ndarray:
    """Per model one row, one column per training share (sorted): the error overlay and the score."""
    rows = []
    for model, points in curves.items():
        cells = [
            _panel(error_overlay(image, data), box, panel, _score_text(f"{model} {fraction * 100:g}%", score_prediction(data)))
            for fraction, data in sorted(points, key=lambda p: p[0])
        ]
        rows.append(_stack(cells, 1))
    return _stack(rows, 0)


def render_seeds(
    image: np.ndarray, seeds: dict[str, Sequence[tuple[int, dict]]], *, box=None, panel: int = 360
) -> np.ndarray:
    """Per model: outlines of every seed overlaid (one colour each) | share of seeds calling each pixel a cell."""
    rows = []
    for model, points in seeds.items():
        ordered = sorted(points, key=lambda p: p[0])
        overlay = image
        for colour, (_, data) in zip(SEED_COLOURS, ordered):
            overlay = overlay_instances(overlay, data["instances"], colour)
        pq = ", ".join(f"s{seed} {score_prediction(data)['pq']:.2f}" for seed, data in ordered)
        votes = np.mean([data["instances"] > 0 for _, data in ordered], axis=0).astype(np.float32)
        rows.append(
            _stack(
                [_panel(overlay, box, panel, f"{model}: seeds overlaid, PQ {pq}"),
                 _panel(heatmap(votes), box, panel, "share of seeds calling a cell")],
                1,
            )
        )
    return _stack(rows, 0)


def render_disagreement(
    image: np.ndarray, entries: Sequence[tuple[str, dict]], *, box=None, panel: int = 360
) -> np.ndarray:
    """Image | number of models (of all shown) calling each pixel a cell | pixels where the models split."""
    votes = np.sum([data["instances"] > 0 for _, data in entries], axis=0).astype(np.float32)
    total = len(entries)
    split = ((votes > 0) & (votes < total)).astype(np.float32)
    return _stack(
        [_panel(image, box, panel, "input"),
         _panel(heatmap(votes, 0, total), box, panel, f"models calling a cell (0-{total})"),
         _panel(heatmap(split), box, panel, f"models disagree ({100 * split.mean():.1f}% of pixels)")],
        1,
    )


# ---- html index -------------------------------------------------------------------------------


def write_index_html(
    path: str | Path, sections: Sequence[dict], *, title: str, notes: Sequence[str], summary: pd.DataFrame | None = None
) -> None:
    """A static page: notes, an optional score table and, per tile, its figures. ``sections``: dicts with ``tile_id``,
    ``caption`` and ``figures`` (a list of ``(kind, file name)``)."""
    parts = [
        "<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{html.escape(title)}</title>",
        "<style>:root{color-scheme:light dark;--bg:#fff;--fg:#1c1c1c;--mute:#666}"
        "@media(prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e8e8e8;--mute:#9a9a9a}}"
        "body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1400px;padding:0 16px 48px}"
        "img{max-width:100%;height:auto;display:block;margin:6px 0 18px}h1{font-size:22px}h2{font-size:17px;margin-top:36px}"
        ".mute{color:var(--mute)}table{border-collapse:collapse;font-size:14px}td,th{padding:3px 10px;border-bottom:1px solid #8884;text-align:right}"
        "th:first-child,td:first-child{text-align:left}</style>",
        f"<h1>{html.escape(title)}</h1>",
    ]
    parts += [f"<p class='mute'>{html.escape(note)}</p>" for note in notes]
    if summary is not None and len(summary):
        parts.append("<table><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in summary.columns) + "</tr>")
        for row in summary.itertuples(index=False):
            parts.append("<tr>" + "".join(f"<td>{html.escape(f'{v:.3f}' if isinstance(v, float) else str(v))}</td>" for v in row) + "</tr>")
        parts.append("</table>")
    for section in sections:
        parts.append(f"<h2>{html.escape(section['tile_id'])}</h2><p class='mute'>{html.escape(section['caption'])}</p>")
        for kind, name in section["figures"]:
            parts.append(f"<p>{html.escape(kind)}</p><img loading='lazy' src='{html.escape(name)}' alt='{html.escape(kind)} of {html.escape(section['tile_id'])}'>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")
