"""Train the segmentation head on top of a frozen SSL base model.

Only the decoder is trained; the encoder's weights are never changed or saved, which keeps
every base-model licence satisfied and makes a training run cheap enough to repeat after
each annotation round (use ``--init-head`` to warm-start from the previous round).

Example::

    python scripts/train_segmentation_head.py --base-model h-optimus-0 \\
        --images tiles_png/ --annotations data/04_annotations/round_00_initial_manual/by_tile \\
        --labels configs/labels/megakaryocyte_binary.yaml --mpp 0.25 --val-fraction 0.2 \\
        --out data/06_models/round_00/run_001

The best head by validation PQ is written to ``<out>/head.pt``.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..embeddings.cli import add_model_arguments, model_from_arguments
from ..embeddings.provenance import run_provenance, write_json
from ..evaluation.instance_metrics import aggregate, instance_counts
from ..segmenter.decoder import CellDecoder
from ..segmenter.labels import LabelSchema
from ..segmenter.model import CellSegmenter, predict_working_maps, read_head, save_head
from ..segmenter.postprocess import PostprocessConfig, hv_to_instances
from .dataset import CropDataset, TrainingTile, load_training_tiles
from .losses import HeadLoss

logger = logging.getLogger("mpn_al.training")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    steps_per_epoch: int = 100
    batch_size: int = 8
    crop_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    grad_clip: float = 1.0
    foreground_bias: float = 0.5  # share of crops centred on a cell
    amp: bool = True
    num_workers: int = 0
    seed: int = 0
    validate_every: int = 1
    max_size: int = 1024  # largest side the segmenter sees in one pass during validation


def evaluate(
    segmenter: CellSegmenter,
    tiles: Sequence[TrainingTile],
    *,
    work_mpp: float | None = None,
    postprocess: PostprocessConfig = PostprocessConfig(),
    amp: bool = True,
    max_size: int = 1024,
) -> dict[str, float]:
    """Segment whole tiles and score the instances against their labels (PQ, DQ, SQ, Dice)."""
    counts = []
    for tile in tiles:
        maps = predict_working_maps(segmenter, tile.image, amp=amp, max_size=max_size)
        predicted = hv_to_instances(maps.prob, maps.hv, mpp=work_mpp, config=postprocess)
        counts.append(instance_counts(tile.instances, predicted, ignore=tile.ignore))
    return aggregate(counts)


def train_head(
    segmenter: CellSegmenter,
    train_tiles: Sequence[TrainingTile],
    val_tiles: Sequence[TrainingTile] = (),
    config: TrainConfig = TrainConfig(),
    out_dir: str | Path | None = None,
    *,
    work_mpp: float | None = None,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, float]]:
    """Fit the decoder of ``segmenter``; returns the per-epoch history.

    With ``out_dir`` the best head (highest validation PQ, or lowest training loss when there
    is no validation set) is saved as ``head.pt``, the final one as ``last.pt``, and the
    history as ``history.json``.
    """
    device = segmenter.base.device
    torch.manual_seed(config.seed)
    dataset = CropDataset(
        train_tiles,
        crop_size=config.crop_size,
        length=config.steps_per_epoch * config.batch_size,
        foreground_bias=config.foreground_bias,
        seed=config.seed,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True)
    criterion = HeadLoss().to(device)
    parameters = list(segmenter.decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
    total_steps = config.epochs * config.steps_per_epoch

    def learning_rate_factor(step: int) -> float:  # linear warm-up, then cosine decay
        warmup = min(1.0, (step + 1) / max(1, config.warmup_steps))
        return warmup * 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / total_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    out_dir = Path(out_dir) if out_dir is not None else None
    history: list[dict[str, float]] = []
    best_score = -math.inf

    for epoch in range(1, config.epochs + 1):
        dataset.epoch = epoch
        segmenter.train()  # the frozen encoder stays in eval mode
        sums: dict[str, float] = defaultdict(float)
        started = time.perf_counter()
        for batch in loader:
            batch = {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = segmenter(batch["image"])
            loss, parts = criterion(outputs, batch)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            sums["loss"] += float(loss.detach())
            for name, value in parts.items():
                sums[name] += value
        record: dict[str, float] = {"epoch": float(epoch)}
        record.update({name: total / len(loader) for name, total in sums.items()})
        record["seconds"] = round(time.perf_counter() - started, 1)

        validated = bool(val_tiles) and (epoch % config.validate_every == 0 or epoch == config.epochs)
        if validated:
            metrics = evaluate(segmenter, val_tiles, work_mpp=work_mpp, amp=config.amp, max_size=config.max_size)
            record.update({f"val_{name}": value for name, value in metrics.items()})
            score = metrics["pq"]
        else:
            score = -record["loss"] if not val_tiles else -math.inf
        history.append(record)
        logger.info(
            "epoch %d/%d  loss %.4f%s  (%.0fs)",
            epoch, config.epochs, record["loss"],
            f"  val PQ {record['val_pq']:.3f} DQ {record['val_dq']:.3f} Dice {record['val_dice']:.3f}" if validated else "",
            record["seconds"],
        )
        if out_dir is not None:
            if score > best_score:
                save_head(out_dir / "head.pt", segmenter, epoch=epoch, metrics=record, **(extra or {}))
            write_json(out_dir / "history.json", {"config": asdict(config), "history": history})
        best_score = max(best_score, score)

    if out_dir is not None:
        save_head(out_dir / "last.pt", segmenter, epoch=config.epochs, metrics=history[-1], **(extra or {}))
    return history


# ---- command line ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a cell-segmentation head on a frozen SSL base model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Example::")[1],
    )
    add_model_arguments(parser)
    data = parser.add_argument_group("training data (give --masks or --annotations)")
    data.add_argument("--images", type=Path, required=True, help="directory of tile images named <tile_id>.<ext>")
    data.add_argument("--annotations", type=Path, help="directory of <tile_id>.geojson (QuPath export, level-0 coordinates)")
    data.add_argument("--masks", type=Path, help="directory of instance masks (see mpn_al.training.dataset)")
    data.add_argument("--labels", type=Path, help="label schema YAML (default: one class, 'cell')")
    data.add_argument("--mpp", type=float, help="resolution of the images in um/px (40x is about 0.25)")
    data.add_argument("--seg-mpp", type=float, help="resolution the head works at (default: the images' own)")
    data.add_argument(
        "--unlabeled",
        choices=("background", "ignore"),
        default="background",
        help="with --annotations: what pixels without any annotation are",
    )
    validation = parser.add_argument_group("validation (pick one; without any, no model selection is possible)")
    validation.add_argument("--val-images", type=Path, help="separate directory of validation images")
    validation.add_argument("--val-annotations", type=Path)
    validation.add_argument("--val-masks", type=Path)
    validation.add_argument("--val-ids", type=Path, help="text file of tile ids (from --images) held out for validation")
    validation.add_argument("--val-fraction", type=float, default=0.0, help="hold out this share of tiles at random")
    model = parser.add_argument_group("head")
    model.add_argument("--base-width", type=int, default=64, help="decoder width; 64 is the full-size CellViT-like head")
    model.add_argument("--decoder-norm", choices=("group", "batch"), default="group")
    model.add_argument("--dropout", type=float, default=0.0)
    model.add_argument("--init-head", type=Path, help="warm-start the decoder from this head checkpoint")
    training = parser.add_argument_group("optimisation")
    defaults = TrainConfig()
    training.add_argument("--epochs", type=int, default=defaults.epochs)
    training.add_argument("--steps-per-epoch", type=int, default=defaults.steps_per_epoch)
    training.add_argument("--batch-size", type=int, default=defaults.batch_size)
    training.add_argument("--crop-size", type=int, default=defaults.crop_size)
    training.add_argument("--lr", type=float, default=defaults.lr)
    training.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    training.add_argument("--seed", type=int, default=defaults.seed)
    training.add_argument("--num-workers", type=int, default=defaults.num_workers)
    training.add_argument(
        "--validate-every",
        type=int,
        default=defaults.validate_every,
        help="validate every N epochs, and always after the last one (whole-tile validation can be slow)",
    )
    training.add_argument(
        "--max-size",
        type=int,
        default=defaults.max_size,
        help="largest side in px the segmenter sees in one validation pass; lower it to save memory "
        "and time with large encoders (multiple of 16, above 128)",
    )
    parser.add_argument("--out", type=Path, required=True, help="run directory, e.g. data/06_models/<round_id>/<run_id>")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if (args.masks is None) == (args.annotations is None):
        parser.error("give exactly one of --masks and --annotations")

    schema = LabelSchema.from_yaml(args.labels) if args.labels else LabelSchema.single_class("cell")
    common = {"schema": schema, "mpp": args.mpp, "seg_mpp": args.seg_mpp, "unlabeled": args.unlabeled}
    tiles = load_training_tiles(args.images, masks=args.masks, annotations=args.annotations, **common)
    train_tiles, val_tiles = _split(tiles, args, common)
    background = sum(int(((t.instances == 0) & ~t.ignore).sum()) for t in train_tiles)
    cells = sum(t.num_instances for t in train_tiles)
    if cells == 0 or background == 0:
        parser.error(
            f"the training tiles hold {cells} cells and {background} background pixels; the head needs both "
            "(with --unlabeled ignore, background must be marked by a region of interest)"
        )
    logger.info(
        "%d training tiles (%d cells), %d validation tiles, classes %s",
        len(train_tiles), cells, len(val_tiles), [c.name for c in schema.classes],
    )

    base = model_from_arguments(args)
    decoder_config = {
        "num_types": schema.num_classes if schema.num_classes > 1 else 0,
        "base_width": args.base_width,
        "norm": args.decoder_norm,
        "dropout": args.dropout,
    }
    decoder = CellDecoder(base.embed_dim, **decoder_config)
    if args.init_head is not None:
        previous = read_head(args.init_head)
        if previous["decoder"] != decoder_config or previous["encoder"]["embed_dim"] != base.embed_dim:
            parser.error(f"{args.init_head} has a different head or encoder, so it cannot initialise this one")
        decoder.load_state_dict(previous["state_dict"])
        logger.info("decoder initialised from %s", args.init_head)
    segmenter = CellSegmenter(base, decoder, schema, seg_mpp=args.seg_mpp).to(args.device)
    logger.info("trainable parameters: %.1fM (decoder only)", sum(p.numel() for p in decoder.parameters()) / 1e6)

    config = TrainConfig(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amp=not args.no_amp,
        num_workers=args.num_workers,
        seed=args.seed,
        validate_every=args.validate_every,
        max_size=args.max_size,
    )
    extra = {
        "train_tiles": [t.tile_id for t in train_tiles],
        "val_tiles": [t.tile_id for t in val_tiles],
        "training": asdict(config),
        "init_head": str(args.init_head) if args.init_head else None,
    }
    started = time.perf_counter()
    history = train_head(
        segmenter, train_tiles, val_tiles, config, args.out, work_mpp=args.seg_mpp or args.mpp, extra=extra
    )
    best = max(history, key=lambda r: r.get("val_pq", -r["loss"])) if history else {}
    write_json(
        args.out / "run.json",
        run_provenance(
            vars(args),
            base_model=base.describe(),
            labels=schema.to_dict(),
            decoder=decoder_config,
            best=best,
            seconds=round(time.perf_counter() - started, 1),
            **{key: extra[key] for key in ("train_tiles", "val_tiles")},
        ),
    )
    logger.info("best epoch %d -> %s", int(best.get("epoch", 0)), args.out / "head.pt")
    return 0


def _split(tiles: list[TrainingTile], args: argparse.Namespace, common: dict) -> tuple[list, list]:
    if args.val_images is not None:
        if (args.val_masks is None) == (args.val_annotations is None):
            raise SystemExit("--val-images needs exactly one of --val-masks and --val-annotations")
        return tiles, load_training_tiles(
            args.val_images, masks=args.val_masks, annotations=args.val_annotations, **common
        )
    if args.val_ids is not None:
        held_out = {line.strip() for line in args.val_ids.read_text(encoding="utf-8").splitlines() if line.strip()}
    elif args.val_fraction > 0:
        logger.warning(
            "splitting by tile at random: tiles of one patient can land on both sides. "
            "Use --val-ids built from the fixed patient split for numbers you intend to report."
        )
        order = np.random.default_rng(args.seed).permutation(len(tiles))
        count = max(1, int(round(args.val_fraction * len(tiles))))
        held_out = {tiles[i].tile_id for i in order[:count]}
    else:
        return tiles, []
    train = [t for t in tiles if t.tile_id not in held_out]
    val = [t for t in tiles if t.tile_id in held_out]
    if not train or not val:
        raise SystemExit(f"the split leaves {len(train)} training and {len(val)} validation tiles; both must be non-empty")
    return train, val
