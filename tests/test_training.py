"""Targets, loss, crops and an end-to-end run: train a head, save it, segment with it."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import synthetic_cells

from mpn_al.image_backend import TileRef
from mpn_al.segmenter import (
    CellDecoder,
    CellSegmenter,
    LabelSchema,
    PipelineConfig,
    RunWriter,
    SegmentationPipeline,
    load_segmenter,
    predict_maps,
    read_head,
    read_tile_stats,
    save_head,
)
from mpn_al.segmenter.model import predict_working_maps, working_size
from mpn_al.training import CropDataset, HeadLoss, TrainConfig, evaluate, hv_maps, make_training_tile, train_head
from mpn_al.training.dataset import _random_symmetry


def test_hv_maps_run_from_minus_one_to_one_across_each_cell():
    instances = np.zeros((32, 48), dtype=np.int32)
    instances[4:15, 10:31] = 1  # 11 rows x 21 columns: integer centre
    maps = hv_maps(instances)
    assert maps.shape == (2, 32, 48)
    assert maps[0, 9, 10] == -1 and maps[0, 9, 20] == 0 and maps[0, 9, 30] == 1
    assert maps[1, 4, 20] == -1 and maps[1, 9, 20] == 0 and maps[1, 14, 20] == 1
    assert (maps[:, instances == 0] == 0).all()
    assert np.all(np.diff(maps[0, 9, 10:31]) > 0)


def test_hv_maps_of_a_crop_equal_the_crop_of_the_full_maps():
    _, instances = synthetic_cells(size=256, count=30, seed=2)
    region = (slice(40, 168), slice(96, 224))
    assert np.array_equal(hv_maps(instances, region), hv_maps(instances)[:, region[0], region[1]])


@pytest.mark.parametrize("seed", range(8))
def test_symmetry_augmentation_transforms_hv_as_vectors(seed):
    """Flipping the image must give the HV maps of the flipped cells, not merely flipped maps."""
    yy, xx = np.mgrid[:64, :64]
    instances = np.zeros((64, 64), dtype=np.int32)
    instances[(xx - 20) ** 2 + (yy - 24) ** 2 <= 9**2] = 1  # symmetric shapes with integer centres
    instances[40:55, 30:51] = 2
    image = np.random.default_rng(0).random((64, 64, 3), dtype=np.float32)
    zeros = np.zeros((64, 64), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    _, flipped, _, _, hv = _random_symmetry(rng, image, instances, zeros, zeros.astype(bool), hv_maps(instances))
    assert np.allclose(hv, hv_maps(np.ascontiguousarray(flipped)), atol=1e-6)


def test_loss_rewards_correct_maps_and_skips_ignored_pixels():
    _, instances = synthetic_cells(size=128, count=10, seed=1)
    foreground = torch.from_numpy(instances > 0)
    hv = torch.from_numpy(hv_maps(instances))[None]
    targets = {
        "np": foreground.long()[None],
        "hv": hv,
        "nt": foreground.long()[None],
        "valid": torch.ones(1, 128, 128),
    }
    logits = torch.where(foreground, 12.0, -12.0)
    good = {"np": torch.stack([-logits, logits])[None], "hv": hv.clone(), "nt": torch.stack([-logits, logits])[None]}
    criterion = HeadLoss()
    perfect, parts = criterion(good, targets)
    assert float(perfect) < 0.02 and set(parts) == {"np_ce", "np_dice", "hv_mse", "hv_grad", "nt_ce", "nt_dice"}

    bad = {name: tensor.clone() for name, tensor in good.items()}
    bad["np"][..., :64] *= -1  # wrong on the left half
    bad["hv"][..., :64] = 0.7
    assert float(criterion(bad, targets)[0]) > 1.0
    targets["valid"][..., :64] = 0  # ... which nobody annotated
    assert float(criterion(bad, targets)[0]) < 0.02


def test_crop_dataset_is_deterministic_and_changes_between_epochs():
    image, instances = synthetic_cells(size=160, count=8, seed=4)
    small = make_training_tile("small", image[:96, :96], instances[:96, :96])  # smaller than a crop: padded
    dataset = CropDataset([make_training_tile("a", image, instances), small], crop_size=128, length=6, seed=1)
    first = dataset[3]
    assert first["image"].shape == (3, 128, 128) and first["image"].dtype == torch.float32
    assert first["hv"].shape == (2, 128, 128) and first["np"].dtype == torch.int64
    assert all(torch.equal(first[key], dataset[3][key]) for key in first)
    dataset.epoch = 1
    assert not torch.equal(first["image"], dataset[3]["image"])
    padded = next(t for t in dataset.tiles if t.tile_id == "small")
    assert padded.image.shape == (128, 128, 3) and padded.ignore[100:, :].all() and not padded.ignore[:96, :96].any()


def test_masks_with_gaps_in_their_numbering_are_renumbered():
    instances = np.zeros((128, 128), dtype=np.int32)
    instances[10:30, 10:30], instances[60:90, 60:100] = 7, 4000  # ids as left by cropping a larger label image
    tile = make_training_tile("gaps", np.zeros((128, 128, 3), np.uint8), instances)
    assert tile.num_instances == 2 and set(np.unique(tile.instances)) == {0, 1, 2}
    assert all(box is not None for box in tile.objects)
    dataset = CropDataset([tile], crop_size=128, length=8, foreground_bias=1.0, augment=False)
    assert all(dataset[i]["np"].sum() == 400 + 1200 for i in range(8))


def test_head_working_at_half_resolution_reports_slide_coordinates(tiny_model):
    """seg_mpp 0.5 on a 0.25 um/px tile: maps are half size, geometry is still in level-0 pixels."""
    segmenter = CellSegmenter(tiny_model, CellDecoder(192, base_width=8), LabelSchema.single_class(), seg_mpp=0.5)
    pipeline = SegmentationPipeline(segmenter=segmenter)
    image = synthetic_cells(size=256, count=12, seed=3)[0]
    tile = TileRef("tl_S1_x1024_y2048_s256", x=1024, y=2048, width=256, height=256, mpp=0.25)
    result = pipeline.run_tile(image, tile)
    assert result.instances.shape == (128, 128) and result.scale == (2.0, 2.0)
    areas = np.bincount(result.instances.ravel())[1:]
    for cell in result.cells:  # an untrained head finds arbitrary blobs; only their bookkeeping matters here
        assert cell.area == areas[cell.label - 1] * 4
        assert 1024 <= cell.polygon[:, 0].min() and cell.polygon[:, 0].max() <= 1280
        assert 2048 <= cell.polygon[:, 1].min() and cell.polygon[:, 1].max() <= 2304
    with pytest.raises(ValueError, match="resolution must be known"):
        pipeline.run_tile(image, TileRef("no_mpp", width=256, height=256))


def test_working_size_follows_the_requested_resolution():
    assert working_size(1024, 1024, 0.25, None) == (1024, 1024)
    assert working_size(1024, 1024, 0.25, 0.5) == (512, 512)
    assert working_size(1010, 1500, 0.23, None) == (1008, 1504)  # nearest multiples of 16
    with pytest.raises(ValueError, match="resolution must be known"):
        working_size(1024, 1024, None, 0.5)


def test_patch14_encoder_is_fed_a_resampled_image_and_stays_pixel_aligned():
    """A patch-14 encoder sees 16 tokens per 256 px (a 224 px input); outputs stay at 256 px."""
    from mpn_al.embeddings import BaseModelSpec, build_base_model

    spec = BaseModelSpec(
        name="tiny-p14", backend="timm", source="vit_tiny_patch16_224", create_kwargs={"patch_size": 14, "img_size": 224}
    )
    base = build_base_model(spec, pretrained=False)
    assert base.patch_size == 14
    segmenter = CellSegmenter(base, CellDecoder(base.embed_dim, num_types=2, base_width=8), LabelSchema.single_class())
    output = segmenter(torch.rand(1, 3, 256, 192))
    assert output["np"].shape == (1, 2, 256, 192) and output["hv"].shape == (1, 2, 256, 192)
    assert output["nt"].shape == (1, 3, 256, 192) and output["tokens"].shape == (1, 192, 16, 12)


def test_large_images_are_predicted_in_overlapping_passes(tiny_model):
    torch.manual_seed(0)
    segmenter = CellSegmenter(tiny_model, CellDecoder(192, base_width=8), LabelSchema.single_class()).eval()
    image = synthetic_cells(size=320, count=20, seed=6)[0][:256]  # 256 x 320
    whole = predict_working_maps(segmenter, image, max_size=512)
    tiled = predict_working_maps(segmenter, image, max_size=192, overlap=64)
    assert tiled.prob.shape == whole.prob.shape == (256, 320)
    assert tiled.tokens.shape == whole.tokens.shape == (192, 16, 20)
    assert np.isfinite(tiled.prob).all() and np.isfinite(tiled.hv).all()
    # A crop sees less context than the whole image, so the maps agree closely but not exactly.
    assert np.abs(tiled.prob - whole.prob).mean() < 0.05
    resampled = predict_maps(segmenter, image[:250, :300], mpp=0.25)
    assert resampled.prob.shape == (256, 304) and resampled.scale == pytest.approx((300 / 304, 250 / 256))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A small head trained on synthetic tiles: the full training path on CPU in well under a minute."""
    from mpn_al.embeddings import build_base_model

    torch.manual_seed(0)
    base = build_base_model("timm:vit_tiny_patch16_224", pretrained=False)
    schema = LabelSchema.single_class("cell")
    segmenter = CellSegmenter(base, CellDecoder(base.embed_dim, base_width=16), schema)
    tiles = [make_training_tile(f"train_{i}", *synthetic_cells(size=256, count=16, seed=i)) for i in range(6)]
    held_out = [make_training_tile(f"val_{i}", *synthetic_cells(size=256, count=16, seed=100 + i)) for i in range(2)]
    out_dir = tmp_path_factory.mktemp("run")
    before = evaluate(segmenter, held_out, work_mpp=0.25)
    config = TrainConfig(
        epochs=5, steps_per_epoch=30, batch_size=4, crop_size=128, lr=2e-3, warmup_steps=10, validate_every=5
    )
    history = train_head(segmenter, tiles, held_out, config, out_dir, work_mpp=0.25, extra={"note": "test"})
    return {"segmenter": segmenter, "history": history, "before": before, "out_dir": out_dir, "held_out": held_out}


def test_head_learns_to_segment_synthetic_cells(trained):
    history = trained["history"]
    assert history[-1]["loss"] < 0.5 * history[0]["loss"]
    assert history[-1]["val_pq"] > max(0.5, trained["before"]["pq"] + 0.3), history[-1]
    assert history[-1]["val_dice"] > 0.9
    assert (trained["out_dir"] / "head.pt").exists() and (trained["out_dir"] / "last.pt").exists()
    saved = json.loads((trained["out_dir"] / "history.json").read_text())
    assert len(saved["history"]) == 5 and saved["config"]["crop_size"] == 128


def test_only_the_decoder_is_trained_and_saved(trained):
    payload = read_head(trained["out_dir"] / "head.pt")
    assert payload["extra"]["note"] == "test" and payload["extra"]["epoch"] == 5
    assert payload["base_model"]["source"] == "vit_tiny_patch16_224"
    assert payload["encoder"] == {"embed_dim": 192, "patch_size": 16, "num_layers": 12}
    decoder_keys = set(trained["segmenter"].decoder.state_dict())
    assert set(payload["state_dict"]) == decoder_keys  # no encoder weights in the file
    assert all(not p.requires_grad for p in trained["segmenter"].base.parameters())


def test_saved_head_reproduces_the_predictions(trained):
    image = trained["held_out"][0].image
    expected = predict_working_maps(trained["segmenter"], image)
    path = save_head(trained["out_dir"] / "copy.pt", trained["segmenter"])
    reloaded = load_segmenter(path, base=trained["segmenter"].base)
    assert reloaded.labels.classes[0].name == "cell" and reloaded.layers == (2, 5, 8, 11)
    assert np.allclose(predict_working_maps(reloaded, image).prob, expected.prob, atol=1e-5)

    from mpn_al.embeddings import build_base_model

    other = build_base_model("timm:vit_small_patch16_224", pretrained=False)
    with pytest.raises(ValueError, match="only works with the base model it was trained on"):
        load_segmenter(path, base=other)


def test_pipeline_writes_geojson_cells_and_tile_statistics(trained, tmp_path):
    pipeline = SegmentationPipeline(segmenter=trained["segmenter"], config=PipelineConfig(default_mpp=0.25))
    writer = RunWriter(tmp_path, labels=pipeline.labels, base_model=pipeline.base_model.name, save_masks=True)
    empty = np.full((256, 256, 3), 240, dtype=np.uint8)
    tiles = {
        "tl_S7_x2048_y1024_s256": trained["held_out"][0].image,
        "tl_S7_x2304_y1024_s256": trained["held_out"][1].image,
        "tl_S7_x2560_y1024_s256": empty,
    }
    results = {}
    for tile_id, image in tiles.items():
        tile = TileRef(tile_id, x=int(tile_id.split("_x")[1].split("_")[0]), y=1024, width=256, height=256,
                       slide_id="S7", prep_id="prep_t")
        assert not writer.is_done(tile_id)
        results[tile_id] = pipeline.run_tile(image, tile)
        writer.write(results[tile_id], seconds=0.1)
        assert writer.is_done(tile_id)

    first = results["tl_S7_x2048_y1024_s256"]
    assert first.method == "head" and len(first.cells) >= 10
    assert first.cell_embeddings.shape == (len(first.cells), 192)
    assert all(2048 <= c.centroid[0] <= 2304 and 1024 <= c.centroid[1] <= 1280 for c in first.cells)
    assert 0 <= first.stats["mean_entropy"] <= 1

    features = json.loads(writer.geojson_path("tl_S7_x2048_y1024_s256").read_text())["features"]
    assert len(features) == len(first.cells)
    assert features[0]["properties"]["classification"]["name"] == "cell"

    cells = pd.read_parquet(tmp_path / "cells")  # every tile, including the empty one, as one table
    assert len(cells) == sum(len(r.cells) for r in results.values())
    assert cells["cell_id"].is_unique and set(cells["tile_id"]) <= set(tiles)
    assert np.stack(cells["embedding"].to_numpy()).shape == (len(cells), 192)
    assert (cells["prep_id"] == "prep_t").all() and (cells["base_model"] == pipeline.base_model.name).all()
    assert cells["x"].between(2048, 2816).all() and (cells["area_um2"] > 3.0).all()
    assert len(pd.read_parquet(writer.cells_path("tl_S7_x2560_y1024_s256"))) == len(results["tl_S7_x2560_y1024_s256"].cells)

    stats = read_tile_stats(tmp_path / "tile_stats.jsonl")
    assert list(stats["tile_id"]) == list(tiles) and stats.loc[0, "n_cells"] == len(first.cells)
    with np.load(tmp_path / "masks" / "tl_S7_x2048_y1024_s256.npz") as archive:
        assert archive["instances"].shape == (256, 256) and archive["instances"].max() == len(first.cells)


def test_stain_baseline_pipeline_embeds_cells_with_the_base_model(tiny_model):
    image, truth = synthetic_cells(size=256, count=14, seed=0)
    tile = TileRef("plain", width=256, height=256, mpp=0.25)
    pipeline = SegmentationPipeline(base_model=tiny_model)
    result = pipeline.run_tile(image, tile)
    assert result.method == "stain-baseline" and abs(len(result.cells) - truth.max()) <= 1
    assert result.cells[0].class_name == "nucleus" and result.cells[0].prob is None
    assert result.cell_embeddings.shape == (len(result.cells), 192)  # tokens of a 0.5 um/px pass
    assert "mean_entropy" not in result.stats  # the baseline has no probability map
    without_model = SegmentationPipeline().run_tile(image, tile)
    assert without_model.cell_embeddings is None and len(without_model.cells) == len(result.cells)
