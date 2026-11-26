"""Visual model comparison and the context test (tiny random encoders: shapes, plumbing and invariants)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import synthetic_cells

from mpn_al.evaluation.visual_compare import (
    agreement_table,
    context_effect,
    crop_box,
    densest_box,
    distance_to_border,
    model_geometry,
    overlay_instances,
    reference_scores,
    render_comparison,
    render_context,
    render_zoom,
    run_model,
    token_pca_rgb,
)
from mpn_al.segmenter import CellDecoder, CellSegmenter, LabelSchema, save_head

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def make_segmenter(tiny_model, seed):
    torch.manual_seed(seed)
    decoder = CellDecoder(tiny_model.embed_dim, base_width=16)
    return CellSegmenter(tiny_model, decoder, LabelSchema.single_class("cell"), seg_mpp=None).eval()


@pytest.fixture()
def image():
    return synthetic_cells(size=256)[0]


def test_geometry_reports_what_the_encoder_receives(tiny_model):
    segmenter = make_segmenter(tiny_model, 0)
    facts = model_geometry(segmenter, 1024, 1024, 0.25)
    assert facts["working_px"] == "1024x1024" and facts["token_grid"] == "64x64" and facts["passes"] == 1
    assert facts["encoder_input_px"] == "1024x1024" and facts["tile_covers_um"] == "256x256"
    assert model_geometry(segmenter, 1536, 1024, 0.25)["passes"] == 2  # larger than max_size: overlapping passes


def test_distance_and_box_helpers():
    distance = distance_to_border(10, 12)
    assert distance[0, 5] == 0 and distance[5, 5] == 4 and distance.max() == 4
    rows, cols = crop_box(100, 120, 20)
    assert (rows.start, rows.stop, cols.start, cols.stop) == (20, 80, 20, 100)
    with pytest.raises(ValueError):
        crop_box(100, 100, 50)


def test_token_pca_has_the_image_size_and_a_range():
    rgb = token_pca_rgb(np.random.default_rng(0).normal(size=(8, 4, 5)).astype(np.float32), 64, 80)
    assert rgb.shape == (64, 80, 3) and rgb.dtype == np.uint8 and rgb.max() > rgb.min()


def test_overlay_only_changes_outline_pixels():
    image = np.full((20, 20, 3), 100, np.uint8)
    instances = np.zeros((20, 20), np.int32)
    instances[5:15, 5:15] = 1
    out = overlay_instances(image, instances, (255, 0, 0))
    changed = (out != image).any(axis=-1)
    assert changed.any() and not changed[10, 10] and not changed[0, 0]


def test_views_comparison_and_agreement(tiny_model, image):
    segmenters = {"a": make_segmenter(tiny_model, 0), "b": make_segmenter(tiny_model, 1)}
    views = [run_model(name, seg, image, mpp=0.25, amp=False) for name, seg in segmenters.items()]
    for view in views:
        assert view.prob.shape == image.shape[:2] and view.hv.shape == (2, *image.shape[:2])
        assert view.token_rgb.shape == image.shape and view.instances.shape == image.shape[:2]
    figure = render_comparison(image, views, reference=views[0].instances, panel=128)
    assert figure.shape == (3 * 128, 4 * 128, 3) and figure.dtype == np.uint8
    table = agreement_table(views)
    assert list(table.index) == ["a", "b"] and table.loc["a", "a"] in (1.0, 0.0)  # 0 only if nothing was found at all


def test_a_model_agrees_with_itself_on_the_same_input(tiny_model, image):
    seg = make_segmenter(tiny_model, 0)
    first, second = (run_model("m", seg, image, mpp=0.25, amp=False) for _ in range(2))
    assert np.array_equal(first.instances, second.instances)


def test_context_effect_shapes_and_columns(tiny_model, image):
    seg = make_segmenter(tiny_model, 0)
    row, alone, with_context = context_effect("m", seg, image, 32, mpp=0.25, band=24, amp=False)
    assert alone.instances.shape == with_context.instances.shape == (192, 192)
    assert row["margin_px"] == 32 and 0.0 <= row["pq_all"] <= 1.0
    assert {"pq_band", "pq_interior", "prob_diff_0_16", "prob_diff_16_32", "prob_diff_256_inf"} <= set(row)
    figure = render_context(image, 32, [(row, alone, with_context)], panel=100)
    assert figure.shape == (100, 400, 3)


def test_script_end_to_end(tiny_model, tmp_path, image):
    from PIL import Image

    images = tmp_path / "images"
    images.mkdir()
    for index in range(2):
        Image.fromarray(image).save(images / f"S_x{index * 256}_y0_s256.png")
    heads = []
    for index, name in enumerate(("alpha", "beta")):
        path = tmp_path / f"{name}.pt"
        save_head(path, make_segmenter(tiny_model, index))
        heads.append(f"{name}={path}")
    sys.path.insert(0, str(SCRIPTS))
    import compare_models_visual

    out = tmp_path / "out"
    argv = ["--random-init", "--device", "cpu", "--no-amp", "--images", str(images), "--mpp", "0.25",
            "--context-margin", "32", "--panel", "64", "--out", str(out)]
    for spec in heads:
        argv += ["--head", spec]
    assert compare_models_visual.main(argv) == 0
    assert len(list(out.glob("*_compare.png"))) == 2 and len(list(out.glob("*_context.png"))) == 2
    assert len(list(out.glob("*_zoom.png"))) == 2
    assert len(pd.read_csv(out / "agreement.csv")) == 2 * 2  # two tiles x ordered pairs
    summary = pd.read_csv(out / "context_summary.csv")
    assert sorted(summary["model"]) == ["alpha", "beta"]
    assert set(pd.read_csv(out / "geometry.csv")["model"]) == {"alpha", "beta"}


def test_figures_of_non_square_images(tiny_model):
    image = np.random.default_rng(0).integers(0, 255, (192, 320, 3), dtype=np.uint8)
    seg = make_segmenter(tiny_model, 0)
    views = [run_model("m", seg, image, mpp=0.25, amp=False)]
    assert render_comparison(image, views, panel=96).ndim == 3
    result = context_effect("m", seg, image, 32, mpp=0.25, band=16, amp=False)
    assert render_context(image, 32, [result], panel=96).ndim == 3


def test_densest_box_finds_the_crowded_region():
    instances = np.zeros((200, 300), np.int32)
    instances[120:180, 200:290] = 1
    rows, cols = densest_box(instances, 64)
    assert (rows.stop - rows.start, cols.stop - cols.start) == (64, 64)
    assert rows.start >= 100 and cols.start >= 180  # the box sits on the block, not in the empty corner
    rows, cols = densest_box(instances, 1000)  # clipped to the shorter side, and still covers the whole block
    assert rows == slice(0, 200) and cols.stop - cols.start == 200 and cols.start <= 200 and cols.stop >= 290


def test_zoom_figure_and_scores_against_a_reference(tiny_model, image):
    seg = make_segmenter(tiny_model, 0)
    views = [run_model("a", seg, image, mpp=0.25, amp=False), run_model("b", make_segmenter(tiny_model, 1), image, mpp=0.25, amp=False)]
    reference = views[0].instances
    scores = reference_scores(views, reference)
    assert list(scores["model"]) == ["a", "b"] and 0.0 <= scores["pq"].min() and scores["pq"].max() <= 1.0
    assert scores.loc[0, "pq"] in (0.0, 1.0)  # a model against its own output: perfect, or empty
    box = densest_box(reference, 96)
    figure = render_zoom(image, views, box, reference=reference, scores=scores, panel=80)
    assert figure.shape == (80, 4 * 80, 3) and figure.dtype == np.uint8  # input | reference | a | b
    assert render_zoom(image, views, box, panel=80).shape == (80, 3 * 80, 3)
