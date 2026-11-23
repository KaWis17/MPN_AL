from pathlib import Path

import numpy as np

from model_registry.comparison import _parser, evaluate


def test_comparison_defaults_output_to_results_directory():
    args = _parser().parse_args(["--runs", "runs"])

    assert args.output == Path(__file__).resolve().parents[2] / "src" / "model_registry" / "results"


def test_megakaryocyte_metrics_ignore_predictions_of_other_cells():
    prediction = np.array([[1, 1, 1, 2, 2, 0]], dtype=np.int32)
    megakaryocyte_reference = np.array([[7, 7, 7, 0, 0, 0]], dtype=np.int32)

    metrics = evaluate(prediction, ref=None, megakaryocyte_ref=megakaryocyte_reference)

    assert metrics["binary_dice"] == "NA"
    assert metrics["megakaryocyte_gt_count"] == 1
    assert metrics["megakaryocyte_detected_count"] == 1
    assert metrics["megakaryocyte_detection_recall"] == 1.0
    assert metrics["megakaryocyte_mean_iou"] == 1.0


def test_megakaryocyte_metrics_score_unmatched_reference_as_zero():
    prediction = np.zeros((2, 2), dtype=np.int32)
    megakaryocyte_reference = np.array([[1, 1], [1, 0]], dtype=np.int32)

    metrics = evaluate(prediction, ref=None, megakaryocyte_ref=megakaryocyte_reference)

    assert metrics["megakaryocyte_detected_count"] == 0
    assert metrics["megakaryocyte_detection_recall"] == 0.0
    assert metrics["megakaryocyte_mean_iou"] == 0.0


def test_megakaryocyte_mean_iou_preserves_subthreshold_overlap():
    prediction = np.array([[1, 0], [0, 0]], dtype=np.int32)
    megakaryocyte_reference = np.array([[4, 4], [4, 4]], dtype=np.int32)

    metrics = evaluate(prediction, ref=None, megakaryocyte_ref=megakaryocyte_reference)

    assert metrics["megakaryocyte_detected_count"] == 0
    assert metrics["megakaryocyte_detection_recall"] == 0.0
    assert metrics["megakaryocyte_mean_iou"] == 0.25


def test_megakaryocyte_metrics_are_na_without_annotated_targets():
    prediction = np.zeros((2, 2), dtype=np.int32)
    megakaryocyte_reference = np.zeros((2, 2), dtype=np.int32)

    metrics = evaluate(prediction, ref=None, megakaryocyte_ref=megakaryocyte_reference)

    assert metrics["megakaryocyte_gt_count"] == 0
    assert metrics["megakaryocyte_detection_recall"] == "NA"
    assert metrics["megakaryocyte_mean_iou"] == "NA"
