"""The cluster smoke test, run on a CPU with a tiny random model."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from conftest import TINY_VIT

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plgrid" / "smoke_test.py"

FAST = [
    "--device", "cpu", "--random-init", "--tiles", "3", "--tile-size", "256", "--cells", "10", "--forward-sizes", "224",
    "--epochs", "1", "--steps", "3", "--train-batch", "2", "--crop-size", "128", "--base-width", "8", "--max-size", "256",
    "--segment-tiles", "1",
]


@pytest.fixture(scope="module")
def smoke():
    spec = importlib.util.spec_from_file_location("mpn_al_smoke_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stages(report):
    return {stage["stage"]: stage for stage in report["stages"]}


def test_smoke_test_passes_on_a_cpu_with_a_tiny_random_model(smoke, tmp_path):
    assert smoke.main(["--out", str(tmp_path), "--models", TINY_VIT, *FAST]) == 0
    report = json.loads((tmp_path / "smoke_report.json").read_text())
    assert report["ok"] and report["environment"]["torch"] == torch.__version__
    by_name = stages(report)
    assert set(by_name) == {
        "environment", "data", f"{TINY_VIT} / load", f"{TINY_VIT} / forward", f"{TINY_VIT} / tile embeddings",
        f"{TINY_VIT} / train head", f"{TINY_VIT} / segment",
    }
    assert all(stage["status"] in ("ok", "warn") for stage in by_name.values())
    assert by_name[f"{TINY_VIT} / load"]["details"]["embed_dim"] == 192
    assert by_name[f"{TINY_VIT} / forward"]["details"]["forward"][0]["tokens"] == 196
    assert by_name[f"{TINY_VIT} / tile embeddings"]["details"]["tiles"] == 3
    assert by_name[f"{TINY_VIT} / segment"]["details"]["tiles"][0]["cells"] >= 0
    summary = (tmp_path / "smoke_report.md").read_text(encoding="utf-8")
    assert "Result: **PASS**" in summary and "| input | tokens |" in summary


def test_a_failing_model_is_reported_and_the_other_stages_still_run(smoke, tmp_path):
    assert smoke.main(["--out", str(tmp_path), "--models", "no-such-model", TINY_VIT, *FAST]) == 1
    report = json.loads((tmp_path / "smoke_report.json").read_text())
    by_name = stages(report)
    assert not report["ok"] and by_name["no-such-model / load"]["status"] == "fail"
    assert "unknown base model" in by_name["no-such-model / load"]["error"]
    assert f"{TINY_VIT} / segment" in by_name and by_name[f"{TINY_VIT} / segment"]["status"] in ("ok", "warn")
    assert "Result: **FAIL**" in (tmp_path / "smoke_report.md").read_text(encoding="utf-8")


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without a GPU")
def test_cuda_is_demanded_when_it_is_asked_for(smoke, tmp_path):
    arguments = [a for a in FAST if a not in ("cpu",)]
    arguments[arguments.index("--device") + 1 :] = ["cuda", *arguments[arguments.index("--device") + 1 :]]
    assert smoke.main(["--out", str(tmp_path), "--models", TINY_VIT, *arguments]) == 1
    report = json.loads((tmp_path / "smoke_report.json").read_text())
    assert [s["stage"] for s in report["stages"]] == ["environment"]  # nothing else makes sense without the device
    assert "sees no CUDA device" in report["stages"][0]["error"]


def test_training_can_be_skipped(smoke, tmp_path):
    assert smoke.main(["--out", str(tmp_path), "--models", TINY_VIT, *FAST, "--steps", "0"]) == 0
    report = json.loads((tmp_path / "smoke_report.json").read_text())
    assert stages(report)[f"{TINY_VIT} / train head"]["details"] == {"skipped": "--steps 0"}
