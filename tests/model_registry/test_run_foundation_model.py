import argparse
import json

from PIL import Image
import pytest

from model_registry.patho_sam import PathoSAMAdapter
from model_registry.run_foundation_model import _create_adapter, _parser, run


def test_create_adapter_resolves_relative_module_and_default_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "weights" / "pathosam" / "weights.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    monkeypatch.setattr(PathoSAMAdapter, "load", lambda self, *args, **kwargs: None)
    args = argparse.Namespace(
        model="pathosam", seed=0, checkpoint=None, checkpoint_root=tmp_path / "weights"
    )

    adapter = _create_adapter(args, "cpu", {})

    assert isinstance(adapter, PathoSAMAdapter)
    assert args.checkpoint == checkpoint.resolve()


def test_checkpoint_discovery_requires_a_single_file(tmp_path):
    from model_registry.run_foundation_model import _resolve_checkpoint

    model_dir = tmp_path / "sam"
    model_dir.mkdir()
    first = model_dir / "vit_b.pth"
    first.touch()

    assert _resolve_checkpoint("sam", None, tmp_path) == first.resolve()

    (model_dir / "vit_h.pth").touch()
    with pytest.raises(ValueError, match="Multiple checkpoint files found"):
        _resolve_checkpoint("sam", None, tmp_path)


def test_checkpoint_root_environment_variable(tmp_path, monkeypatch):
    from model_registry.run_foundation_model import _resolve_checkpoint

    checkpoint = tmp_path / "cellsam" / "model.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    monkeypatch.setenv("MPN_AL_CHECKPOINT_ROOT", str(tmp_path))

    assert _resolve_checkpoint("cellsam", None) == checkpoint.resolve()


def test_png_directory_runs_each_tile_with_filename_coordinates(tmp_path):
    input_dir = tmp_path / "tiles"
    input_dir.mkdir()
    Image.new("RGB", (1024, 1024), (10, 20, 30)).save(
        input_dir / "slideA_x128_y256.png"
    )
    Image.new("RGB", (1024, 1024), (40, 50, 60)).save(
        input_dir / "slideA_x640_y256.png"
    )
    output_dir = tmp_path / "runs"
    args = _parser().parse_args([
        "--model", "sam",
        "--input", str(input_dir),
        "--output", str(output_dir),
        "--dry-run",
    ])

    run_dirs = run(args)

    assert len(run_dirs) == 2
    metadata = [
        json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        for run_dir in run_dirs
    ]
    assert [(item["tile"]["image_id"], item["tile"]["x"], item["tile"]["y"])
            for item in metadata] == [("slideA", 128, 256), ("slideA", 640, 256)]
    assert all(item["tile"]["requested_box"] == item["tile"]["readable_box"]
               for item in metadata)
    with Image.open(run_dirs[0] / "input.png") as image:
        assert image.size == (1024, 1024)
        assert image.getpixel((0, 0)) == (10, 20, 30)
