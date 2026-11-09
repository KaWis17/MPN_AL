import json

from PIL import Image

from src.model_registry.run_foundation_model import _parser, run


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
