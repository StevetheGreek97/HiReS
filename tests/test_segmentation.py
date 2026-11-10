import cv2
import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import BLOBS, FakeModel

import hires
from hires import MissingDependencyError, Segmenter
from hires.segmentation import _annotations_from_result

SETTINGS = {"tile_size": 256, "overlap": 96, "batch_size": 4}


def centres(collection):
    """Blob centres in whole pixels (a 40 px blob at x has its centre at x + 19.5)."""
    width, height = collection.image_size
    return sorted(
        (int(a.polygon.centroid.x * width + 0.75), int(a.polygon.centroid.y * height + 0.75))
        for a in collection
    )


def test_segment_file(scan_path, fake_model):
    result = Segmenter(fake_model, **SETTINGS, device="cuda:0").segment(scan_path)
    assert result.name == "scan" and result.image_path == scan_path
    assert result.image_size == (640, 520)
    # Every blob is found exactly once, although most are seen by several tiles.
    assert centres(result) == sorted((x + 20, y + 20) for x, y in BLOBS)
    assert result.class_counts() == {0: 4, 1: 1}
    areas = result.to_frame()["area"]
    assert areas.between(39 * 39 - 1, 40 * 40).all()
    confidences = [a.confidence for a in result]
    assert confidences == sorted(confidences, reverse=True)
    # 640 x 520 with 256 px tiles every 160 px: 4 x 3 tiles in batches of 4.
    assert [call["n"] for call in fake_model.calls] == [4, 4, 4]
    assert fake_model.calls[0] == {"n": 4, "conf": 0.5, "imgsz": 1024, "device": "cuda:0"}


def test_segment_array_and_function(scan_path, fake_model):
    pixels = cv2.imread(str(scan_path))
    from_array = Segmenter(fake_model, **SETTINGS).segment(pixels, name="mine")
    assert from_array.name == "mine" and from_array.image_path is None
    assert centres(from_array) == centres(hires.segment(scan_path, fake_model, **SETTINGS))
    assert Segmenter(fake_model, **SETTINGS).segment(pixels).name is None
    with pytest.raises(ValueError, match="shape"):
        Segmenter(fake_model).segment(pixels[:, :, 0])
    with pytest.raises(FileNotFoundError):
        Segmenter(fake_model).segment(scan_path.with_name("nope.png"))


def test_segment_matches_untiled_result(scan_path, fake_model):
    tiled = Segmenter(fake_model, **SETTINGS).segment(scan_path)
    whole = Segmenter(fake_model, tile_size=(640, 520), overlap=0).segment(scan_path)
    assert centres(tiled) == centres(whole)
    matches = hires.metrics.match(whole, tiled, iou_threshold=0.95)
    assert matches.counts() == {"tp": 5, "misclassified": 0, "fp": 0, "fn": 0}


def test_segment_drops_objects_at_borders(tmp_path, fake_model):
    pixels = np.zeros((300, 300, 3), dtype=np.uint8)
    pixels[0:30, 100:130] = 255  # touches the image border
    pixels[150:180, 150:180] = 255
    result = Segmenter(fake_model, tile_size=200, overlap=80).segment(pixels)
    assert centres(result) == [(165, 165)]
    # Without a margin nothing is filtered: the border blob stays, and so does the
    # fragment of it that the neighbouring tile sees.
    unfiltered = Segmenter(fake_model, tile_size=200, overlap=80, edge_margin=0).segment(pixels)
    assert len(unfiltered) == 3


def test_segment_blank_image(fake_model):
    result = Segmenter(fake_model, **SETTINGS).segment(np.zeros((300, 300, 3), np.uint8))
    assert len(result) == 0 and result.image_size == (300, 300)


def test_segment_debug_dir(scan_path, fake_model, tmp_path):
    debug = tmp_path / "debug"
    Segmenter(fake_model, **SETTINGS).segment(scan_path, debug_dir=debug)
    assert len(list((debug / "tiles").glob("scan_*_*.png"))) == 12
    assert len(list((debug / "tile_labels").glob("*.txt"))) == 12
    assert 0 < len(list((debug / "tile_overlays").glob("*.png"))) < 12
    merged = hires.read_txt(debug / "scan_merged.txt")
    assert len(merged) > len(BLOBS)  # duplicates from overlapping tiles, before NMS
    tile = hires.read_txt(debug / "tile_labels" / "scan_0_0.txt")
    assert len(tile) == 2  # blob 0 in full, blob 1 cut by the tile border


def test_segmenter_settings(fake_model, tmp_path):
    segmenter = Segmenter(fake_model, tile_size=(512, 256), overlap=10, confidence=0.25)
    assert segmenter.class_names == {0: "bright", 1: "dim"}
    assert segmenter.parameters() == {
        "model": None, "tile_size": [512, 256], "overlap": 10, "confidence": 0.25,
        "inference_size": 1024, "device": "cpu", "edge_margin": 0.01, "nms_iou": 0.7,
        "batch_size": 8,
    }  # fmt: skip
    assert "tile_size=(512, 256)" in repr(segmenter)
    for bad in (
        {"overlap": 1024}, {"overlap": -1}, {"confidence": 1.5}, {"nms_iou": -0.1},
        {"edge_margin": 0.5}, {"batch_size": 0}, {"tile_size": 0},
    ):  # fmt: skip
        with pytest.raises(ValueError):
            Segmenter(fake_model, **bad)
    with pytest.raises(TypeError, match="predict"):
        Segmenter(object())
    with pytest.raises(FileNotFoundError, match="weights"):
        Segmenter(tmp_path / "missing.pt")


def test_segmenter_loads_weights_with_ultralytics(tmp_path, monkeypatch, scan_path):
    import types

    weights = tmp_path / "model.pt"
    weights.write_bytes(b"")
    loaded = []
    fake = types.ModuleType("ultralytics")
    fake.YOLO = lambda path: loaded.append(path) or FakeModel()
    monkeypatch.setitem(__import__("sys").modules, "ultralytics", fake)
    segmenter = Segmenter(weights, **SETTINGS)
    assert loaded == [str(weights)] and segmenter.parameters()["model"] == str(weights)
    assert len(segmenter.segment(scan_path)) == 5

    monkeypatch.setitem(__import__("sys").modules, "ultralytics", None)  # "not installed"
    with pytest.raises(MissingDependencyError, match=r"hireseg\[yolo\]"):
        Segmenter(weights)


def test_annotations_from_result():
    class Result:
        masks = None

    assert _annotations_from_result(Result()) == []

    class Masks:
        xyn = [  # noqa: RUF012
            np.array([[0.1, 0.1], [0.3, 0.1], [0.3, 0.3]]),
            np.array([[0.5, 0.5], [0.6, 0.6]]),  # degenerate: skipped
            np.array([[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.4]]),
        ]

    class Boxes:
        cls = np.array([2.0, 0.0, 1.0])
        conf = np.array([0.8, 0.9, 1.0000001])

    Result.masks, Result.boxes = Masks, Boxes
    found = _annotations_from_result(Result())
    assert [(a.class_id, a.confidence) for a in found] == [(2, 0.8), (1, 1.0)]
    assert found[0].polygon.area == pytest.approx(0.02)


def test_run_pipeline_outputs(scan_path, fake_model, tmp_path):
    out = tmp_path / "results"
    album = hires.run_pipeline(
        scan_path, fake_model, out, dpi=254, unit="mm", save_crops=True, debug=True, **SETTINGS
    )
    assert album.names == ["scan"] and album.name == "results" and len(album["scan"]) == 5
    assert sorted(p.name for p in out.iterdir()) == [
        "run_config.yaml", "scan.txt", "scan_annotated.tif", "scan_crops", "scan_debug",
        "scan_shapes.csv",
    ]  # fmt: skip

    assert len(hires.read_txt(out / "scan.txt")) == 5
    assert all(a.confidence is not None for a in hires.read_txt(out / "scan.txt"))
    table = pd.read_csv(out / "scan_shapes.csv")
    assert len(table) == 5
    assert table["area"].between(15.0, 16.0).all()  # 40 px = 4 mm at 254 dpi
    assert cv2.imread(str(out / "scan_annotated.tif")).shape == (520, 640, 3)
    assert len(list((out / "scan_crops").iterdir())) == 5
    assert (out / "scan_debug" / "scan_merged.txt").is_file()

    config = yaml.safe_load((out / "run_config.yaml").read_text())
    assert config["n_images"] == 1
    settings = config["settings"]
    assert settings["source"] == [str(scan_path)] and settings["dpi"] == 254
    assert settings["tile_size"] == [256, 256] and settings["model"] is None


def test_run_pipeline_directory_layout(scan_path, fake_model, tmp_path):
    root = scan_path.parent
    (root / "plate2").mkdir()
    cv2.imwrite(str(root / "plate2" / "other.tif"), cv2.imread(str(scan_path)))
    (root / "notes.txt").write_text("not an image")

    flat = hires.run_pipeline(root, fake_model, tmp_path / "flat", save_overlay=False, **SETTINGS)
    assert flat.names == ["scan"]
    assert not (tmp_path / "flat" / "scan_annotated.tif").exists()

    deep = hires.run_pipeline(root, fake_model, tmp_path / "deep", recursive=True, **SETTINGS)
    assert sorted(deep.names) == ["other", "scan"]
    assert (tmp_path / "deep" / "plate2" / "other.txt").is_file()
    assert (tmp_path / "deep" / "scan_shapes.csv").is_file()

    listed = hires.run_pipeline(
        [scan_path, root / "plate2"], fake_model, tmp_path / "listed", **SETTINGS
    )
    assert sorted(listed.names) == ["other", "scan"]
    assert (tmp_path / "listed" / "other.txt").is_file()


def test_run_pipeline_errors(scan_path, fake_model, tmp_path, caplog):
    root = scan_path.parent
    (root / "broken.png").write_text("not an image")
    with pytest.raises(OSError, match="decode"):
        hires.run_pipeline(root, fake_model, tmp_path / "a", **SETTINGS)
    album = hires.run_pipeline(root, fake_model, tmp_path / "b", on_error="skip", **SETTINGS)
    assert album.names == ["scan"] and "failed" in caplog.text
    assert not (tmp_path / "b" / "broken.txt").exists()

    with pytest.raises(ValueError, match="together"):
        hires.run_pipeline(scan_path, fake_model, tmp_path / "c", dpi=300)
    with pytest.raises(ValueError, match="on_error"):
        hires.run_pipeline(scan_path, fake_model, tmp_path / "c", on_error="ignore")
    with pytest.raises(FileNotFoundError):
        hires.run_pipeline(tmp_path / "missing", fake_model, tmp_path / "c")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No images found"):
        hires.run_pipeline(tmp_path / "empty", fake_model, tmp_path / "c")
    assert not (tmp_path / "c").exists()
