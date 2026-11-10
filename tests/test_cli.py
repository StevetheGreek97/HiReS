import sys
import types

import cv2
import pytest
from conftest import FakeModel

import hires
from hires.cli import main

SQUARE = "0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3 0.9"


@pytest.fixture
def weights(tmp_path, monkeypatch):
    """A weights file that "loads" as the fake model."""
    fake = types.ModuleType("ultralytics")
    fake.YOLO = lambda path: FakeModel()
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    path = tmp_path / "model.pt"
    path.write_bytes(b"")
    return path


def test_version_and_usage(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"hires {hires.__version__}"
    for argv in ([], ["nonsense"], ["run"]):
        with pytest.raises(SystemExit) as exit_info:
            main(argv)
        assert exit_info.value.code == 2
    for command in ("run", "chunk", "plot", "evaluate"):
        with pytest.raises(SystemExit) as exit_info:
            main([command, "--help"])
        assert exit_info.value.code == 0
    assert "--chunk-size" in capsys.readouterr().out


def test_run(scan_path, weights, tmp_path):
    out = tmp_path / "out"
    code = main([
        "run", "-s", str(scan_path), "-m", str(weights), "-o", str(out),
        "--chunk-size", "256", "256", "--overlap", "96", "--dpi", "254", "--unit", "mm",
        "--save-crops", "--debug",
    ])  # fmt: skip
    assert code == 0
    assert len(hires.read_txt(out / "scan.txt")) == 5
    for name in ("scan_shapes.csv", "scan_annotated.tif", "scan_crops", "scan_debug",
                 "run_config.yaml"):  # fmt: skip
        assert (out / name).exists()


def test_run_reports_failures(scan_path, weights, tmp_path, caplog):
    (scan_path.parent / "broken.png").write_text("not an image")
    args = ["-m", str(weights), "--chunk-size", "256", "256", "--overlap", "96"]
    assert main(["run", "-s", str(scan_path.parent), "-o", str(tmp_path / "out"), *args]) == 1
    assert (tmp_path / "out" / "scan.txt").is_file()

    no_unit = ["run", "-s", str(scan_path), "-o", str(tmp_path / "x"), "--dpi", "300", *args]
    assert main(no_unit) == 2
    assert "--dpi and --unit" in caplog.text
    assert main(["run", "-s", str(tmp_path / "missing"), *args]) == 2
    assert main(["run", "-s", str(scan_path), "-m", str(tmp_path / "missing.pt")]) == 2
    assert main(["run", "-s", str(scan_path), "-o", str(tmp_path / "x"), *args[:2],
                 "--overlap", "5000"]) == 2  # fmt: skip
    assert not (tmp_path / "x").exists()


def test_chunk(scan_path, tmp_path):
    out = tmp_path / "chunks"
    code = main(["chunk", "-s", str(scan_path.parent), "-o", str(out),
                 "--chunk-size", "400", "300", "--overlap", "50"])  # fmt: skip
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == [
        "scan_0_0.png", "scan_0_250.png", "scan_350_0.png", "scan_350_250.png",
    ]  # fmt: skip
    assert cv2.imread(str(out / "scan_0_0.png")).shape == (300, 400, 3)
    (tmp_path / "empty").mkdir()
    assert main(["chunk", "-s", str(tmp_path / "empty"), "-o", str(out)]) == 2
    assert main(["chunk", "-s", str(tmp_path / "missing"), "-o", str(out)]) == 2


def test_plot(scan_path, tmp_path):
    names = tmp_path / "data.yaml"
    names.write_text("names: [pulex]\n")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "scan.txt").write_text(SQUARE + "\n")
    out = tmp_path / "plots"

    base = ["plot", "-s", str(scan_path), "-m", str(names), "-o", str(out)]
    assert main([*base, "--ann", str(labels)]) == 0
    assert cv2.imread(str(out / "scan_annotated.png")).shape == (520, 640, 3)
    (out / "scan_annotated.png").unlink()
    assert main([*base, "--ann", str(labels / "scan.txt")]) == 0
    assert (out / "scan_annotated.png").is_file()


def test_plot_defaults_to_output_dir_and_reports_missing(scan_path, tmp_path, caplog):
    names = tmp_path / "data.yaml"
    names.write_text("names: [pulex]\n")
    out = tmp_path / "results"
    base = ["plot", "-s", str(scan_path), "-m", str(names), "-o", str(out)]
    assert main(base) == 1
    assert "No annotation file for scan.png" in caplog.text
    out.mkdir()
    (out / "scan.txt").write_text(SQUARE + "\n")
    assert main(base) == 0 and (out / "scan_annotated.png").is_file()
    assert main(["plot", "-s", str(scan_path), "-m", str(tmp_path / "nope.yaml")]) == 2


def test_evaluate(scan_path, tmp_path, capsys):
    gt, pred, out = tmp_path / "gt", tmp_path / "pred", tmp_path / "evaluation"
    gt.mkdir()
    pred.mkdir()
    (gt / "scan.txt").write_text(SQUARE.rsplit(" ", 1)[0] + "\n")
    (pred / "scan.txt").write_text(SQUARE + "\n")
    names = tmp_path / "data.yaml"
    names.write_text("names: [pulex]\n")

    base = ["evaluate", "--gt", str(gt), "--pred", str(pred), "-o", str(out)]
    assert main([*base, "--images", str(scan_path.parent), "-n", str(names)]) == 0
    printed = capsys.readouterr().out
    assert "precision" in printed and "pulex" in printed
    assert (out / "detection_report.csv").is_file()
    assert (out / "comparisons" / "scan.png").is_file()

    assert main([*base[:-1], str(tmp_path / "quick"), "--no-images", "--iou", "0.9"]) == 0
    assert not (tmp_path / "quick" / "comparisons").exists()
    (tmp_path / "empty").mkdir()
    assert main(["evaluate", "--gt", str(gt), "--pred", str(tmp_path / "empty")]) == 2
    assert main(["evaluate", "--gt", str(gt), "--pred", str(tmp_path / "missing")]) == 2
    assert main([*base, "--iou", "7"]) == 2


def test_show_versions(capsys):
    hires.show_versions()
    printed = capsys.readouterr().out
    assert hires.__version__ in printed
    assert all(name in printed for name in ("hireseg", "python", "numpy", "shapely", "cv2"))
