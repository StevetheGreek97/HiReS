import cv2
import numpy as np
import pytest

import hires
from hires import AnnotationFormatError, HiresWarning
from hires.io import find_images, parse_line, read_image, read_image_size

SQUARE = "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2"


def test_parse_line():
    ann = parse_line(SQUARE)
    assert ann.class_id == 0 and ann.confidence is None
    assert ann.polygon.area == pytest.approx(0.01)
    assert parse_line(SQUARE + " 0.75").confidence == 0.75


def test_parse_line_repairs_self_intersection():
    bow_tie = "1 0 0 1 1 1 0 0 1"
    ann = parse_line(bow_tie)
    assert ann.polygon.is_valid and ann.polygon.area > 0


@pytest.mark.parametrize(
    "line",
    ["0 0.1 0.1 0.2 0.2", "x 0.1 0.1 0.2 0.1 0.2 0.2", "0 0.1 0.1 0.2 0.2 0.3 0.3", SQUARE + " 7"],
    ids=["too-short", "non-numeric", "zero-area", "bad-confidence"],
)
def test_parse_line_errors(line):
    with pytest.raises(AnnotationFormatError):
        parse_line(line)


def test_read_txt(tmp_path, image_path):
    path = tmp_path / "scan.txt"
    path.write_text(f"{SQUARE} 0.9\n\n1 0.5 0.5 0.7 0.5 0.7 0.6\n")
    c = hires.read_txt(path)
    assert c.name == "scan" and len(c) == 2 and c.image_size is None
    assert [a.confidence for a in c] == [0.9, None]
    with_image = hires.read_txt(path, image=image_path, name="other")
    assert with_image.image_size == (400, 200) and with_image.name == "other"
    assert hires.Collection.from_txt(path, image_size=(10, 20)).image_size == (10, 20)


def test_read_txt_bad_lines(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text(f"{SQUARE}\nthis is not an annotation line at all\n")
    with pytest.warns(HiresWarning, match=r"Skipped 1 unreadable line.*bad\.txt:2"):
        assert len(hires.read_txt(path)) == 1
    with pytest.raises(AnnotationFormatError, match=r"bad\.txt:2"):
        hires.read_txt(path, on_error="raise")
    assert len(hires.read_txt(path, on_error="ignore")) == 1
    with pytest.raises(ValueError):
        hires.read_txt(path, on_error="explode")


def test_read_txt_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        hires.read_txt(tmp_path / "nope.txt")
    (tmp_path / "empty.txt").write_text("")
    assert len(hires.read_txt(tmp_path / "empty.txt")) == 0


def test_read_album_binds_images(tmp_path, image_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "scan.txt").write_text(SQUARE + "\n")
    (labels / "orphan.txt").write_text(SQUARE + "\n")
    album = hires.read_album(labels, image_dir=image_path.parent)
    assert album.names == ["orphan", "scan"]
    assert album["scan"].image_size == (400, 200)
    assert album["orphan"].image_size is None
    from_list = hires.read_album([labels / "scan.txt"], name="mine")
    assert from_list.names == ["scan"] and from_list.name == "mine"
    with pytest.raises(FileNotFoundError):
        hires.read_album(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        hires.read_album(labels, image_dir=tmp_path / "missing")


def test_read_class_names(tmp_path):
    as_list = tmp_path / "data.yaml"
    as_list.write_text("names: [pulex, galeata]\n")
    assert hires.read_class_names(as_list) == {0: "pulex", 1: "galeata"}
    as_dict = tmp_path / "data.yml"
    as_dict.write_text("names:\n  0: pulex\n  3: ballooned\n")
    assert hires.read_class_names(as_dict) == {0: "pulex", 3: "ballooned"}
    empty = tmp_path / "empty.yaml"
    empty.write_text("path: x\n")
    with pytest.raises(KeyError):
        hires.read_class_names(empty)
    with pytest.raises(FileNotFoundError):
        hires.read_class_names(tmp_path / "nope.yaml")


def test_images(tmp_path, image_path):
    assert read_image_size(image_path) == (400, 200)
    assert read_image(image_path).shape == (200, 400, 3)
    gray16 = tmp_path / "gray.tif"
    cv2.imwrite(str(gray16), np.full((30, 50), 40_000, dtype=np.uint16))
    assert read_image(gray16).shape == (30, 50, 3) and read_image(gray16).dtype == np.uint8
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "nope.png")
    (tmp_path / "broken.png").write_text("not an image")
    with pytest.raises(OSError):
        read_image(tmp_path / "broken.png")


def test_read_image_size_restores_pil_limit(image_path):
    from PIL import Image

    before = Image.MAX_IMAGE_PIXELS
    read_image_size(image_path)
    assert before == Image.MAX_IMAGE_PIXELS


def test_find_images(tmp_path, image_path):
    root = image_path.parent
    (root / "notes.txt").write_text("x")
    (root / "sub").mkdir()
    cv2.imwrite(str(root / "sub" / "deep.TIF"), np.zeros((4, 4, 3), np.uint8))
    assert find_images(root) == [image_path]
    assert [p.name for p in find_images(root, recursive=True)] == ["scan.png", "deep.TIF"]
    assert find_images(image_path) == [image_path]
    assert find_images(root / "notes.txt") == []
    with pytest.raises(FileNotFoundError):
        find_images(tmp_path / "missing")
