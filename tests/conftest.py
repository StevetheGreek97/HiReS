from __future__ import annotations

import cv2
import matplotlib
import numpy as np
import pytest
from shapely.geometry import box

import hires

matplotlib.use("Agg")


def square(x, y, size=0.1, class_id=0, confidence=0.9):
    """Square annotation with its top-left corner at (x, y)."""
    return hires.Annotation(class_id, box(x, y, x + size, y + size), confidence)


@pytest.fixture
def image_path(tmp_path):
    """A 400 x 200 px colour image on disk."""
    path = tmp_path / "images" / "scan.png"
    path.parent.mkdir()
    pixels = np.full((200, 400, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(path), pixels)
    return path


@pytest.fixture
def collection(image_path):
    return hires.Collection(
        [
            square(0.1, 0.1, class_id=0, confidence=0.9),
            square(0.5, 0.5, class_id=1, confidence=0.6),
        ],
        name="scan",
        image_path=image_path,
        image_size=(400, 200),
    )


@pytest.fixture(autouse=True)
def _doctest_namespace(doctest_namespace):
    doctest_namespace["hires"] = hires


class _Boxes:
    def __init__(self, classes, confidences):
        self.cls = np.array(classes, dtype=float)
        self.conf = np.array(confidences, dtype=float)


class _Masks:
    def __init__(self, polygons):
        self.xyn = polygons


class _Result:
    def __init__(self, polygons, classes, confidences):
        self.masks = _Masks(polygons) if polygons else None
        self.boxes = _Boxes(classes, confidences)


class FakeModel:
    """Stand-in for an Ultralytics model: "detects" every bright blob in a tile.

    Blobs brighter than 200 are class 0, dimmer ones class 1. The confidence
    is the fraction of the blob's bounding box that it fills.
    """

    names = {0: "bright", 1: "dim"}  # noqa: RUF012

    def __init__(self):
        self.calls = []

    def predict(self, batch, *, conf, imgsz, device, verbose):
        self.calls.append({"n": len(batch), "conf": conf, "imgsz": imgsz, "device": device})
        results = []
        for tile in batch:
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            contours, _ = cv2.findContours(
                (gray > 50).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            polygons, classes, confidences = [], [], []
            for contour in contours:
                points = contour.reshape(-1, 2).astype(float)
                if len(points) < 3:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                polygons.append(points / (width, height))
                classes.append(0 if gray[y : y + h, x : x + w].max() > 200 else 1)
                confidences.append(min(1.0, cv2.contourArea(contour) / (w * h)))
            results.append(_Result(polygons, classes, confidences))
        return results


@pytest.fixture
def fake_model():
    return FakeModel()


#: Top-left corners of the 40 px blobs in the ``scan`` fixture.
BLOBS = ((60, 60), (230, 100), (300, 300), (500, 220), (120, 400))


@pytest.fixture
def scan_path(tmp_path):
    """A 640 x 520 px black scan with five 40 px blobs; the last one is dim."""
    pixels = np.zeros((520, 640, 3), dtype=np.uint8)
    for number, (x, y) in enumerate(BLOBS):
        pixels[y : y + 40, x : x + 40] = 120 if number == len(BLOBS) - 1 else 255
    path = tmp_path / "scans" / "scan.png"
    path.parent.mkdir()
    cv2.imwrite(str(path), pixels)
    return path
