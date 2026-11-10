import cv2
import numpy as np
import pytest
from conftest import square

import hires
from hires import Collection, HiresWarning


def test_sequence_protocol(collection):
    assert len(collection) == 2
    assert collection[0].class_id == 0
    assert [a.class_id for a in collection] == [0, 1]
    assert collection[0] in collection
    sliced = collection[1:]
    assert isinstance(sliced, Collection)
    assert len(sliced) == 1 and sliced.name == "scan" and sliced.image_size == (400, 200)


def test_is_immutable(collection):
    with pytest.raises(AttributeError):
        collection.name = "other"
    with pytest.raises(TypeError):
        collection[0] = square(0, 0)


def test_rejects_non_annotations():
    with pytest.raises(TypeError):
        Collection(["not an annotation"])


def test_equality_and_add(collection):
    assert collection == collection.replace()
    assert collection != collection.replace(name="x")
    assert len(collection + [square(0.7, 0.1)]) == 3  # noqa: RUF005 (tests __add__)
    assert len(collection) == 2


def test_filter_keeps_metadata(collection):
    kept = collection.filter(lambda a: a.confidence > 0.8)
    assert len(kept) == 1
    assert kept.name == "scan"
    assert kept.image_path == collection.image_path
    assert kept.image_size == (400, 200)


def test_class_selection(collection):
    assert collection.class_counts() == {0: 1, 1: 1}
    assert collection.keep_classes(1).class_counts() == {1: 1}
    assert collection.drop_classes([1]).class_counts() == {0: 1}
    assert collection.remap_classes({1: 5}).class_counts() == {0: 1, 5: 1}


def test_nms_removes_duplicates_and_keeps_most_confident():
    c = Collection(
        [
            square(0.1, 0.1, confidence=0.6),
            square(0.105, 0.1, confidence=0.9),  # near duplicate, more confident
            square(0.5, 0.5, confidence=0.3),
        ]
    )
    kept = c.nms(0.5)
    assert [a.confidence for a in kept] == [0.9, 0.3]


def test_nms_class_aware():
    c = Collection(
        [square(0.1, 0.1, class_id=0, confidence=0.9), square(0.1, 0.1, class_id=1, confidence=0.8)]
    )
    assert len(c.nms(0.5)) == 1
    assert len(c.nms(0.5, class_aware=True)) == 2


def test_nms_requires_confidence():
    with pytest.raises(ValueError, match="confidence"):
        Collection([square(0, 0, confidence=None)]).nms()
    assert len(Collection().nms()) == 0


def test_to_frame_measures_in_pixels(collection):
    frame = collection.to_frame()
    assert list(frame.columns[:8]) == [
        "name", "image_path", "image_width", "image_height",
        "dpi", "unit", "class_id", "confidence",
    ]  # fmt: skip
    assert list(frame.columns[8:]) == list(hires.TRAIT_NAMES)
    # 0.1 x 0.1 normalized on 400 x 200 px
    assert frame.loc[0, "area"] == pytest.approx(40 * 20)
    assert frame.loc[0, "bbox_width"] == pytest.approx(40)
    assert frame.loc[0, "obb_length"] == pytest.approx(40)


def test_to_frame_without_image_size_warns():
    c = Collection([square(0.1, 0.1)], name="x")
    with pytest.warns(HiresWarning, match="image_size"):
        frame = c.to_frame()
    assert frame.loc[0, "area"] == pytest.approx(0.01)


def test_empty_to_frame_has_columns():
    frame = Collection(name="x").to_frame()
    assert frame.empty and "area" in frame.columns


def test_with_scale(collection):
    px = collection.to_frame()
    mm = collection.with_scale(dpi=254, unit="mm").to_frame()  # 0.1 mm per pixel
    assert mm["area"].tolist() == pytest.approx((px["area"] * 0.01).tolist())
    assert mm["perimeter"].tolist() == pytest.approx((px["perimeter"] * 0.1).tolist())
    assert mm["solidity"].tolist() == px["solidity"].tolist()
    assert mm["unit"].unique().tolist() == ["mm"]
    assert collection.unit is None  # original untouched


def test_with_scale_validation(collection):
    with pytest.raises(ValueError, match="Unsupported unit"):
        collection.with_scale(300, "parsec")
    with pytest.raises(ValueError, match="together"):
        collection.with_scale(300, None)
    with pytest.raises(ValueError, match="image_size"):
        Collection([square(0, 0)]).with_scale(300, "mm")
    assert collection.with_scale(300, "mm").with_scale(None, None).pixel_size == 1.0


def test_with_image(image_path):
    c = Collection([square(0.1, 0.1)], name="scan").with_image(image_path)
    assert c.image_size == (400, 200)
    with pytest.raises(FileNotFoundError):
        c.with_image(image_path.with_name("missing.png"))


def test_txt_round_trip(collection, tmp_path):
    path = tmp_path / "out" / "scan.txt"
    collection.to_txt(path)
    loaded = hires.read_txt(path, image=collection.image_path)
    assert loaded.class_counts() == collection.class_counts()
    assert [a.confidence for a in loaded] == [0.9, 0.6]
    for original, restored in zip(collection, loaded, strict=True):
        assert restored.polygon.equals_exact(original.polygon.normalize(), 1e-6) or (
            restored.polygon.symmetric_difference(original.polygon).area < 1e-9
        )
    collection.to_txt(path, confidence=False)
    assert [a.confidence for a in hires.read_txt(path)] == [None, None]


def test_to_csv(collection, tmp_path):
    import pandas as pd

    path = tmp_path / "nested" / "shapes.csv"
    collection.to_csv(path)
    assert len(pd.read_csv(path)) == 2


def test_save_crops(collection, tmp_path):
    paths = collection.save_crops(tmp_path / "crops")
    assert [p.name for p in paths] == ["scan_0000_class_0.png", "scan_0001_class_1.png"]
    crop = cv2.imread(str(paths[0]))
    assert crop.shape[:2] == (20, 40)
    padded = collection.save_crops(tmp_path / "padded", padding=5, mask=False, prefix="p")
    assert cv2.imread(str(padded[0])).shape[:2] == (30, 50)
    assert padded[0].name.startswith("p_0000")


def test_save_crops_masks_outside_polygon(image_path, tmp_path):
    from shapely.geometry import Polygon

    triangle = hires.Annotation(0, Polygon([(0.1, 0.1), (0.6, 0.1), (0.1, 0.6)]))
    c = Collection([triangle], name="t", image_path=image_path, image_size=(400, 200))
    crop = cv2.imread(str(c.save_crops(tmp_path)[0]))
    assert crop[2, 2].tolist() == [200, 200, 200]  # inside
    assert crop[-2, -2].tolist() == [0, 0, 0]  # outside the hypotenuse
    assert np.count_nonzero(crop) > 0


def test_save_crops_needs_image():
    with pytest.raises(ValueError, match="image_path"):
        Collection([square(0, 0)]).save_crops("unused")


def test_filter_confidence(collection):
    assert [a.confidence for a in collection.filter_confidence(0.7)] == [0.9]
    assert collection.filter_confidence(0.0) == collection
    assert collection.filter_confidence(0.7).image_size == (400, 200)
    unscored = collection.replace(annotations=[a.replace(confidence=None) for a in collection])
    assert len(unscored.filter_confidence(0.99)) == 2
    with pytest.raises(ValueError, match="minimum"):
        collection.filter_confidence(1.5)
    assert len(hires.Album([collection]).filter_confidence(0.7)["scan"]) == 1


def test_filter_area():
    # 0.1 and 0.2 squares in a 400 x 200 px image: 800 and 3200 square pixels.
    sized = hires.Collection(
        [square(0.1, 0.1, 0.1), square(0.5, 0.5, 0.2)], name="scan", image_size=(400, 200)
    )
    assert sized.filter_area() == sized
    assert list(sized.filter_area(min_area=1000)) == [sized[1]]
    assert list(sized.filter_area(max_area=1000)) == [sized[0]]
    assert len(sized.filter_area(799, 3201)) == 2 and len(sized.filter_area(900, 3000)) == 0
    # 254 dpi in mm: one pixel is 0.1 mm, so the squares are 8 and 32 square mm.
    assert list(sized.with_scale(254, "mm").filter_area(max_area=10)) == [sized[0]]
    assert len(sized.replace(image_size=None).filter_area(max_area=0.02)) == 1
    with pytest.raises(ValueError, match="min_area"):
        sized.filter_area(10, 5)
    assert len(hires.Album([sized]).filter_area(min_area=1000)["scan"]) == 1
