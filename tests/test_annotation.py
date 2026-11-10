import math
import pickle

import pytest
from shapely.geometry import LineString, Point, Polygon, box

import hires
from hires import Annotation, measure


def test_measure_rectangle():
    traits = measure(box(0, 0, 4, 2))
    assert traits["area"] == 8
    assert traits["perimeter"] == 12
    assert traits["solidity"] == 1
    assert traits["convexity"] == 1
    assert traits["circularity"] == pytest.approx(4 * math.pi * 8 / 144)
    assert (traits["bbox_width"], traits["bbox_height"]) == (4, 2)
    assert (traits["obb_width"], traits["obb_length"]) == (2, 4)
    assert tuple(traits) == hires.TRAIT_NAMES


def test_measure_circle_is_circular():
    assert measure(Point(0, 0).buffer(1, 256))["circularity"] == pytest.approx(1, abs=1e-3)


def test_measure_rotated_rectangle_obb():
    from shapely import affinity

    rotated = affinity.rotate(box(0, 0, 10, 2), 30)
    traits = measure(rotated)
    assert traits["obb_width"] == pytest.approx(2)
    assert traits["obb_length"] == pytest.approx(10)
    assert traits["bbox_width"] > 2


def test_measure_concave_shape():
    l_shape = Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    traits = measure(l_shape)
    assert traits["solidity"] == pytest.approx(3 / 3.5)
    assert traits["convexity"] < 1


def test_measure_pixel_size_scales_lengths_and_areas():
    base = measure(box(0, 0, 4, 2))
    scaled = measure(box(0, 0, 4, 2), pixel_size=0.5)
    assert scaled["area"] == base["area"] * 0.25
    assert scaled["perimeter"] == base["perimeter"] * 0.5
    assert scaled["obb_length"] == base["obb_length"] * 0.5
    assert scaled["circularity"] == base["circularity"]


def test_traits_use_pixel_space_for_non_square_images():
    # 0.1 x 0.1 normalized is 400 x 100 px on a 4000 x 1000 image: not a square.
    ann = Annotation(0, box(0.1, 0.1, 0.2, 0.2))
    traits = ann.traits(image_size=(4000, 1000))
    assert traits["area"] == pytest.approx(40_000)
    assert traits["circularity"] == pytest.approx(4 * math.pi * 40_000 / 1000**2)
    assert (traits["obb_width"], traits["obb_length"]) == pytest.approx((100, 400))


def test_annotation_is_immutable_and_hashable():
    ann = Annotation(0, box(0, 0, 1, 1), 0.5)
    with pytest.raises(AttributeError):
        ann.class_id = 2
    assert ann == Annotation(0, box(0, 0, 1, 1), 0.5)
    assert len({ann, ann.replace()}) == 1
    assert pickle.loads(pickle.dumps(ann)) == ann


def test_replace_returns_new_object():
    ann = Annotation(0, box(0, 0, 1, 1), 0.5)
    assert ann.replace(class_id=3).class_id == 3
    assert ann.class_id == 0


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"class_id": 0.5, "polygon": box(0, 0, 1, 1)}, TypeError),
        ({"class_id": 0, "polygon": LineString([(0, 0), (1, 1)])}, TypeError),
        ({"class_id": 0, "polygon": Polygon()}, ValueError),
        ({"class_id": 0, "polygon": box(0, 0, 1, 1), "confidence": 1.5}, ValueError),
    ],
)
def test_annotation_validation(kwargs, error):
    with pytest.raises(error):
        Annotation(**kwargs)


def test_numpy_integer_class_id_is_accepted():
    import numpy as np

    ann = Annotation(np.int64(2), box(0, 0, 1, 1))
    assert ann.class_id == 2 and type(ann.class_id) is int


def test_within_frame():
    assert Annotation(0, box(0.2, 0.2, 0.4, 0.4)).within_frame(0.01)
    assert not Annotation(0, box(0.005, 0.2, 0.4, 0.4)).within_frame(0.01)
    assert not Annotation(0, box(0.2, 0.2, 0.4, 1.2)).within_frame()


def test_to_pixels():
    assert Annotation(0, box(0.1, 0.1, 0.2, 0.3)).to_pixels((1000, 100)).bounds == (
        100,
        10,
        200,
        30,
    )
