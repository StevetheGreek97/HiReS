import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from conftest import square
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import hires
from hires import metrics, plot


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def matches(collection):
    """One tp, one fp and one fn on the ``scan`` image."""
    pred = collection.replace(annotations=[collection[0], square(0.7, 0.2, confidence=0.4)])
    return metrics.match(collection, pred)


def test_lazy_submodule():
    assert hires.plot is plot and "plot" in dir(hires)
    with pytest.raises(AttributeError):
        hires.nonexistent  # noqa: B018


def test_overlay(image_path, collection, tmp_path):
    target = tmp_path / "out" / "overlay.png"
    rgb = plot.overlay(image_path, collection, class_names={0: "pulex"}, save=target)
    assert rgb.shape == (200, 400, 3) and rgb.dtype == np.uint8
    # Class 0 is drawn in the first palette colour, blended 50 % with the grey image.
    expected = (np.array(plot.PALETTE[0]) + 200) / 2
    assert rgb[35, 45] == pytest.approx(expected, abs=1)
    assert (rgb[190, 390] == 200).all()  # untouched background
    saved = cv2.cvtColor(cv2.imread(str(target)), cv2.COLOR_BGR2RGB)
    assert np.array_equal(saved, rgb)


def test_overlay_options(image_path, collection):
    pixels = cv2.imread(str(image_path))
    before = pixels.copy()
    plain = plot.overlay(pixels, collection, fill=False, boxes=False, labels=False)
    assert (plain == 200).all() and np.array_equal(pixels, before)
    assert (plot.overlay(pixels, []) == 200).all()
    assert not (plot.overlay(pixels, collection, fill=False, labels=False) == 200).all()
    opaque = plot.overlay(pixels, collection, boxes=False, labels=False, alpha=1.0)
    assert tuple(opaque[110, 210]) == plot.PALETTE[1]
    with pytest.raises(ValueError, match="shape"):
        plot.overlay(pixels[:, :, 0], collection)
    with pytest.raises(FileNotFoundError):
        plot.overlay("missing.png", collection)


def test_comparison(image_path, matches, tmp_path):
    assert matches.counts() == {"tp": 1, "misclassified": 0, "fp": 1, "fn": 1}
    rgb = plot.comparison(image_path, matches, alpha=1.0, save=tmp_path / "cmp.png")
    assert (tmp_path / "cmp.png").is_file()
    assert tuple(rgb[30, 60]) == plot.STATUS_COLORS["tp"]
    assert tuple(rgb[50, 300]) == plot.STATUS_COLORS["fp"]
    assert tuple(rgb[110, 220]) == plot.STATUS_COLORS["fn"]
    assert (rgb[190, 390] == 200).all()


@pytest.mark.parametrize("status", ["tp", "fp", "fn"])
def test_individual(matches, status, tmp_path):
    figure = plot.individual(
        matches.pick(status), class_names={0: "pulex"}, save=tmp_path / f"{status}.png"
    )
    assert isinstance(figure, Figure) and len(figure.axes) == 4
    assert (tmp_path / f"{status}.png").stat().st_size > 0
    titles = [ax.get_title() for ax in figure.axes]
    assert f"status = {status}" in titles[3]
    assert ("class = -" in titles[0]) == (status == "fp")
    assert ("class = -" in titles[1]) == (status == "fn")
    assert "scan" in figure.get_suptitle()


def test_individual_image_argument(image_path, collection, matches):
    with_array = plot.individual(matches[0], cv2.imread(str(image_path)))
    assert isinstance(with_array, Figure)
    bare = collection.replace(image_path=None)
    orphan = metrics.match(bare, bare)[0]
    assert orphan.image_path is None
    with pytest.raises(ValueError, match="image_path"):
        plot.individual(orphan)
    assert isinstance(plot.individual(orphan, image_path), Figure)


def test_confusion_matrix(collection, matches, tmp_path):
    matrix = metrics.confusion_matrix(collection, collection.replace(annotations=collection[:1]))
    ax = plot.confusion_matrix(matrix, save=tmp_path / "cm.png")
    assert isinstance(ax, Axes) and (tmp_path / "cm.png").is_file()
    assert ax.get_xlabel() == "Ground truth" and ax.get_ylabel() == "Predicted"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "1", "background"]
    shown = ax.images[0].get_array()
    # Displayed array is [prediction, ground truth]; each ground-truth column sums to 1.
    assert shown[0, 0] == 1.0 and shown[2, 1] == 1.0 and shown[:, 2].sum() == 0.0

    _, own = plt.subplots()
    counts = plot.confusion_matrix(matrix, normalize=False, ax=own, title=None)
    assert counts is own and own.get_title() == ""
    assert own.images[0].get_array().sum() == matrix.to_numpy().sum()
    assert {t.get_text() for t in own.texts} == {"0", "1"}


def test_training_curves(tmp_path):
    csv = tmp_path / "results.csv"
    lines = ["   epoch,   train/seg_loss,   train/cls_loss,   val/seg_loss,   val/cls_loss"]
    lines += [f"   {e},   {2 / e},   {1 / e},   {2.5 / e},   {1.5 / e}" for e in range(1, 11)]
    csv.write_text("\n".join(lines) + "\n")

    figure = plot.training_curves(csv, save=tmp_path / "curves.png")
    assert (tmp_path / "curves.png").is_file()
    assert [ax.get_title() for ax in figure.axes] == ["Classification loss", "Segmentation loss"]
    assert all(len(ax.lines) == 2 for ax in figure.axes)

    marked = plot.training_curves(csv, patience=3, losses={"seg_loss": "Seg"})
    (ax,) = marked.axes
    assert ax.get_title() == "Seg" and len(ax.lines) == 3
    assert ax.lines[2].get_xdata()[0] == 7
    with pytest.raises(KeyError, match="box_loss"):
        plot.training_curves(csv, losses={"box_loss": "Box"})


def test_annotation(tmp_path):
    ann = square(0.2, 0.4, size=0.2, confidence=0.75)
    ax = plot.annotation(ann, save=tmp_path / "ann.png")
    assert (tmp_path / "ann.png").is_file()
    assert ax.get_title() == "class 0, confidence 0.75" and ax.get_xlabel() == "x (normalized)"
    assert ax.get_legend() is None and len(ax.lines) == 1
    low, high = ax.get_ylim()
    assert low > high  # y axis points down, like an image

    _, own = plt.subplots()
    pixels = plot.annotation(ann.replace(confidence=None), image_size=(1000, 500),
                             obb=True, bbox=True, ax=own)  # fmt: skip
    assert pixels is own and own.get_title() == "class 0" and own.get_xlabel() == "x (px)"
    assert len(own.lines) == 3 and own.get_legend() is not None
    assert own.get_xlim() == pytest.approx((150, 450))


def test_precision_recall(tmp_path):
    curve = pd.DataFrame(
        {"confidence": [0.9, 0.5], "precision": [1.0, 0.5], "recall": [0.5, 1.0], "f1": [0.6, 0.7]}
    )
    ax = plot.precision_recall(curve, save=tmp_path / "pr.png")
    assert isinstance(ax, Axes) and (tmp_path / "pr.png").is_file()
    assert ax.get_xlabel() == "Recall" and ax.get_ylabel() == "Precision"
    assert ax.get_legend() is None and len(ax.lines) == 2
    assert tuple(ax.lines[1].get_xydata()[0]) == (1.0, 0.5)  # the best-F1 marker

    _, own = plt.subplots()
    both = plot.precision_recall({"pulex": curve, "empty": curve.iloc[:0]}, ax=own, title=None)
    assert both is own and own.get_title() == "" and len(own.lines) == 3
    assert [text.get_text() for text in own.get_legend().get_texts()] == ["pulex", "empty"]
