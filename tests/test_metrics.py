import numpy as np
import pandas as pd
import pytest
from conftest import square
from shapely.geometry import box

import hires
from hires import metrics
from hires.metrics import Match, MatchResult


def shifted(x, y, dx, **kwargs):
    """A 0.1 square at (x, y) moved right by ``dx`` of its width."""
    return square(x + 0.1 * dx, y, **kwargs)


@pytest.fixture
def gt():
    return hires.Collection(
        [square(0.1 + 0.2 * k, 0.1, class_id=k // 2, confidence=None) for k in range(4)],
        name="a",
        image_size=(1000, 500),
    )


@pytest.fixture
def pred():
    """tp (IoU 1), tp (IoU 0.6), misclassified (IoU 1), one fp; gt[3] is missed."""
    return hires.Collection(
        [square(0.1, 0.1, class_id=0, confidence=0.9),
         shifted(0.3, 0.1, 0.25, class_id=0, confidence=0.7),
         square(0.5, 0.1, class_id=0, confidence=0.8),
         square(0.1, 0.7, class_id=1, confidence=0.6)],
        name="a",
        image_size=(1000, 500),
    )  # fmt: skip


def test_iou():
    assert metrics.iou(box(0, 0, 2, 1), box(1, 0, 3, 1)) == pytest.approx(1 / 3)
    assert metrics.iou(box(0, 0, 1, 1), box(2, 2, 3, 3)) == 0.0
    assert metrics.iou(box(0, 0, 1, 1), box(1, 0, 2, 1)) == 0.0  # shared edge only
    a, b = square(0.1, 0.1), shifted(0.1, 0.1, 0.5)
    assert metrics.iou(a, b) == a.iou(b) == pytest.approx(1 / 3)


def test_iou_matrix(gt, pred):
    matrix = metrics.iou_matrix(gt, pred)
    assert matrix.shape == (4, 4)
    expected = np.array([[metrics.iou(g, p) for p in pred] for g in gt])
    assert matrix == pytest.approx(expected)
    assert matrix[1, 1] == pytest.approx(0.6)
    assert metrics.iou_matrix([], pred).shape == (0, 4)
    assert metrics.iou_matrix(gt, []).shape == (4, 0)
    assert not metrics.iou_matrix([box(0, 0, 1, 1)], [box(5, 5, 6, 6)]).any()


def test_match_statuses(gt, pred):
    result = metrics.match(gt, pred)
    assert result.counts() == {"tp": 2, "misclassified": 1, "fp": 1, "fn": 1}
    assert len(result) == 5 and result.iou_threshold == 0.5 and not result.class_aware
    assert repr(result) == "MatchResult(iou_threshold=0.5, tp=2, misclassified=1, fp=1, fn=1)"
    by_gt = {m.gt_index: m for m in result if m.gt is not None}
    assert [by_gt[i].status for i in range(4)] == ["tp", "tp", "misclassified", "fn"]
    assert [by_gt[i].pred_index for i in range(4)] == [0, 1, 2, None]
    assert by_gt[1].iou == pytest.approx(0.6) and by_gt[3].iou == 0.0
    assert by_gt[0].is_matched and by_gt[0].class_match
    assert by_gt[2].is_matched and not by_gt[2].class_match
    assert not by_gt[3].is_matched
    (fp,) = result.with_status("fp")
    assert fp.gt is None and fp.pred is pred[3] and fp.pred_index == 3 and fp.image == "a"


def test_match_threshold_and_class_awareness(gt, pred):
    assert metrics.match(gt, pred, iou_threshold=0.7).counts() == {
        "tp": 1, "misclassified": 1, "fp": 2, "fn": 2,
    }  # fmt: skip
    aware = metrics.match(gt, pred, class_aware=True)
    assert aware.class_aware
    assert aware.counts() == {"tp": 2, "misclassified": 0, "fp": 2, "fn": 2}
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="iou_threshold"):
            metrics.match(gt, pred, iou_threshold=bad)


def test_match_is_one_to_one_and_prefers_higher_iou():
    gt = hires.Collection([square(0.1, 0.1)])
    pred = hires.Collection(
        [shifted(0.1, 0.1, 0.2, confidence=0.99), shifted(0.1, 0.1, 0.05, confidence=0.3)]
    )
    result = metrics.match(gt, pred)
    assert result.counts() == {"tp": 1, "misclassified": 0, "fp": 1, "fn": 0}
    assert result.find(gt_index=0).pred_index == 1  # the closer one, not the more confident


def test_match_zero_threshold_never_pairs_disjoint_polygons():
    gt = hires.Collection([square(0.1, 0.1)])
    pred = hires.Collection([square(0.6, 0.6, confidence=0.5)])
    assert metrics.match(gt, pred, iou_threshold=0.0).counts()["tp"] == 0


def test_match_empty_inputs(gt):
    empty = hires.Collection([], name="a")
    assert metrics.match(gt, empty).counts() == {"tp": 0, "misclassified": 0, "fp": 0, "fn": 4}
    assert metrics.match(empty, gt).counts() == {"tp": 0, "misclassified": 0, "fp": 4, "fn": 0}
    assert len(metrics.match(empty, empty)) == 0


def test_match_albums(gt, pred):
    gt_album = hires.Album([gt, gt.replace(name="b"), gt.replace(name="only_gt")])
    pred_album = hires.Album([pred.replace(name="b"), pred, pred.replace(name="only_pred")])
    result = metrics.match(gt_album, pred_album)
    assert result.counts() == {"tp": 4, "misclassified": 2, "fp": 2, "fn": 2}
    assert result.images == ["a", "b"]
    assert result.unpaired_gt == ("only_gt",) and result.unpaired_pred == ("only_pred",)
    assert result.for_image("b").counts() == metrics.match(gt, pred).counts()
    assert result[:3].unpaired_gt == ("only_gt",)
    with pytest.raises(ValueError, match="only_gt"):
        metrics.match(gt_album, pred_album, strict=True)
    with pytest.raises(TypeError, match="Collection and Album"):
        metrics.match(gt, pred_album)


def test_match_result_is_a_sequence(gt, pred):
    result = metrics.match(gt, pred)
    assert isinstance(result[0], Match) and isinstance(result[1:3], MatchResult)
    assert len(result[1:3]) == 2 and result[1:3].iou_threshold == 0.5
    assert list(result) == [result[i] for i in range(len(result))]
    assert len(result.filter(lambda m: m.iou > 0.9)) == 2
    errors = result.with_status("fp", "fn")
    assert errors.counts() == {"tp": 0, "misclassified": 0, "fp": 1, "fn": 1}
    with pytest.raises(ValueError, match="Unknown status"):
        result.with_status("tn")


def test_find(gt, pred):
    result = metrics.match(gt, pred)
    assert result.find(pred_index=2).status == "misclassified"
    assert result.find(gt_index=3).status == "fn"
    assert result.find(pred_index=3, image="a").status == "fp"
    for kwargs in ({}, {"gt_index": 0, "pred_index": 0}):
        with pytest.raises(ValueError, match="exactly one"):
            result.find(**kwargs)
    with pytest.raises(LookupError, match="pred_index=9"):
        result.find(pred_index=9)
    with pytest.raises(LookupError, match="No image named"):
        result.find(pred_index=0, image="zzz")

    both = metrics.match(hires.Album([gt, gt.replace(name="b")]),
                         hires.Album([pred, pred.replace(name="b")]))  # fmt: skip
    with pytest.raises(ValueError, match="several images"):
        both.find(pred_index=0)
    assert both.find(pred_index=0, image="b").image == "b"


def test_find_in_image_without_matches(gt):
    album = hires.Album([gt, hires.Collection([], name="blank")])
    result = metrics.match(album, album.map(lambda c: c.replace(annotations=[])))
    with pytest.raises(LookupError, match="gt_index=0"):
        result.find(gt_index=0, image="blank")


def test_pick(gt, pred):
    result = metrics.match(gt, pred)
    assert result.pick("tp", by="best_iou").iou == pytest.approx(1.0)
    assert result.pick("tp", by="worst_iou").iou == pytest.approx(0.6)
    assert result.pick("tp").status == "tp"
    assert result.pick("fn", image="a").gt_index == 3
    assert result.pick("tp", by="random", seed=1) is result.pick("tp", by="random", seed=1)
    with pytest.raises(ValueError, match="Unknown status"):
        result.pick("tn")
    with pytest.raises(ValueError, match="by must be"):
        result.pick("tp", by="biggest")
    with pytest.raises(LookupError, match="No match with status 'fn'"):
        result.with_status("tp").pick("fn")


def test_pick_median():
    gt = hires.Collection([square(0.1 + 0.2 * k, 0.1) for k in range(3)])
    pred = hires.Collection(
        [shifted(0.1 + 0.2 * k, 0.1, dx, confidence=0.9) for k, dx in enumerate((0.0, 0.1, 0.2))]
    )
    assert metrics.match(gt, pred).pick("tp").gt_index == 1


def test_to_frame(gt, pred):
    result = metrics.match(gt, pred)
    frame = result.to_frame(traits=False)
    assert list(frame.columns) == [
        "image", "status", "iou", "class_match", "gt_index", "pred_index",
        "gt_class", "pred_class", "gt_confidence", "pred_confidence",
    ]  # fmt: skip
    assert len(frame) == len(result)
    for position, m in enumerate(result):
        assert frame.loc[position, "status"] == m.status
    assert frame["class_match"].dtype == bool and frame["gt_index"].dtype == "Int64"
    fn = frame[frame["status"] == "fn"].iloc[0]
    assert pd.isna(fn["pred_class"]) and pd.isna(fn["pred_confidence"]) and fn["gt_class"] == 1
    assert frame["gt_confidence"].isna().all()

    full = result.to_frame(class_names={0: "pulex"})
    assert {f"gt_{t}" for t in hires.TRAIT_NAMES} <= set(full.columns)
    assert {f"pred_{t}" for t in hires.TRAIT_NAMES} <= set(full.columns)
    assert list(full.columns[6:10]) == [
        "gt_class", "gt_class_name", "pred_class", "pred_class_name",
    ]  # fmt: skip
    tp = full[full["status"] == "tp"].iloc[0]
    assert tp["gt_area"] == pytest.approx(100 * 50)  # pixels of the 1000 x 500 image
    assert tp["gt_class_name"] == "pulex"
    assert full.loc[full["status"] == "fn", "gt_class_name"].item() == "1"
    assert full.loc[full["status"] == "fn", "pred_class_name"].item() is None
    assert full.loc[full["status"] == "fp", "gt_area"].isna().all()


def test_to_frame_uses_physical_scale(gt, pred):
    scaled = metrics.match(gt.with_scale(254, "mm"), pred).to_frame()
    tp = scaled[scaled["status"] == "tp"].iloc[0]
    assert tp["gt_area"] == pytest.approx(100 * 50 * 0.1**2)
    assert tp["pred_area"] == pytest.approx(100 * 50)


def test_to_frame_empty():
    empty = hires.Collection([])
    frame = metrics.match(empty, empty).to_frame(class_names={0: "x"})
    assert len(frame) == 0 and "gt_class_name" in frame.columns and "pred_area" in frame.columns


def test_confusion_matrix(gt, pred):
    matrix = metrics.confusion_matrix(gt, pred)
    assert matrix.index.name == "gt" and matrix.columns.name == "pred"
    assert list(matrix.index) == list(matrix.columns) == [0, 1, "background"]
    assert matrix.to_numpy().tolist() == [[2, 0, 0], [1, 0, 1], [0, 1, 0]]
    named = metrics.confusion_matrix(gt, pred, class_names={0: "a", 1: "b", 2: "c"}, background="-")
    assert list(named.index) == ["a", "b", "c", "-"]
    assert named.loc["b", "a"] == 1 and named.loc["b", "-"] == 1 and named.loc["-", "b"] == 1
    assert named.loc["c"].sum() == 0 and named.to_numpy().sum() == 5


def test_average_precision_by_hand():
    # Two objects; predictions ranked by confidence are hit, miss, hit.
    gt = hires.Collection([square(0.1, 0.1), square(0.5, 0.1)])
    pred = hires.Collection(
        [square(0.1, 0.1, confidence=0.9), square(0.1, 0.6, confidence=0.8),
         square(0.5, 0.1, confidence=0.7)]
    )  # fmt: skip
    ap = metrics.average_precision(gt, pred)
    assert ap.name == "ap@0.5" and ap.index.name == "class_id"
    assert ap[0] == pytest.approx(0.5 * 1.0 + 0.5 * 2 / 3)
    assert metrics.average_precision(gt, gt.replace(
        annotations=[a.replace(confidence=0.5) for a in gt]))[0] == pytest.approx(1.0)  # fmt: skip


def test_average_precision_edge_cases(gt, pred):
    ap = metrics.average_precision(gt, pred)
    # Class 1: its only prediction is a false positive.
    assert ap[1] == 0.0
    # Class 0: three predictions (hit 0.9, miss 0.8, hit 0.7) for two objects.
    assert ap[0] == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert metrics.average_precision(gt, pred, iou_threshold=0.7)[0] == pytest.approx(0.5)
    assert metrics.average_precision(gt, hires.Collection([])).tolist() == [0.0, 0.0]
    # Predictions without confidence rank last instead of failing.
    unscored = pred.replace(annotations=[a.replace(confidence=None) for a in pred])
    assert 0.0 < metrics.average_precision(gt, unscored)[0] <= 1.0


def test_detection_report(gt, pred):
    report = metrics.detection_report(gt, pred)
    assert list(report.index) == ["all", 0, 1] and report.index.name == "class_id"
    assert list(report.columns) == [
        "tp", "fp", "fn", "precision", "recall", "f1", "ap50", "ap50_95",
    ]  # fmt: skip
    assert report.loc[0, ["tp", "fp", "fn"]].tolist() == [2, 1, 0]
    assert report.loc[1, ["tp", "fp", "fn"]].tolist() == [0, 1, 2]
    assert report.loc["all", ["tp", "fp", "fn"]].tolist() == [2, 2, 2]
    assert report["tp"].dtype.kind == "i"
    assert report.loc[0, "precision"] == pytest.approx(2 / 3) and report.loc[0, "recall"] == 1.0
    assert report.loc[0, "f1"] == pytest.approx(0.8)
    assert report.loc[1, ["precision", "recall", "f1"]].tolist() == [0.0, 0.0, 0.0]
    assert report.loc["all", "precision"] == report.loc["all", "recall"] == 0.5
    ap = metrics.average_precision(gt, pred)
    assert report.loc[0, "ap50"] == pytest.approx(ap[0])
    assert report.loc["all", "ap50"] == pytest.approx(ap.mean())
    # One of the two class-0 hits has IoU 0.6, so it drops out above that threshold.
    by_threshold = [metrics.average_precision(gt, pred, iou_threshold=t)[0]
                    for t in metrics.COCO_THRESHOLDS]  # fmt: skip
    assert report.loc[0, "ap50_95"] == pytest.approx(np.mean(by_threshold))
    assert report.loc[0, "ap50_95"] < report.loc[0, "ap50"]


def test_detection_report_options(gt, pred):
    strict = metrics.detection_report(gt, pred, iou_threshold=0.7)
    assert strict.loc[0, ["tp", "fp", "fn"]].tolist() == [1, 2, 1]
    assert strict.loc[0, "ap50"] == metrics.detection_report(gt, pred).loc[0, "ap50"]
    named = metrics.detection_report(gt, pred, class_names={0: "pulex", 5: "rare"})
    assert named.columns[0] == "class_name"
    assert named["class_name"].tolist() == ["all", "pulex", "1", "rare"]
    assert named.loc[5, ["tp", "fp", "fn", "f1", "ap50"]].tolist() == [0, 0, 0, 0.0, 0.0]
    single = metrics.detection_report(gt, pred, ap_thresholds=[0.5])
    assert single["ap50_95"].tolist() == single["ap50"].tolist()


def test_detection_report_perfect_and_empty(gt):
    perfect = metrics.detection_report(
        gt, gt.replace(annotations=[a.replace(confidence=0.9) for a in gt])
    )
    assert (perfect[["precision", "recall", "f1", "ap50", "ap50_95"]] == 1.0).all().all()
    empty = hires.Collection([])
    report = metrics.detection_report(empty, empty)
    assert list(report.index) == ["all"] and report.loc["all", "tp"] == 0
    assert report.loc["all", "f1"] == 0.0


def test_metrics_accept_albums(gt, pred):
    gt_album = hires.Album([gt, gt.replace(name="b")])
    pred_album = hires.Album([pred, pred.replace(name="b")])
    assert metrics.confusion_matrix(gt_album, pred_album).to_numpy().sum() == 10
    report = metrics.detection_report(gt_album, pred_album)
    assert report.loc["all", ["tp", "fp", "fn"]].tolist() == [4, 4, 4]
    assert report.loc[0, "ap50"] == pytest.approx(metrics.average_precision(gt, pred)[0])


def test_summary(gt, pred):
    summary = metrics.match(gt, pred).summary()
    assert summary["n_gt"] == 4 and summary["n_pred"] == 4
    assert [summary[s] for s in metrics.STATUSES] == [2, 1, 1, 1]
    assert summary["precision"] == summary["recall"] == summary["f1"] == 0.5
    assert summary["mean_iou"] == pytest.approx((1 + 0.6 + 1) / 3)
    report = metrics.detection_report(gt, pred)
    assert summary["precision"] == report.loc["all", "precision"]
    assert summary["recall"] == report.loc["all", "recall"]
    empty = metrics.match(hires.Collection([]), hires.Collection([])).summary()
    assert empty["precision"] == empty["recall"] == empty["f1"] == 0.0
    assert np.isnan(empty["mean_iou"])


def test_precision_recall_curve(gt, pred):
    # Ranked by confidence: hit (0.9), wrong class (0.8), hit (0.7), false positive (0.6).
    curve = metrics.precision_recall_curve(gt, pred)
    assert list(curve.columns) == ["confidence", "precision", "recall", "f1"]
    assert curve["confidence"].tolist() == [0.9, 0.8, 0.7, 0.6]
    assert curve["precision"].tolist() == pytest.approx([1, 1 / 2, 2 / 3, 1 / 2])
    assert curve["recall"].tolist() == pytest.approx([0.25, 0.25, 0.5, 0.5])
    assert curve["f1"].tolist() == pytest.approx([0.4, 1 / 3, 4 / 7, 0.5])
    assert curve.loc[curve["f1"].idxmax(), "confidence"] == 0.7

    class_0 = metrics.precision_recall_curve(gt, pred, class_id=0)
    assert class_0["recall"].tolist() == pytest.approx([0.5, 0.5, 1.0])
    class_1 = metrics.precision_recall_curve(gt, pred, class_id=1)
    assert class_1["confidence"].tolist() == [0.6]
    assert class_1["precision"].tolist() == class_1["recall"].tolist() == [0.0]
    assert len(metrics.precision_recall_curve(gt, hires.Collection([]))) == 0
    assert metrics.precision_recall_curve(hires.Collection([]), pred)["recall"].eq(0).all()


def test_precision_recall_curve_merges_equal_confidences():
    gt = hires.Collection([square(0.1, 0.1, confidence=None)])
    pred = hires.Collection([square(0.1, 0.1, confidence=0.5), square(0.6, 0.6, confidence=0.5)])
    curve = metrics.precision_recall_curve(gt, pred)
    assert len(curve) == 1
    assert curve.loc[0, ["precision", "recall"]].tolist() == [0.5, 1.0]


def test_evaluate(collection, tmp_path, caplog):
    names = {0: "pulex", 1: "galeata"}
    pred = collection.replace(annotations=[collection[0], square(0.7, 0.2, confidence=0.4)])
    gt_album = hires.Album([collection, collection.replace(name="extra")])
    out = tmp_path / "evaluation"
    report = metrics.evaluate(gt_album, hires.Album([pred]), out, class_names=names)
    pd.testing.assert_frame_equal(
        report, metrics.detection_report(collection, pred, class_names=names)
    )
    assert "extra" in caplog.text
    assert sorted(p.name for p in out.iterdir()) == [
        "comparisons", "confusion_matrix.csv", "confusion_matrix.png", "detection_report.csv",
        "matches.csv", "precision_recall.csv", "precision_recall.png",
    ]  # fmt: skip
    assert [p.name for p in (out / "comparisons").iterdir()] == ["scan.png"]
    assert sorted(pd.read_csv(out / "matches.csv")["status"]) == ["fn", "fp", "tp"]
    matrix = pd.read_csv(out / "confusion_matrix.csv", index_col=0)
    assert matrix.loc["galeata", "background"] == 1 and matrix.loc["background", "pulex"] == 1
    assert pd.read_csv(out / "detection_report.csv", index_col=0).loc["all", "tp"] == 1
    assert len(pd.read_csv(out / "precision_recall.csv")) == 2
    assert hires.evaluate is metrics.evaluate


def test_evaluate_options(collection, tmp_path):
    pred = collection.replace(annotations=collection[:1])
    metrics.evaluate(collection, pred, tmp_path / "plain", save_images=False)
    assert not (tmp_path / "plain" / "comparisons").exists()
    bare = collection.replace(image_path=None)
    metrics.evaluate(bare, pred.replace(image_path=None), tmp_path / "bare")
    assert (tmp_path / "bare" / "matches.csv").is_file()
    assert not (tmp_path / "bare" / "comparisons").exists()
    with pytest.raises(ValueError, match="in common"):
        metrics.evaluate(
            hires.Album([collection]), hires.Album([pred.replace(name="other")]), tmp_path / "x"
        )
    assert not (tmp_path / "x").exists()
