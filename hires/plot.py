"""Visualisation of annotations, matches and evaluation results.

Image-based functions (:func:`overlay`, :func:`comparison`) return an RGB
``numpy.ndarray`` that can be shown with ``matplotlib.pyplot.imshow``.
Chart functions return the Matplotlib ``Axes`` or ``Figure`` they drew on
and never call ``plt.show()``. All functions accept ``save=`` to write the
result to a file.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from .annotation import Annotation
from .io import read_image

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .metrics import Match, MatchResult

__all__ = [
    "PALETTE",
    "STATUS_COLORS",
    "annotation",
    "comparison",
    "confusion_matrix",
    "individual",
    "overlay",
    "precision_recall",
    "training_curves",
]

#: RGB colours cycled through by class id in :func:`overlay`.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (188, 189, 34),
    (23, 190, 207),
    (127, 127, 127),
)

#: RGB colour of each match status in :func:`comparison`.
STATUS_COLORS: dict[str, tuple[int, int, int]] = {
    "tp": (0, 200, 0),
    "misclassified": (255, 165, 0),
    "fp": (220, 0, 0),
    "fn": (0, 100, 220),
}


#: Colours of ground truth and prediction in :func:`individual`.
GT_COLOR = "#1f77b4"
PRED_COLOR = "#d62728"


def _load_bgr(image: str | os.PathLike[str] | np.ndarray) -> np.ndarray:
    if isinstance(image, (str, os.PathLike)):
        return read_image(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("An image array must have shape (height, width, 3).")
    return image


def _pixel_points(annotation: Annotation, width: int, height: int) -> np.ndarray:
    import numpy as np

    points = np.asarray(annotation.polygon.exterior.coords[:-1], dtype=np.float64)
    return np.round(points * (width, height)).astype(np.int32)


def _finish_image(blended: np.ndarray, save: str | os.PathLike[str] | None) -> np.ndarray:
    import cv2

    if save is not None:
        path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), blended):
            raise OSError(f"Could not write image to {path}.")
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def overlay(
    image: str | os.PathLike[str] | np.ndarray,
    annotations: Iterable[Annotation],
    *,
    class_names: Mapping[int, str] | None = None,
    fill: bool = True,
    boxes: bool = True,
    labels: bool = True,
    alpha: float = 0.5,
    save: str | os.PathLike[str] | None = None,
) -> np.ndarray:
    """Draw annotations on top of their image.

    Parameters
    ----------
    image : path-like or numpy.ndarray
        Image file, or an 8-bit BGR array of shape ``(height, width, 3)``.
    annotations : iterable of Annotation
        Usually a :class:`~hires.Collection`.
    class_names : mapping of int to str, optional
        ``{class_id: label}`` for the text labels. Class ids are shown when
        omitted.
    fill : bool, default True
        Fill and outline the polygons.
    boxes : bool, default True
        Draw axis-aligned bounding boxes.
    labels : bool, default True
        Write ``"{index} {class} {confidence}"`` at each object.
    alpha : float, default 0.5
        Opacity of the drawing over the image.
    save : path-like, optional
        Write the result to this file (any format OpenCV supports).

    Returns
    -------
    numpy.ndarray
        RGB image of shape ``(height, width, 3)``.

    Examples
    --------
    >>> rgb = hires.plot.overlay("scan.tif", collection, save="annotated.png")  # doctest: +SKIP
    >>> plt.imshow(rgb)  # doctest: +SKIP
    """
    import cv2

    base = _load_bgr(image)
    height, width = base.shape[:2]
    canvas = base.copy()
    names = class_names or {}

    for index, ann in enumerate(annotations):
        red, green, blue = PALETTE[ann.class_id % len(PALETTE)]
        color = (blue, green, red)
        points = _pixel_points(ann, width, height)

        if fill:
            cv2.polylines(canvas, [points], True, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.polylines(canvas, [points], True, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.fillPoly(canvas, [points], color)
        if boxes:
            x0, y0 = points.min(axis=0)
            x1, y1 = points.max(axis=0)
            cv2.rectangle(canvas, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
        if labels:
            text = f"{index} {names.get(ann.class_id, ann.class_id)}"
            if ann.confidence is not None:
                text += f" {ann.confidence:.2f}"
            x, y = (int(v) for v in points.mean(axis=0))
            y = y - 10 if y > 20 else y + 20
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, text, (x + 1, y + 1), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, text, (x, y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    blended = cv2.addWeighted(canvas, alpha, base, 1.0 - alpha, 0)
    return _finish_image(blended, save)


def comparison(
    image: str | os.PathLike[str] | np.ndarray,
    matches: MatchResult,
    *,
    alpha: float = 0.5,
    save: str | os.PathLike[str] | None = None,
) -> np.ndarray:
    """Draw the outcome of matching predictions against ground truth.

    True positives and misclassifications are drawn with the predicted
    polygon, false positives with the prediction and false negatives with
    the missed ground-truth polygon, coloured by :data:`STATUS_COLORS`
    (green, orange, red, blue).

    Parameters
    ----------
    image : path-like or numpy.ndarray
        Image file, or an 8-bit BGR array.
    matches : MatchResult
        Result of :func:`hires.metrics.match` for this image. For an album
        result, select one image first:
        ``matches.filter(lambda m: m.image == "scan")``.
    alpha : float, default 0.5
        Opacity of the drawing over the image.
    save : path-like, optional
        Write the result to this file.

    Returns
    -------
    numpy.ndarray
        RGB image of shape ``(height, width, 3)``.
    """
    import cv2

    base = _load_bgr(image)
    height, width = base.shape[:2]
    canvas = base.copy()

    for m in matches:
        shown = m.pred if m.pred is not None else m.gt
        red, green, blue = STATUS_COLORS[m.status]
        points = _pixel_points(shown, width, height)
        cv2.fillPoly(canvas, [points], (blue, green, red))
        cv2.polylines(canvas, [points], True, (255, 255, 255), 2, cv2.LINE_AA)

    blended = cv2.addWeighted(canvas, alpha, base, 1.0 - alpha, 0)
    return _finish_image(blended, save)


def individual(
    match: Match,
    image: str | os.PathLike[str] | np.ndarray | None = None,
    *,
    class_names: Mapping[int, str] | None = None,
    padding: float = 0.4,
    figsize: tuple[float, float] = (10, 10),
    save: str | os.PathLike[str] | None = None,
) -> Figure:
    """Compare the ground-truth and predicted mask of a single object.

    Crops the image around the object and draws a 2 x 2 grid: the
    ground-truth polygon, the predicted polygon, the raw crop, and both
    outlines together with their IoU and match status.

    Parameters
    ----------
    match : Match
        One element of a :class:`~hires.metrics.MatchResult`. Select it by
        position (``matches[i]``), with
        :meth:`~hires.metrics.MatchResult.pick` (a representative example) or
        with :meth:`~hires.metrics.MatchResult.find` (a specific annotation).
    image : path-like or numpy.ndarray, optional
        Image file, or an 8-bit BGR array. Defaults to ``match.image_path``,
        which is known when the collections were loaded with their images.
        Pass an array to avoid re-reading a large scan for every figure.
    class_names : mapping of int to str, optional
        ``{class_id: label}`` for the panel titles.
    padding : float, default 0.4
        Margin around the object, as a fraction of its larger side.
    figsize : tuple of float, default (10, 10)
        Figure size in inches.
    save : path-like, optional
        Write the figure to this file.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If no image is given and the match has no ``image_path``.

    See Also
    --------
    comparison : All matches of an image at once.

    Examples
    --------
    >>> gt = hires.read_album("gt/", image_dir="images/")  # doctest: +SKIP
    >>> pred = hires.read_album("pred/", image_dir="images/")  # doctest: +SKIP
    >>> matches = hires.metrics.match(gt, pred)  # doctest: +SKIP
    >>> typical = matches.pick("tp", by="median_iou", image="C1-1_control")  # doctest: +SKIP
    >>> hires.plot.individual(typical, save="plots/example.png")  # doctest: +SKIP

    The object labelled ``158`` in an overlay image:

    >>> m = matches.find(pred_index=158, image="C1-1_control")  # doctest: +SKIP
    >>> hires.plot.individual(m)  # doctest: +SKIP
    """
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Polygon as PolygonPatch

    if image is None:
        if match.image_path is None:
            raise ValueError(
                "The match has no image_path; pass image=..., or load the "
                "collections with their images (read_album(..., image_dir=...))."
            )
        image = match.image_path
    pixels = cv2.cvtColor(_load_bgr(image), cv2.COLOR_BGR2RGB)
    height, width = pixels.shape[:2]

    def outline(ann: Annotation | None) -> np.ndarray | None:
        if ann is None:
            return None
        return np.asarray(ann.to_pixels((width, height)).exterior.coords, dtype=float)

    gt_xy, pred_xy = outline(match.gt), outline(match.pred)
    points = np.vstack([xy for xy in (gt_xy, pred_xy) if xy is not None])
    (min_x, min_y), (max_x, max_y) = points.min(axis=0), points.max(axis=0)
    pad = padding * max(max_x - min_x, max_y - min_y, 1.0)
    x0, y0 = max(0, int(min_x - pad)), max(0, int(min_y - pad))
    x1, y1 = min(width, int(np.ceil(max_x + pad))), min(height, int(np.ceil(max_y + pad)))
    crop = pixels[y0:y1, x0:x1]
    if gt_xy is not None:
        gt_xy = gt_xy - (x0, y0)
    if pred_xy is not None:
        pred_xy = pred_xy - (x0, y0)

    def label(ann: Annotation | None) -> str:
        if ann is None:
            return "-"
        return str((class_names or {}).get(ann.class_id, ann.class_id))

    figure, axes = plt.subplots(2, 2, figsize=figsize)
    (ax_gt, ax_pred), (ax_raw, ax_both) = axes
    for ax in axes.flat:
        ax.imshow(crop)
        ax.set_xticks([])
        ax.set_yticks([])

    def draw_mask(ax: Axes, xy: np.ndarray | None, color: str, missing: str) -> None:
        if xy is None:
            ax.text(0.5, 0.5, missing, ha="center", va="center", transform=ax.transAxes,
                    color="white", bbox={"facecolor": "black", "alpha": 0.5, "lw": 0})
        else:
            ax.add_patch(
                PolygonPatch(xy, closed=True, facecolor=color, edgecolor="white", lw=2, alpha=0.45)
            )

    ax_gt.set_title(f"Ground truth\nclass = {label(match.gt)}")
    draw_mask(ax_gt, gt_xy, GT_COLOR, "no ground truth")

    title = f"Prediction\nclass = {label(match.pred)}"
    if match.pred is not None and match.pred.confidence is not None:
        title += f"   confidence = {match.pred.confidence:.2f}"
    ax_pred.set_title(title)
    draw_mask(ax_pred, pred_xy, PRED_COLOR, "no prediction")

    ax_raw.set_title("Raw crop")

    ax_both.set_title(f"Overlay\nIoU = {match.iou:.3f}   status = {match.status}")
    for xy, color, name in ((gt_xy, GT_COLOR, "Ground truth"), (pred_xy, PRED_COLOR, "Prediction")):
        if xy is not None:
            ax_both.add_patch(
                PolygonPatch(xy, closed=True, fill=False, edgecolor=color, lw=2.5, label=name)
            )
    ax_both.legend(loc="lower right", framealpha=0.85)

    heading = [str(match.image)] if match.image is not None else []
    if match.gt_index is not None:
        heading.append(f"gt #{match.gt_index}")
    if match.pred_index is not None:
        heading.append(f"pred #{match.pred_index}")
    figure.suptitle("  |  ".join(heading), fontsize=13)
    figure.tight_layout()

    _save_figure(figure, save, dpi=200)
    return figure


def _save_figure(figure: Figure, save: str | os.PathLike[str] | None, dpi: int = 300) -> None:
    if save is not None:
        path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight")


def confusion_matrix(
    matrix: pd.DataFrame,
    *,
    normalize: bool = True,
    ax: Axes | None = None,
    cmap: str = "Blues",
    title: str | None = "Confusion matrix",
    save: str | os.PathLike[str] | None = None,
) -> Axes:
    """Draw a confusion matrix as a heat map.

    Parameters
    ----------
    matrix : pandas.DataFrame
        Result of :func:`hires.metrics.confusion_matrix`
        (rows = ground truth, columns = prediction).
    normalize : bool, default True
        Show each ground-truth class as fractions that sum to one, so the
        diagonal reads as recall. If false, raw counts are shown.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when omitted.
    cmap : str, default "Blues"
        Matplotlib colour map.
    title : str or None, default "Confusion matrix"
        Axes title.
    save : path-like, optional
        Write the figure to this file.

    Returns
    -------
    matplotlib.axes.Axes
        Ground truth on the x axis, prediction on the y axis, first class at
        the bottom left.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    counts = matrix.to_numpy(dtype=float).T  # rows = prediction, columns = ground truth
    if normalize:
        totals = counts.sum(axis=0, keepdims=True)
        values = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
        v_max = 1.0
    else:
        values = counts
        v_max = max(float(counts.max()), 1.0) if counts.size else 1.0

    if ax is None:
        side = max(5.0, 0.9 * len(matrix) + 2.5)
        _, ax = plt.subplots(figsize=(side + 1.0, side))

    mesh = ax.imshow(values, cmap=cmap, origin="lower", vmin=0, vmax=v_max, aspect="auto")
    colorbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Fraction of ground truth" if normalize else "Count")

    ticks = np.arange(len(matrix))
    ax.set_xticks(ticks, [str(label) for label in matrix.index], rotation=45, ha="right")
    ax.set_yticks(ticks, [str(label) for label in matrix.columns])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}" if normalize else f"{int(value):,}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > v_max / 2 else "black",
            )
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Predicted")
    if title:
        ax.set_title(title)

    _save_figure(ax.figure, save)
    return ax


def precision_recall(
    curves: pd.DataFrame | Mapping[str, pd.DataFrame],
    *,
    ax: Axes | None = None,
    title: str | None = "Precision-recall curve",
    save: str | os.PathLike[str] | None = None,
) -> Axes:
    """Draw precision-recall curves and mark the point with the best F1.

    Parameters
    ----------
    curves : pandas.DataFrame or mapping of str to pandas.DataFrame
        Result of :func:`hires.metrics.precision_recall_curve`, or several
        of them as ``{label: curve}`` (e.g. one per class or per model).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when omitted.
    title : str or None, default "Precision-recall curve"
        Axes title.
    save : path-like, optional
        Write the figure to this file.

    Returns
    -------
    matplotlib.axes.Axes
        Recall on the x axis, precision on the y axis.

    Examples
    --------
    >>> curves = {  # doctest: +SKIP
    ...     name: hires.metrics.precision_recall_curve(gt, pred, class_id=class_id)
    ...     for class_id, name in class_names.items()
    ... }
    >>> hires.plot.precision_recall(curves, save="pr.png")  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    labelled = not hasattr(curves, "columns")
    if not labelled:
        curves = {"": curves}
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    for label, frame in curves.items():
        (line,) = ax.plot(
            frame["recall"], frame["precision"], lw=2, label=str(label) if labelled else None
        )
        if len(frame):
            best = frame.loc[frame["f1"].idxmax()]
            ax.plot(best["recall"], best["precision"], "o", color=line.get_color())
            ax.annotate(
                f"F1 {best['f1']:.2f} at confidence {best['confidence']:.2f}",
                (best["recall"], best["precision"]),
                textcoords="offset points",
                xytext=(-6, -12),
                ha="right",
                fontsize=8,
            )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    if labelled:
        ax.legend(fontsize=9)

    _save_figure(ax.figure, save)
    return ax


def training_curves(
    results_csv: str | os.PathLike[str],
    *,
    patience: int | None = None,
    losses: Mapping[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
    save: str | os.PathLike[str] | None = None,
) -> Figure:
    """Plot training and validation losses from an Ultralytics ``results.csv``.

    Parameters
    ----------
    results_csv : path-like
        The ``results.csv`` written by YOLO training.
    patience : int, optional
        Early-stopping patience used for training. When given, a vertical
        line marks the best epoch, ``last_epoch - patience``. Only meaningful
        if training was actually stopped by early stopping.
    losses : mapping of str to str, optional
        ``{loss_name: panel_title}`` of the losses to plot, where the CSV has
        the columns ``train/{loss_name}`` and ``val/{loss_name}``. Defaults to
        the classification and segmentation loss.
    figsize : tuple of float, optional
        Figure size in inches.
    save : path-like, optional
        Write the figure to this file.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per loss.

    Raises
    ------
    KeyError
        If a requested loss column is missing from the CSV.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    losses = dict(
        losses or {"cls_loss": "Classification loss", "seg_loss": "Segmentation loss"}
    )
    frame = pd.read_csv(results_csv, skipinitialspace=True)
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [
        column
        for loss in losses
        for column in (f"train/{loss}", f"val/{loss}")
        if column not in frame.columns
    ]
    if missing:
        raise KeyError(f"{results_csv} has no column(s) {missing}.")

    best_epoch = None if patience is None else int(frame["epoch"].max()) - int(patience)

    figure, axes = plt.subplots(
        len(losses),
        1,
        figsize=figsize or (10, 3.5 * len(losses)),
        sharex=True,
        squeeze=False,
    )
    for ax, (loss, panel_title) in zip(axes[:, 0], losses.items(), strict=True):
        ax.plot(frame["epoch"], frame[f"val/{loss}"], "--", color="#5b9bd5", label="Validation")
        ax.plot(frame["epoch"], frame[f"train/{loss}"], "-", color="#f28c28", label="Training")
        if best_epoch is not None:
            ax.axvline(
                best_epoch, color="#d94f3d", linestyle="--", label=f"Best epoch ({best_epoch})"
            )
        ax.set_title(panel_title)
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("Epoch")
    figure.tight_layout()

    _save_figure(figure, save)
    return figure


def annotation(
    ann: Annotation,
    *,
    image_size: tuple[int, int] | None = None,
    obb: bool = False,
    bbox: bool = False,
    padding: float = 0.25,
    ax: Axes | None = None,
    save: str | os.PathLike[str] | None = None,
) -> Axes:
    """Draw the outline of a single annotation.

    Parameters
    ----------
    ann : Annotation
        The annotation to draw.
    image_size : tuple of int, optional
        ``(width, height)`` of the image. When given, the polygon is drawn in
        pixel coordinates with its true aspect ratio; otherwise in normalized
        coordinates.
    obb : bool, default False
        Also draw the minimum rotated bounding rectangle.
    bbox : bool, default False
        Also draw the axis-aligned bounding box.
    padding : float, default 0.25
        Empty margin around the polygon, as a fraction of its larger side.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when omitted.
    save : path-like, optional
        Write the figure to this file.

    Returns
    -------
    matplotlib.axes.Axes
        With the y axis pointing down, like image coordinates.
    """
    import matplotlib.pyplot as plt

    polygon = ann.polygon if image_size is None else ann.to_pixels(image_size)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    x, y = polygon.exterior.xy
    ax.fill(x, y, alpha=0.35, color="#4c72b0")
    ax.plot(x, y, color="#1f3b5d", lw=2, label="polygon")
    if obb:
        ox, oy = polygon.minimum_rotated_rectangle.exterior.xy
        ax.plot(ox, oy, "-.", color="#e76f51", lw=2, label="oriented box")
    if bbox:
        bx, by = polygon.envelope.exterior.xy
        ax.plot(bx, by, "--", color="black", lw=1.5, label="bounding box")

    min_x, min_y, max_x, max_y = polygon.bounds
    half = max(max_x - min_x, max_y - min_y) * (0.5 + padding)
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    ax.set_xlim(center_x - half, center_x + half)
    ax.set_ylim(center_y + half, center_y - half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (px)" if image_size else "x (normalized)")
    ax.set_ylabel("y (px)" if image_size else "y (normalized)")

    title = f"class {ann.class_id}"
    if ann.confidence is not None:
        title += f", confidence {ann.confidence:.2f}"
    ax.set_title(title)
    if obb or bbox:
        ax.legend(fontsize=8)

    _save_figure(ax.figure, save)
    return ax

