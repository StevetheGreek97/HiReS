"""Comparing predicted annotations against ground truth.

Every function in this module takes the ground truth first and the
predictions second, like ``sklearn.metrics``::

    matches = hires.metrics.match(gt, pred, iou_threshold=0.5)
    report = hires.metrics.detection_report(gt, pred)

Both arguments are either two :class:`~hires.Collection` objects (one image)
or two :class:`~hires.Album` objects (a dataset; collections are paired by
name).

Match statuses
--------------
``"tp"``
    A prediction matched a ground-truth object of the same class.
``"misclassified"``
    A prediction matched a ground-truth object of a different class.
``"fp"``
    A prediction without a ground-truth partner (false positive).
``"fn"``
    A ground-truth object without a prediction (false negative).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from .album import Album
from .annotation import TRAIT_NAMES, Annotation
from .collection import Collection

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "STATUSES",
    "Match",
    "MatchResult",
    "average_precision",
    "confusion_matrix",
    "detection_report",
    "evaluate",
    "iou",
    "iou_matrix",
    "match",
    "precision_recall_curve",
]

logger = logging.getLogger(__name__)

#: The possible values of :attr:`Match.status`.
STATUSES: tuple[str, ...] = ("tp", "misclassified", "fp", "fn")

#: IoU thresholds averaged for the COCO-style ``ap50_95`` column.
COCO_THRESHOLDS: tuple[float, ...] = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2).tolist())


# ---------------------------------------------------------------------- #
# IoU
# ---------------------------------------------------------------------- #
def _polygon(obj: Annotation | Polygon) -> Polygon:
    return obj.polygon if isinstance(obj, Annotation) else obj


def iou(a: Annotation | Polygon, b: Annotation | Polygon) -> float:
    """Intersection-over-union of two polygons.

    Parameters
    ----------
    a, b : Annotation or shapely.geometry.Polygon

    Returns
    -------
    float
        Value in ``[0, 1]``; ``0.0`` when the polygons do not overlap.

    Examples
    --------
    >>> from shapely.geometry import box
    >>> iou(box(0, 0, 2, 1), box(1, 0, 3, 1))
    0.3333333333333333
    """
    first, second = _polygon(a), _polygon(b)
    intersection = first.intersection(second).area
    if intersection == 0:
        return 0.0
    union = first.area + second.area - intersection
    return float(intersection / union) if union > 0 else 0.0


def iou_matrix(
    gt: Iterable[Annotation | Polygon],
    pred: Iterable[Annotation | Polygon],
) -> np.ndarray:
    """Pairwise polygon IoU between two sets of annotations.

    Parameters
    ----------
    gt, pred : iterable of Annotation or Polygon
        Typically two :class:`~hires.Collection` objects of the same image.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(gt), len(pred))``; element ``[i, j]`` is the
        IoU of ``gt[i]`` and ``pred[j]``.
    """
    gt_polygons = np.array([_polygon(a) for a in gt], dtype=object)
    pred_polygons = np.array([_polygon(a) for a in pred], dtype=object)
    matrix = np.zeros((len(gt_polygons), len(pred_polygons)), dtype=float)
    if matrix.size == 0:
        return matrix

    tree = STRtree(pred_polygons)
    gt_index, pred_index = tree.query(gt_polygons, predicate="intersects")
    if gt_index.size == 0:
        return matrix

    left = gt_polygons[gt_index]
    right = pred_polygons[pred_index]
    intersection = shapely.area(shapely.intersection(left, right))
    union = shapely.area(left) + shapely.area(right) - intersection
    values = np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0
    )
    matrix[gt_index, pred_index] = values
    return matrix


# ---------------------------------------------------------------------- #
# Matching
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class Match:
    """The outcome for one ground-truth object and/or one prediction.

    Attributes
    ----------
    gt : Annotation or None
        The ground-truth annotation; ``None`` for a false positive.
    pred : Annotation or None
        The predicted annotation; ``None`` for a false negative.
    iou : float
        IoU of the two polygons; ``0.0`` when unmatched.
    status : str
        One of ``"tp"``, ``"misclassified"``, ``"fp"``, ``"fn"``.
    image : str or None
        Name of the collection the annotations come from.
    gt_index, pred_index : int or None
        Positions of the annotations in their collections. ``pred_index`` is
        the number drawn next to each object by :func:`hires.plot.overlay`.
    image_path : pathlib.Path or None
        Image file of the source collections, when known.
    """

    gt: Annotation | None
    pred: Annotation | None
    iou: float
    status: str
    image: str | None = None
    gt_index: int | None = None
    pred_index: int | None = None
    image_path: Path | None = None

    @property
    def is_matched(self) -> bool:
        """Whether a prediction was paired with a ground-truth object."""
        return self.gt is not None and self.pred is not None

    @property
    def class_match(self) -> bool:
        """Whether both annotations exist and share a class id."""
        return self.status == "tp"


class MatchResult(Sequence[Match]):
    """The result of :func:`match`: an immutable sequence of :class:`Match`.

    Attributes
    ----------
    iou_threshold : float
        IoU threshold the matching was run with.
    class_aware : bool
        Whether only same-class pairs were allowed to match.
    unpaired_gt, unpaired_pred : tuple of str
        Names of collections present in only one of the two albums. Their
        annotations are *not* part of the result.

    Examples
    --------
    >>> matches = hires.metrics.match(gt, pred)  # doctest: +SKIP
    >>> matches.counts()  # doctest: +SKIP
    {'tp': 412, 'misclassified': 9, 'fp': 31, 'fn': 27}
    >>> matches.with_status("fp", "fn").to_frame()  # doctest: +SKIP
    """

    def __init__(
        self,
        matches: Iterable[Match],
        *,
        iou_threshold: float,
        class_aware: bool = False,
        sources: Mapping[str | None, tuple[Collection, Collection]] | None = None,
        unpaired_gt: Iterable[str] = (),
        unpaired_pred: Iterable[str] = (),
    ) -> None:
        self._matches = tuple(matches)
        self.iou_threshold = float(iou_threshold)
        self.class_aware = bool(class_aware)
        self._sources = dict(sources or {})
        self.unpaired_gt = tuple(unpaired_gt)
        self.unpaired_pred = tuple(unpaired_pred)

    def __len__(self) -> int:
        return len(self._matches)

    def __iter__(self) -> Iterator[Match]:
        return iter(self._matches)

    @overload
    def __getitem__(self, index: int) -> Match: ...
    @overload
    def __getitem__(self, index: slice) -> MatchResult: ...
    def __getitem__(self, index: int | slice) -> Match | MatchResult:
        if isinstance(index, slice):
            return self._derive(self._matches[index])
        return self._matches[index]

    def __repr__(self) -> str:
        counts = ", ".join(f"{k}={v}" for k, v in self.counts().items())
        return f"MatchResult(iou_threshold={self.iou_threshold:g}, {counts})"

    def _derive(self, matches: Iterable[Match]) -> MatchResult:
        return MatchResult(
            matches,
            iou_threshold=self.iou_threshold,
            class_aware=self.class_aware,
            sources=self._sources,
            unpaired_gt=self.unpaired_gt,
            unpaired_pred=self.unpaired_pred,
        )

    def counts(self) -> dict[str, int]:
        """Count the matches per status, for every status in :data:`STATUSES`."""
        counts = dict.fromkeys(STATUSES, 0)
        for m in self._matches:
            counts[m.status] += 1
        return counts

    def summary(self) -> dict[str, float]:
        """Return the overall counts, precision, recall, F1 and mean IoU.

        Returns
        -------
        dict
            ``n_gt`` and ``n_pred`` (objects in the ground truth and in the
            predictions), the four status counts, ``precision``, ``recall``
            and ``f1`` (a misclassified pair counts against both; ``0.0`` when
            undefined), and ``mean_iou`` over the matched pairs (``nan``
            without any).

        See Also
        --------
        detection_report : The same metrics per class, with average precision.

        Examples
        --------
        >>> hires.metrics.match(gt, pred).summary()["f1"]  # doctest: +SKIP
        0.93
        """
        counts = self.counts()
        tp = counts["tp"]
        n_gt = tp + counts["misclassified"] + counts["fn"]
        n_pred = tp + counts["misclassified"] + counts["fp"]
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_gt if n_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ious = [m.iou for m in self._matches if m.is_matched]
        return {
            "n_gt": n_gt,
            "n_pred": n_pred,
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": float(np.mean(ious)) if ious else float("nan"),
        }

    def filter(self, predicate: Callable[[Match], bool]) -> MatchResult:
        """Return the matches for which ``predicate`` is true."""
        return self._derive(m for m in self._matches if predicate(m))

    @property
    def images(self) -> list[str | None]:
        """Names of the images that have at least one match, in order."""
        return list(dict.fromkeys(m.image for m in self._matches))

    def for_image(self, image: str) -> MatchResult:
        """Return only the matches of the image (collection) called ``image``."""
        return self.filter(lambda m: m.image == image)

    def find(
        self,
        *,
        gt_index: int | None = None,
        pred_index: int | None = None,
        image: str | None = None,
    ) -> Match:
        """Return the match that a given annotation takes part in.

        Parameters
        ----------
        gt_index : int, optional
            Position of the annotation in the ground-truth collection.
        pred_index : int, optional
            Position of the annotation in the predicted collection, i.e. the
            number :func:`hires.plot.overlay` draws next to each object.
            Give exactly one of ``gt_index`` and ``pred_index``.
        image : str, optional
            Image (collection) name. Required when the result covers more
            than one image, because indices are counted per image.

        Returns
        -------
        Match

        Raises
        ------
        ValueError
            If the arguments are ambiguous.
        LookupError
            If no such annotation exists.

        Examples
        --------
        >>> m = matches.find(pred_index=158, image="C1-1_control")  # doctest: +SKIP
        >>> m.status, m.iou  # doctest: +SKIP
        ('tp', 0.91)
        """
        if (gt_index is None) == (pred_index is None):
            raise ValueError("Give exactly one of gt_index and pred_index.")
        candidates = self._in_image(image)
        for m in candidates:
            if gt_index is not None and m.gt_index == gt_index:
                return m
            if pred_index is not None and m.pred_index == pred_index:
                return m
        which = f"gt_index={gt_index}" if gt_index is not None else f"pred_index={pred_index}"
        raise LookupError(f"No match with {which}" + (f" in image {image!r}." if image else "."))

    def pick(
        self,
        status: str = "tp",
        *,
        by: str = "median_iou",
        image: str | None = None,
        seed: int | None = None,
    ) -> Match:
        """Select one representative match with the given status.

        Parameters
        ----------
        status : str, default "tp"
            One of ``"tp"``, ``"misclassified"``, ``"fp"``, ``"fn"``.
        by : {"median_iou", "best_iou", "worst_iou", "random"}, default "median_iou"
            Which candidate to return: the one with the median, highest or
            lowest IoU, or a random one.
        image : str, optional
            Restrict the selection to one image (collection).
        seed : int, optional
            Seed for ``by="random"``.

        Returns
        -------
        Match

        Raises
        ------
        LookupError
            If there is no match with that status.

        See Also
        --------
        hires.plot.individual : Draw the selected match.

        Examples
        --------
        >>> typical = matches.pick("tp", image="C1-1_control")  # doctest: +SKIP
        >>> hires.plot.individual(typical, save="example.png")  # doctest: +SKIP
        """
        if status not in STATUSES:
            raise ValueError(f"Unknown status {status!r}; choose from {STATUSES}.")
        choices = ("median_iou", "best_iou", "worst_iou", "random")
        if by not in choices:
            raise ValueError(f"by must be one of {choices}, got {by!r}.")

        pool = self._matches if image is None else self._in_image(image)
        candidates = [m for m in pool if m.status == status]
        if not candidates:
            raise LookupError(
                f"No match with status {status!r}" + (f" in image {image!r}." if image else ".")
            )
        if by == "random":
            return candidates[int(np.random.default_rng(seed).integers(len(candidates)))]
        ordered = sorted(candidates, key=lambda m: m.iou)
        if by == "best_iou":
            return ordered[-1]
        if by == "worst_iou":
            return ordered[0]
        return ordered[len(ordered) // 2]

    def _in_image(self, image: str | None) -> tuple[Match, ...]:
        """Return the matches of ``image``; without a name the result must be single-image."""
        if image is None:
            if len(self.images) > 1:
                raise ValueError(
                    "The result covers several images; pass image=... "
                    f"(one of {self.images[:5]}{'...' if len(self.images) > 5 else ''})."
                )
            return self._matches
        if image not in self._sources and image not in self.images:
            raise LookupError(f"No image named {image!r} in the result.")
        return tuple(m for m in self._matches if m.image == image)

    def with_status(self, *statuses: str) -> MatchResult:
        """Return only the matches with one of the given statuses.

        Examples
        --------
        >>> errors = matches.with_status("fp", "fn")  # doctest: +SKIP
        """
        unknown = set(statuses) - set(STATUSES)
        if unknown:
            raise ValueError(f"Unknown status {sorted(unknown)}; choose from {STATUSES}.")
        return self.filter(lambda m: m.status in statuses)

    def to_frame(
        self,
        *,
        traits: bool = True,
        class_names: Mapping[int, str] | None = None,
    ) -> pd.DataFrame:
        """Return one row per match as a ``pandas.DataFrame``.

        The row index is the position of the match in this result, so
        ``result[i]`` is the match described by row ``i``.

        Parameters
        ----------
        traits : bool, default True
            Add ``gt_{trait}`` and ``pred_{trait}`` columns for every trait in
            :data:`hires.TRAIT_NAMES`, measured with the image size and
            physical scale of each source collection.
        class_names : mapping of int to str, optional
            ``{class_id: label}``. Adds the columns ``gt_class_name`` and
            ``pred_class_name``.

        Returns
        -------
        pandas.DataFrame
            Columns ``image``, ``status``, ``iou``, ``class_match``,
            ``gt_index``, ``pred_index``, ``gt_class``, ``pred_class``,
            ``gt_confidence``, ``pred_confidence`` and, optionally, the class
            name and trait columns. Fields of a missing annotation are null.

        Examples
        --------
        List the individuals of one image, then look one of them up:

        >>> table = matches.for_image("C1-1_control").to_frame(  # doctest: +SKIP
        ...     traits=False, class_names={0: "D. pulex", 1: "D. galeata"}
        ... )
        >>> table[table["status"] == "misclassified"]  # doctest: +SKIP
        """
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for m in self._matches:
            row: dict[str, Any] = {
                "image": m.image,
                "status": m.status,
                "iou": m.iou,
                "class_match": m.class_match,
                "gt_index": m.gt_index,
                "pred_index": m.pred_index,
                "gt_class": None if m.gt is None else m.gt.class_id,
                "pred_class": None if m.pred is None else m.pred.class_id,
                "gt_confidence": None if m.gt is None else m.gt.confidence,
                "pred_confidence": None if m.pred is None else m.pred.confidence,
            }
            if traits:
                gt_source, pred_source = self._sources.get(m.image, (None, None))
                row.update(_prefixed_traits("gt_", m.gt, gt_source))
                row.update(_prefixed_traits("pred_", m.pred, pred_source))
            rows.append(row)

        columns = [
            "image",
            "status",
            "iou",
            "class_match",
            "gt_index",
            "pred_index",
            "gt_class",
            "pred_class",
            "gt_confidence",
            "pred_confidence",
        ]
        if traits:
            columns += [f"gt_{t}" for t in TRAIT_NAMES] + [f"pred_{t}" for t in TRAIT_NAMES]

        frame = pd.DataFrame(rows, columns=columns)
        for column in ("gt_index", "pred_index", "gt_class", "pred_class"):
            frame[column] = frame[column].astype("Int64")
        for column in columns[8:]:
            frame[column] = frame[column].astype(float)
        frame["class_match"] = frame["class_match"].astype(bool)
        if class_names is not None:
            for position, side in ((8, "pred"), (7, "gt")):
                labels = [
                    None if pd.isna(c) else class_names.get(int(c), str(int(c)))
                    for c in frame[f"{side}_class"]
                ]
                frame.insert(position, f"{side}_class_name", pd.Series(labels, dtype=object))
        return frame


def _prefixed_traits(
    prefix: str, annotation: Annotation | None, source: Collection | None
) -> dict[str, float]:
    if annotation is None:
        return {f"{prefix}{name}": np.nan for name in TRAIT_NAMES}
    image_size = None if source is None else source.image_size
    pixel_size = 1.0 if source is None else source.pixel_size
    measured = annotation.traits(image_size, pixel_size=pixel_size)
    return {f"{prefix}{name}": value for name, value in measured.items()}


@dataclass(frozen=True)
class _Pairing:
    """Collections of one image plus their cached IoU matrix."""

    name: str | None
    gt: Collection
    pred: Collection
    ious: np.ndarray


def _pair_inputs(
    gt: Collection | Album, pred: Collection | Album, strict: bool = False
) -> tuple[list[_Pairing], list[str], list[str]]:
    if isinstance(gt, Collection) and isinstance(pred, Collection):
        name = gt.name if gt.name is not None else pred.name
        return [_Pairing(name, gt, pred, iou_matrix(gt, pred))], [], []

    if isinstance(gt, Album) and isinstance(pred, Album):
        gt_only = [n for n in gt.names if n not in pred]
        pred_only = [n for n in pred.names if n not in gt]
        if strict and (gt_only or pred_only):
            raise ValueError(
                "The albums do not contain the same collections. "
                f"Only in gt: {gt_only}. Only in pred: {pred_only}."
            )
        pairings = [
            _Pairing(n, gt[n], pred[n], iou_matrix(gt[n], pred[n]))
            for n in gt.names
            if n in pred
        ]
        return pairings, gt_only, pred_only

    raise TypeError(
        "gt and pred must be two Collection objects or two Album objects, got "
        f"{type(gt).__name__} and {type(pred).__name__}."
    )


def _match_one(pairing: _Pairing, iou_threshold: float, class_aware: bool) -> list[Match]:
    """Greedy one-to-one matching in order of decreasing IoU."""
    gt, pred, ious = pairing.gt, pairing.pred, pairing.ious
    image_path = gt.image_path or pred.image_path
    gt_idx, pred_idx = np.nonzero((ious >= iou_threshold) & (ious > 0))
    order = np.argsort(-ious[gt_idx, pred_idx], kind="stable")

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[Match] = []

    for k in order:
        i, j = int(gt_idx[k]), int(pred_idx[k])
        if i in used_gt or j in used_pred:
            continue
        same_class = gt[i].class_id == pred[j].class_id
        if class_aware and not same_class:
            continue
        used_gt.add(i)
        used_pred.add(j)
        matches.append(
            Match(
                gt=gt[i],
                pred=pred[j],
                iou=float(ious[i, j]),
                status="tp" if same_class else "misclassified",
                image=pairing.name,
                gt_index=i,
                pred_index=j,
                image_path=image_path,
            )
        )

    matches.extend(
        Match(gt[i], None, 0.0, "fn", pairing.name, gt_index=i, image_path=image_path)
        for i in range(len(gt))
        if i not in used_gt
    )
    matches.extend(
        Match(None, pred[j], 0.0, "fp", pairing.name, pred_index=j, image_path=image_path)
        for j in range(len(pred))
        if j not in used_pred
    )
    return matches


def _check_threshold(iou_threshold: float) -> float:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold!r}.")
    return float(iou_threshold)


def match(
    gt: Collection | Album,
    pred: Collection | Album,
    *,
    iou_threshold: float = 0.5,
    class_aware: bool = False,
    strict: bool = False,
) -> MatchResult:
    """Match predictions to ground truth by polygon IoU.

    Candidate pairs with ``IoU >= iou_threshold`` (and ``IoU > 0``) are
    accepted greedily in order of decreasing IoU, so that every annotation
    takes part in at most one match.

    Parameters
    ----------
    gt : Collection or Album
        Ground-truth annotations.
    pred : Collection or Album
        Predicted annotations, of the same type as ``gt``. Albums are paired
        collection by collection using their names.
    iou_threshold : float, default 0.5
        Minimum IoU for two polygons to be matched.
    class_aware : bool, default False
        If false, polygons are matched regardless of class and a class
        disagreement yields the status ``"misclassified"``. If true, only
        polygons of the same class can match, so a class disagreement yields
        one ``"fn"`` and one ``"fp"`` instead.
    strict : bool, default False
        For albums: raise if a collection exists in only one album. By
        default such collections are skipped and listed in
        ``MatchResult.unpaired_gt`` / ``MatchResult.unpaired_pred``.

    Returns
    -------
    MatchResult

    Raises
    ------
    TypeError
        If ``gt`` and ``pred`` are not both collections or both albums.
    ValueError
        If ``strict`` is true and the albums differ in their collections.

    See Also
    --------
    detection_report : Precision, recall, F1 and average precision per class.
    confusion_matrix : Ground truth versus predicted class counts.

    Examples
    --------
    >>> from shapely.geometry import box
    >>> import hires
    >>> gt = hires.Collection([hires.Annotation(0, box(0, 0, 0.2, 0.2))], name="img")
    >>> pred = hires.Collection(
    ...     [
    ...         hires.Annotation(0, box(0, 0, 0.2, 0.19), confidence=0.9),
    ...         hires.Annotation(0, box(0.5, 0.5, 0.6, 0.6), confidence=0.4),
    ...     ],
    ...     name="img",
    ... )
    >>> hires.metrics.match(gt, pred).counts()
    {'tp': 1, 'misclassified': 0, 'fp': 1, 'fn': 0}
    """
    iou_threshold = _check_threshold(iou_threshold)
    pairings, gt_only, pred_only = _pair_inputs(gt, pred, strict)
    return _match_pairings(pairings, iou_threshold, class_aware, gt_only, pred_only)


def _match_pairings(
    pairings: Sequence[_Pairing],
    iou_threshold: float,
    class_aware: bool = False,
    gt_only: Iterable[str] = (),
    pred_only: Iterable[str] = (),
) -> MatchResult:
    matches: list[Match] = []
    for pairing in pairings:
        matches.extend(_match_one(pairing, iou_threshold, class_aware))
    return MatchResult(
        matches,
        iou_threshold=iou_threshold,
        class_aware=class_aware,
        sources={p.name: (p.gt, p.pred) for p in pairings},
        unpaired_gt=gt_only,
        unpaired_pred=pred_only,
    )


# ---------------------------------------------------------------------- #
# Aggregate metrics
# ---------------------------------------------------------------------- #
def _class_ids(matches: Iterable[Match], class_names: Mapping[int, str] | None) -> list[int]:
    ids = set(class_names or ())
    for m in matches:
        if m.gt is not None:
            ids.add(m.gt.class_id)
        if m.pred is not None:
            ids.add(m.pred.class_id)
    return sorted(ids)


def confusion_matrix(
    gt: Collection | Album,
    pred: Collection | Album,
    *,
    iou_threshold: float = 0.5,
    class_names: Mapping[int, str] | None = None,
    background: str = "background",
) -> pd.DataFrame:
    """Confusion matrix of ground-truth versus predicted classes.

    Parameters
    ----------
    gt, pred : Collection or Album
        Ground truth and predictions; see :func:`match`.
    iou_threshold : float, default 0.5
        Minimum IoU for two polygons to be matched.
    class_names : mapping of int to str, optional
        ``{class_id: label}``. Used as row and column labels, and to include
        classes that do not occur in the data.
    background : str, default "background"
        Label of the extra row/column that collects unmatched objects.

    Returns
    -------
    pandas.DataFrame
        Square table of counts with **rows = ground truth** and
        **columns = prediction**, ordered by class id with ``background``
        last. False negatives are counted in the ``background`` column,
        false positives in the ``background`` row.

    See Also
    --------
    hires.plot.confusion_matrix : Draw the matrix.
    """
    result = match(gt, pred, iou_threshold=iou_threshold)
    return _confusion_matrix(result, class_names, background)


def _confusion_matrix(
    result: MatchResult, class_names: Mapping[int, str] | None, background: str
) -> pd.DataFrame:
    import pandas as pd

    ids = _class_ids(result, class_names)
    position = {class_id: k for k, class_id in enumerate(ids)}
    last = len(ids)

    counts = np.zeros((last + 1, last + 1), dtype=int)
    for m in result:
        row = last if m.gt is None else position[m.gt.class_id]
        column = last if m.pred is None else position[m.pred.class_id]
        counts[row, column] += 1

    labels: list[Any] = [(class_names or {}).get(i, i) for i in ids] + [background]
    return pd.DataFrame(
        counts,
        index=pd.Index(labels, name="gt"),
        columns=pd.Index(labels, name="pred"),
    )


def _average_precision(result: MatchResult, class_ids: Sequence[int]) -> dict[int, float]:
    """All-point interpolated AP per class from one matching result."""
    out: dict[int, float] = {}
    for class_id in class_ids:
        n_gt = sum(m.gt is not None and m.gt.class_id == class_id for m in result)
        predictions = [
            m for m in result if m.pred is not None and m.pred.class_id == class_id
        ]
        if n_gt == 0 or not predictions:
            out[class_id] = 0.0
            continue

        confidence = np.array(
            [0.0 if m.pred.confidence is None else m.pred.confidence for m in predictions]
        )
        order = np.argsort(-confidence, kind="stable")
        hits = np.array([m.status == "tp" for m in predictions], dtype=float)[order]
        tp = np.cumsum(hits)
        fp = np.cumsum(1.0 - hits)
        recall = np.concatenate(([0.0], tp / n_gt, [1.0]))
        precision = np.concatenate(([0.0], tp / (tp + fp), [0.0]))
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        steps = np.nonzero(recall[1:] != recall[:-1])[0]
        out[class_id] = float(np.sum((recall[steps + 1] - recall[steps]) * precision[steps + 1]))
    return out


def average_precision(
    gt: Collection | Album,
    pred: Collection | Album,
    *,
    iou_threshold: float = 0.5,
) -> pd.Series:
    """Average precision per class at one IoU threshold.

    Predictions of each class are ranked by confidence and the area under
    the interpolated precision-recall curve is returned. A prediction counts
    as correct when :func:`match` paired it with a ground-truth object of the
    same class.

    Parameters
    ----------
    gt, pred : Collection or Album
        Ground truth and predictions; see :func:`match`.
    iou_threshold : float, default 0.5
        Minimum IoU for two polygons to be matched.

    Returns
    -------
    pandas.Series
        AP indexed by class id. Classes without ground truth or without
        predictions score ``0.0``.

    Notes
    -----
    Matching is greedy by IoU and independent of confidence, which differs
    slightly from the COCO protocol where predictions claim ground truth in
    confidence order. Predictions without a confidence are ranked last.
    """
    import pandas as pd

    iou_threshold = _check_threshold(iou_threshold)
    pairings, _, _ = _pair_inputs(gt, pred)
    result = _match_pairings(pairings, iou_threshold)
    values = _average_precision(result, _class_ids(result, None))
    return pd.Series(values, name=f"ap@{iou_threshold:g}", dtype=float).rename_axis("class_id")


def detection_report(
    gt: Collection | Album,
    pred: Collection | Album,
    *,
    iou_threshold: float = 0.5,
    ap_thresholds: Sequence[float] = COCO_THRESHOLDS,
    class_names: Mapping[int, str] | None = None,
) -> pd.DataFrame:
    """Per-class detection metrics.

    Parameters
    ----------
    gt, pred : Collection or Album
        Ground truth and predictions; see :func:`match`.
    iou_threshold : float, default 0.5
        IoU threshold used for the ``tp``, ``fp``, ``fn``, ``precision``,
        ``recall`` and ``f1`` columns.
    ap_thresholds : sequence of float, default 0.50, 0.55, ..., 0.95
        IoU thresholds averaged for the ``ap50_95`` column.
    class_names : mapping of int to str, optional
        ``{class_id: label}``; adds a ``class_name`` column and includes
        classes that do not occur in the data.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``class_id``, with a first row ``"all"``. Columns:

        ``tp``, ``fp``, ``fn``
            A misclassified pair counts as a false negative for the
            ground-truth class and a false positive for the predicted class.
        ``precision``, ``recall``, ``f1``
            Derived from the counts; ``0.0`` when undefined. The ``"all"``
            row is computed from the summed counts (micro average).
        ``ap50``, ``ap50_95``
            Average precision at IoU 0.5 and averaged over
            ``ap_thresholds``. The ``"all"`` row is the mean over classes
            (mAP).

    See Also
    --------
    average_precision : AP at a single threshold.

    Examples
    --------
    >>> report = hires.metrics.detection_report(gt, pred, iou_threshold=0.7)  # doctest: +SKIP
    >>> report.loc["all", ["precision", "recall", "ap50"]]  # doctest: +SKIP
    """
    iou_threshold = _check_threshold(iou_threshold)
    pairings, _, _ = _pair_inputs(gt, pred)
    return _detection_report(pairings, iou_threshold, ap_thresholds, class_names)


def _detection_report(
    pairings: Sequence[_Pairing],
    iou_threshold: float,
    ap_thresholds: Sequence[float],
    class_names: Mapping[int, str] | None,
) -> pd.DataFrame:
    import pandas as pd

    result = _match_pairings(pairings, iou_threshold)
    ids = _class_ids(result, class_names)

    rows: dict[Any, dict[str, float]] = {}
    for class_id in ids:
        tp = sum(m.status == "tp" and m.gt.class_id == class_id for m in result)
        fn = sum(
            m.status in ("fn", "misclassified") and m.gt.class_id == class_id for m in result
        )
        fp = sum(
            m.status in ("fp", "misclassified") and m.pred.class_id == class_id
            for m in result
        )
        rows[class_id] = {"tp": tp, "fp": fp, "fn": fn}

    ap50 = _average_precision(_match_pairings(pairings, 0.5), ids)
    per_threshold = [
        _average_precision(_match_pairings(pairings, float(t)), ids) for t in ap_thresholds
    ]
    for class_id in ids:
        rows[class_id]["ap50"] = ap50[class_id]
        rows[class_id]["ap50_95"] = (
            float(np.mean([ap[class_id] for ap in per_threshold])) if per_threshold else np.nan
        )

    total = {
        "tp": sum(r["tp"] for r in rows.values()),
        "fp": sum(r["fp"] for r in rows.values()),
        "fn": sum(r["fn"] for r in rows.values()),
        "ap50": float(np.mean([r["ap50"] for r in rows.values()])) if rows else np.nan,
        "ap50_95": float(np.mean([r["ap50_95"] for r in rows.values()])) if rows else np.nan,
    }

    report = pd.DataFrame.from_dict({"all": total, **rows}, orient="index")
    report.index.name = "class_id"
    tp, fp, fn = report["tp"], report["fp"], report["fn"]
    report["precision"] = (tp / (tp + fp)).where(tp + fp > 0, 0.0)
    report["recall"] = (tp / (tp + fn)).where(tp + fn > 0, 0.0)
    denominator = report["precision"] + report["recall"]
    report["f1"] = (2 * report["precision"] * report["recall"] / denominator).where(
        denominator > 0, 0.0
    )
    report[["tp", "fp", "fn"]] = report[["tp", "fp", "fn"]].astype(int)

    columns = ["tp", "fp", "fn", "precision", "recall", "f1", "ap50", "ap50_95"]
    if class_names is not None:
        report.insert(0, "class_name", [class_names.get(i, str(i)) for i in report.index])
        columns.insert(0, "class_name")
    return report[columns]


def precision_recall_curve(
    gt: Collection | Album,
    pred: Collection | Album,
    *,
    iou_threshold: float = 0.5,
    class_id: int | None = None,
) -> pd.DataFrame:
    """Compute precision and recall for every confidence threshold.

    Use it to choose the confidence threshold of a model: each row tells how
    the model would score if only the predictions with at least that
    confidence were kept.

    Parameters
    ----------
    gt, pred : Collection or Album
        Ground truth and predictions; see :func:`match`. Run the model with
        a low confidence threshold so the curve covers the whole range.
    iou_threshold : float, default 0.5
        Minimum IoU for two polygons to be matched.
    class_id : int, optional
        Restrict the curve to one class. By default all classes are pooled.

    Returns
    -------
    pandas.DataFrame
        Columns ``confidence``, ``precision``, ``recall`` and ``f1``, one row
        per distinct confidence, ordered by decreasing confidence.

    See Also
    --------
    average_precision : The area under this curve.
    hires.plot.precision_recall : Draw the curve.

    Notes
    -----
    Predictions are matched once, using all of them, and then ranked; see
    the notes of :func:`average_precision`. A prediction counts as correct
    when it matched a ground-truth object of the same class.

    Examples
    --------
    The confidence threshold with the best F1 score:

    >>> curve = hires.metrics.precision_recall_curve(gt, pred)  # doctest: +SKIP
    >>> curve.loc[curve["f1"].idxmax(), "confidence"]  # doctest: +SKIP
    0.42
    """
    iou_threshold = _check_threshold(iou_threshold)
    pairings, _, _ = _pair_inputs(gt, pred)
    return _precision_recall_curve(_match_pairings(pairings, iou_threshold), class_id)


def _precision_recall_curve(result: MatchResult, class_id: int | None) -> pd.DataFrame:
    import pandas as pd

    predictions = [
        m for m in result if m.pred is not None and class_id in (None, m.pred.class_id)
    ]
    n_gt = sum(m.gt is not None and class_id in (None, m.gt.class_id) for m in result)

    confidence = np.array(
        [0.0 if m.pred.confidence is None else m.pred.confidence for m in predictions],
        dtype=float,
    )
    order = np.argsort(-confidence, kind="stable")
    hits = np.array([m.status == "tp" for m in predictions], dtype=float)[order]
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, len(hits) + 1)
    recall = tp / n_gt if n_gt else np.zeros_like(tp)
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(tp), where=total > 0)

    frame = pd.DataFrame(
        {"confidence": confidence[order], "precision": precision, "recall": recall, "f1": f1}
    )
    # Predictions with equal confidence pass or fail a threshold together.
    return frame.drop_duplicates("confidence", keep="last").reset_index(drop=True)


def evaluate(
    gt: Collection | Album,
    pred: Collection | Album,
    output_dir: str | os.PathLike[str],
    *,
    iou_threshold: float = 0.5,
    class_names: Mapping[int, str] | None = None,
    save_images: bool = True,
) -> pd.DataFrame:
    """Evaluate predictions against ground truth and write a full report.

    The Python equivalent of the ``hires evaluate`` command. The following
    is written to ``output_dir``:

    ``detection_report.csv``
        Per-class metrics; see :func:`detection_report`.
    ``confusion_matrix.csv``, ``confusion_matrix.png``
        See :func:`confusion_matrix`.
    ``precision_recall.csv``, ``precision_recall.png``
        See :func:`precision_recall_curve`.
    ``matches.csv``
        One row per match with the traits of both objects; see
        :meth:`MatchResult.to_frame`.
    ``comparisons/{name}.png``
        The matches drawn on each image (``save_images``), for collections
        whose image is known; see :func:`hires.plot.comparison`.

    Parameters
    ----------
    gt, pred : Collection or Album
        Ground truth and predictions; see :func:`match`. Collections present
        in only one album are skipped with a logged warning.
    output_dir : path-like
        Directory for all outputs; created if needed.
    iou_threshold : float, default 0.5
        Minimum IoU for two polygons to be matched.
    class_names : mapping of int to str, optional
        ``{class_id: label}`` used in the tables and figures.
    save_images : bool, default True
        Write the comparison images.

    Returns
    -------
    pandas.DataFrame
        The detection report.

    Raises
    ------
    ValueError
        If two albums have no collection in common.

    Examples
    --------
    >>> gt = hires.read_album("labels/", image_dir="images/")  # doctest: +SKIP
    >>> pred = hires.read_album("results/", image_dir="images/")  # doctest: +SKIP
    >>> report = hires.evaluate(gt, pred, "evaluation/")  # doctest: +SKIP
    >>> report.loc["all", ["precision", "recall", "f1"]]  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    from . import plot

    iou_threshold = _check_threshold(iou_threshold)
    pairings, gt_only, pred_only = _pair_inputs(gt, pred)
    if not pairings:
        raise ValueError("gt and pred have no collection in common; nothing to evaluate.")
    for side, names in (("ground truth", gt_only), ("predictions", pred_only)):
        if names:
            logger.warning(
                "Skipped %d collection(s) found only in the %s: %s%s",
                len(names),
                side,
                ", ".join(names[:10]),
                ", ..." if len(names) > 10 else "",
            )

    result = _match_pairings(pairings, iou_threshold, False, gt_only, pred_only)
    report = _detection_report(pairings, iou_threshold, COCO_THRESHOLDS, class_names)
    matrix = _confusion_matrix(result, class_names, "background")
    curve = _precision_recall_curve(result, None)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "detection_report.csv")
    matrix.to_csv(output_dir / "confusion_matrix.csv")
    curve.to_csv(output_dir / "precision_recall.csv", index=False)
    result.to_frame(class_names=class_names).to_csv(output_dir / "matches.csv", index=False)
    plt.close(plot.confusion_matrix(matrix, save=output_dir / "confusion_matrix.png").figure)
    plt.close(plot.precision_recall(curve, save=output_dir / "precision_recall.png").figure)

    if save_images:
        drawn = 0
        for pairing in pairings:
            image_path = pairing.gt.image_path or pairing.pred.image_path
            if image_path is None:
                continue
            plot.comparison(
                image_path,
                result.filter(lambda m, name=pairing.name: m.image == name),
                save=output_dir / "comparisons" / f"{pairing.name or 'image'}.png",
            )
            drawn += 1
        if drawn < len(pairings):
            logger.info(
                "Comparison images written for %d of %d collections; the others have "
                "no image_path.",
                drawn,
                len(pairings),
            )

    logger.info("Evaluation of %d collection(s) written to %s", len(pairings), output_dir)
    return report
