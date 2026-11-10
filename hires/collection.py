"""The :class:`Collection` container: all annotations of one image."""

from __future__ import annotations

import math
import os
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from .annotation import TRAIT_NAMES, UNITS, Annotation, pixel_size_from_dpi
from .errors import HiresWarning

if TYPE_CHECKING:
    import pandas as pd

    from .classes import ClassMapping

__all__ = ["Collection"]

_UNSET: Any = object()


class Collection(Sequence[Annotation]):
    """An immutable, ordered set of annotations belonging to one image.

    A collection behaves like a read-only sequence of
    :class:`~hires.Annotation` objects and carries the image metadata needed
    to turn normalized polygons into measurements. Every method that changes
    something returns a new collection.

    Parameters
    ----------
    annotations : iterable of Annotation, optional
        The annotations, in order.
    name : str, optional
        Identifier of the collection, usually the image file stem. Used to
        pair ground truth with predictions.
    image_path : path-like, optional
        Location of the image the annotations belong to.
    image_size : tuple of int, optional
        ``(width, height)`` of the image in pixels. Required for measurements
        in pixel or physical units.
    dpi : float, optional
        Scan resolution in dots per inch. Must be given together with `unit`.
    unit : str, optional
        Physical unit for measurements; one of ``"nm"``, ``"um"``, ``"mm"``,
        ``"cm"``, ``"m"``, ``"inch"``.

    See Also
    --------
    hires.read_txt : Load a collection from a YOLO segmentation label file.
    Album : A set of collections, one per image.

    Examples
    --------
    >>> from shapely.geometry import box
    >>> import hires
    >>> annotations = [
    ...     hires.Annotation(0, box(0.1, 0.1, 0.2, 0.2), confidence=0.9),
    ...     hires.Annotation(1, box(0.5, 0.5, 0.7, 0.6), confidence=0.6),
    ... ]
    >>> collection = hires.Collection(annotations, name="scan", image_size=(2000, 1000))
    >>> len(collection)
    2
    >>> collection.class_counts()
    {0: 1, 1: 1}
    >>> collection.filter(lambda a: a.confidence > 0.8)
    Collection(name='scan', n=1, image_size=(2000, 1000))
    """

    __slots__ = ("_annotations", "_dpi", "_image_path", "_image_size", "_name", "_unit")

    def __init__(
        self,
        annotations: Iterable[Annotation] = (),
        *,
        name: str | None = None,
        image_path: str | os.PathLike[str] | None = None,
        image_size: tuple[int, int] | None = None,
        dpi: float | None = None,
        unit: str | None = None,
    ) -> None:
        items = tuple(annotations)
        for item in items:
            if not isinstance(item, Annotation):
                raise TypeError(
                    f"Collection items must be Annotation objects, got {type(item).__name__}."
                )
        if (dpi is None) != (unit is None):
            raise ValueError("dpi and unit must be given together.")
        if unit is not None:
            pixel_size_from_dpi(dpi, unit)  # validates both
            if image_size is None:
                raise ValueError(
                    "image_size is required to measure in physical units; "
                    "pass image_size=(width, height) or load with an image."
                )
        if image_size is not None:
            width, height = image_size
            if not (width > 0 and height > 0):
                raise ValueError(f"image_size must be positive, got {image_size!r}.")
            image_size = (int(width), int(height))

        self._annotations = items
        self._name = None if name is None else str(name)
        self._image_path = None if image_path is None else Path(image_path)
        self._image_size = image_size
        self._dpi = None if dpi is None else float(dpi)
        self._unit = unit

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    @property
    def annotations(self) -> tuple[Annotation, ...]:
        """The annotations as a tuple."""
        return self._annotations

    @property
    def name(self) -> str | None:
        """Identifier of the collection, usually the image file stem."""
        return self._name

    @property
    def image_path(self) -> Path | None:
        """Path of the image the annotations belong to."""
        return self._image_path

    @property
    def image_size(self) -> tuple[int, int] | None:
        """``(width, height)`` of the image in pixels."""
        return self._image_size

    @property
    def dpi(self) -> float | None:
        """Scan resolution in dots per inch, if a physical scale is set."""
        return self._dpi

    @property
    def unit(self) -> str | None:
        """Physical measurement unit, if a physical scale is set."""
        return self._unit

    @property
    def pixel_size(self) -> float:
        """Edge length of one pixel in :attr:`unit` (1.0 without a scale)."""
        if self._dpi is None or self._unit is None:
            return 1.0
        return UNITS[self._unit] / self._dpi

    # ------------------------------------------------------------------ #
    # Sequence protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._annotations)

    def __iter__(self) -> Iterator[Annotation]:
        return iter(self._annotations)

    @overload
    def __getitem__(self, index: int) -> Annotation: ...
    @overload
    def __getitem__(self, index: slice) -> Collection: ...
    def __getitem__(self, index: int | slice) -> Annotation | Collection:
        if isinstance(index, slice):
            return self.replace(annotations=self._annotations[index])
        return self._annotations[index]

    def __add__(self, other: Iterable[Annotation]) -> Collection:
        if not isinstance(other, Iterable):
            return NotImplemented
        return self.replace(annotations=(*self._annotations, *other))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Collection):
            return NotImplemented
        return (
            self._annotations == other._annotations
            and self._name == other._name
            and self._image_path == other._image_path
            and self._image_size == other._image_size
            and self._dpi == other._dpi
            and self._unit == other._unit
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        parts = [f"name={self._name!r}", f"n={len(self)}"]
        if self._image_size is not None:
            parts.append(f"image_size={self._image_size}")
        if self._unit is not None:
            parts.append(f"dpi={self._dpi:g}, unit={self._unit!r}")
        return f"Collection({', '.join(parts)})"

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_txt(cls, path: str | os.PathLike[str], **kwargs: Any) -> Collection:
        """Load a collection from a YOLO segmentation label file.

        Alias of :func:`hires.read_txt`; see there for the parameters.
        """
        from .io import read_txt

        return read_txt(path, **kwargs)

    def replace(
        self,
        *,
        annotations: Iterable[Annotation] = _UNSET,
        name: str | None = _UNSET,
        image_path: str | os.PathLike[str] | None = _UNSET,
        image_size: tuple[int, int] | None = _UNSET,
        dpi: float | None = _UNSET,
        unit: str | None = _UNSET,
    ) -> Collection:
        """Return a copy with the given attributes replaced.

        Examples
        --------
        >>> Collection(name="a").replace(name="b")
        Collection(name='b', n=0)
        """
        return Collection(
            self._annotations if annotations is _UNSET else annotations,
            name=self._name if name is _UNSET else name,
            image_path=self._image_path if image_path is _UNSET else image_path,
            image_size=self._image_size if image_size is _UNSET else image_size,
            dpi=self._dpi if dpi is _UNSET else dpi,
            unit=self._unit if unit is _UNSET else unit,
        )

    def with_image(self, image_path: str | os.PathLike[str]) -> Collection:
        """Return a copy bound to an image, reading its size from disk.

        Parameters
        ----------
        image_path : path-like
            Image file the annotations belong to.

        Raises
        ------
        FileNotFoundError
            If the image does not exist.
        """
        from .io import read_image_size

        return self.replace(image_path=image_path, image_size=read_image_size(image_path))

    def with_scale(self, dpi: float | None, unit: str | None) -> Collection:
        """Return a copy that measures in physical units.

        Parameters
        ----------
        dpi : float or None
            Scan resolution in dots per inch.
        unit : str or None
            One of ``"nm"``, ``"um"``, ``"mm"``, ``"cm"``, ``"m"``, ``"inch"``.
            Pass ``None`` for both arguments to remove the scale.

        Raises
        ------
        ValueError
            If the collection has no ``image_size``, or the unit is unknown.

        Examples
        --------
        >>> from shapely.geometry import box
        >>> import hires
        >>> c = hires.Collection(
        ...     [hires.Annotation(0, box(0, 0, 0.5, 0.5))], image_size=(2400, 2400)
        ... )
        >>> c.with_scale(dpi=2400, unit="inch").to_frame()["area"].item()
        0.25
        """
        return self.replace(dpi=dpi, unit=unit)

    # ------------------------------------------------------------------ #
    # Queries and transformations
    # ------------------------------------------------------------------ #
    def class_counts(self) -> dict[int, int]:
        """Count the annotations per class id, sorted by class id."""
        counts = Counter(annotation.class_id for annotation in self._annotations)
        return dict(sorted(counts.items()))

    def filter(self, predicate: Callable[[Annotation], bool]) -> Collection:
        """Return the annotations for which ``predicate`` is true.

        Parameters
        ----------
        predicate : callable
            Function taking an :class:`~hires.Annotation` and returning a bool.

        Examples
        --------
        >>> confident = collection.filter(lambda a: a.confidence >= 0.8)  # doctest: +SKIP
        """
        return self.replace(annotations=[a for a in self._annotations if predicate(a)])

    def filter_confidence(self, minimum: float) -> Collection:
        """Return the annotations with a confidence of at least ``minimum``.

        Annotations without a confidence (ground truth) are kept.

        Examples
        --------
        >>> confident = collection.filter_confidence(0.8)  # doctest: +SKIP
        """
        if not 0.0 <= minimum <= 1.0:
            raise ValueError(f"minimum must be in [0, 1], got {minimum!r}.")
        return self.filter(lambda a: a.confidence is None or a.confidence >= minimum)

    def filter_area(
        self, min_area: float | None = None, max_area: float | None = None
    ) -> Collection:
        """Return the annotations whose area lies in ``[min_area, max_area]``.

        Removes debris (too small) and clumps (too large). Areas are in the
        units of :meth:`to_frame`: square pixels, or square physical units
        after :meth:`with_scale`; normalized when the image size is unknown.

        Parameters
        ----------
        min_area, max_area : float, optional
            Inclusive bounds; ``None`` leaves that side open.

        Examples
        --------
        >>> scaled = collection.with_scale(dpi=2400, unit="um")  # doctest: +SKIP
        >>> animals = scaled.filter_area(min_area=5e4)  # doctest: +SKIP
        """
        if min_area is not None and max_area is not None and min_area > max_area:
            raise ValueError("min_area must not exceed max_area.")
        scale = self.pixel_size**2
        if self._image_size is not None:
            scale *= self._image_size[0] * self._image_size[1]
        low = -math.inf if min_area is None else min_area
        high = math.inf if max_area is None else max_area
        return self.filter(lambda a: low <= a.polygon.area * scale <= high)

    def keep_classes(self, class_ids: int | Iterable[int]) -> Collection:
        """Return only the annotations of the given class id(s)."""
        keep = _as_id_set(class_ids)
        return self.filter(lambda a: a.class_id in keep)

    def drop_classes(self, class_ids: int | Iterable[int]) -> Collection:
        """Return the collection without the given class id(s)."""
        drop = _as_id_set(class_ids)
        return self.filter(lambda a: a.class_id not in drop)

    def remap_classes(
        self,
        mapping: ClassMapping | Mapping[int, int],
        resolve: Mapping[str, str] | None = None,
    ) -> Collection:
        """Return a copy with class ids translated through ``mapping``.

        Parameters
        ----------
        mapping : ClassMapping or mapping of int to int
            Either a :class:`~hires.ClassMapping` built with
            :func:`hires.build_class_mapping`, or a plain ``{old_id: new_id}``
            dict. Ids missing from a plain dict are left unchanged.
        resolve : mapping of str to str, optional
            ``{old_label: chosen_new_label}`` for every ambiguous entry of a
            :class:`~hires.ClassMapping`.
        """
        from .classes import ClassMapping

        if isinstance(mapping, ClassMapping):
            flat = mapping.flatten(resolve or {})
        else:
            flat = dict(mapping)
        return self.replace(
            annotations=[
                a.replace(class_id=flat.get(a.class_id, a.class_id))
                for a in self._annotations
            ]
        )

    def nms(self, iou_threshold: float = 0.5, *, class_aware: bool = False) -> Collection:
        """Apply polygon non-maximum suppression.

        Annotations are visited in order of decreasing confidence (ties are
        broken in favour of the larger polygon). Each visited annotation
        suppresses every remaining annotation whose polygon IoU with it
        exceeds ``iou_threshold``.

        Parameters
        ----------
        iou_threshold : float, default 0.5
            Annotations overlapping a kept annotation by more than this IoU
            are removed.
        class_aware : bool, default False
            If true, only annotations of the same class suppress each other.

        Returns
        -------
        Collection
            The kept annotations, ordered by decreasing confidence.

        Raises
        ------
        ValueError
            If any annotation has no confidence, or the threshold is outside
            ``[0, 1]``.
        """
        from shapely.strtree import STRtree

        from .metrics import iou

        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1.")
        if any(a.confidence is None for a in self._annotations):
            raise ValueError("All annotations need a confidence for NMS.")
        if not self._annotations:
            return self

        ordered = sorted(
            self._annotations,
            key=lambda a: (a.confidence, a.polygon.area),
            reverse=True,
        )
        tree = STRtree([a.polygon for a in ordered])
        suppressed: set[int] = set()
        kept: list[Annotation] = []

        for i, annotation in enumerate(ordered):
            if i in suppressed:
                continue
            kept.append(annotation)
            for j in tree.query(annotation.polygon):
                j = int(j)
                if j <= i or j in suppressed:
                    continue
                other = ordered[j]
                if class_aware and other.class_id != annotation.class_id:
                    continue
                if iou(annotation, other) > iou_threshold:
                    suppressed.add(j)

        return self.replace(annotations=kept)

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    def to_records(self) -> list[dict[str, Any]]:
        """Return one flat dict per annotation, including morphometric traits.

        Traits are measured in pixels when :attr:`image_size` is known, and
        in physical units when a scale is set with :meth:`with_scale`.
        Without an image size they are measured on the normalized polygons
        and a :class:`~hires.HiresWarning` is emitted.
        """
        if self._image_size is None and self._annotations:
            warnings.warn(
                f"Collection {self._name!r} has no image_size; traits are measured "
                "in normalized coordinates. Load it with an image to measure in pixels.",
                HiresWarning,
                stacklevel=2,
            )
        base = {
            "name": self._name,
            "image_path": None if self._image_path is None else str(self._image_path),
            "image_width": None if self._image_size is None else self._image_size[0],
            "image_height": None if self._image_size is None else self._image_size[1],
            "dpi": self._dpi,
            "unit": self._unit,
        }
        pixel_size = self.pixel_size
        return [
            {
                **base,
                "class_id": a.class_id,
                "confidence": a.confidence,
                **a.traits(self._image_size, pixel_size=pixel_size),
            }
            for a in self._annotations
        ]

    def to_frame(self) -> pd.DataFrame:
        """Return the annotations and their traits as a ``pandas.DataFrame``.

        One row per annotation with the columns ``name``, ``image_path``,
        ``image_width``, ``image_height``, ``dpi``, ``unit``, ``class_id``,
        ``confidence`` followed by the traits in :data:`hires.TRAIT_NAMES`.
        """
        import pandas as pd

        columns = [
            "name",
            "image_path",
            "image_width",
            "image_height",
            "dpi",
            "unit",
            "class_id",
            "confidence",
            *TRAIT_NAMES,
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", HiresWarning)
            records = self.to_records()
        for warning in caught:
            warnings.warn(warning.message, warning.category, stacklevel=2)
        return pd.DataFrame(records, columns=columns)

    def to_csv(self, path: str | os.PathLike[str], **kwargs: Any) -> None:
        """Write :meth:`to_frame` to a CSV file.

        Parameters
        ----------
        path : path-like
            Destination file; parent directories are created.
        **kwargs
            Forwarded to :meth:`pandas.DataFrame.to_csv`. ``index`` defaults
            to ``False``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs.setdefault("index", False)
        self.to_frame().to_csv(path, **kwargs)

    def to_txt(self, path: str | os.PathLike[str], *, confidence: bool = True) -> None:
        """Write the collection as a YOLO segmentation label file.

        Each line is ``class_id x1 y1 x2 y2 ... xn yn [confidence]`` with
        normalized coordinates. Only polygon exteriors are written; interior
        holes cannot be represented in this format.

        Parameters
        ----------
        path : path-like
            Destination file; parent directories are created.
        confidence : bool, default True
            Append the confidence to each line when the annotation has one.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for a in self._annotations:
            coords = list(a.polygon.exterior.coords)[:-1]
            line = f"{a.class_id} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in coords)
            if confidence and a.confidence is not None:
                line += f" {a.confidence:.6f}"
            lines.append(line)
        path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")

    def save_crops(
        self,
        directory: str | os.PathLike[str],
        *,
        mask: bool = True,
        padding: int = 0,
        prefix: str | None = None,
        format: str = "png",
    ) -> list[Path]:
        """Save one cropped image per annotation.

        Parameters
        ----------
        directory : path-like
            Output directory; created if needed.
        mask : bool, default True
            Black out the pixels outside the polygon.
        padding : int, default 0
            Extra pixels added around each bounding box.
        prefix : str, optional
            File name prefix. Defaults to the collection name.
        format : str, default "png"
            Image file extension understood by OpenCV.

        Returns
        -------
        list of pathlib.Path
            The written files, named ``{prefix}_{index:04d}_class_{class_id}``.

        Raises
        ------
        ValueError
            If the collection has no ``image_path``.
        """
        import cv2
        import numpy as np

        from .io import read_image

        if self._image_path is None:
            raise ValueError("The collection has no image_path; use with_image() first.")
        image = read_image(self._image_path)
        height, width = image.shape[:2]

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        prefix = prefix or self._name or "crop"
        extension = format.lstrip(".")
        saved: list[Path] = []

        for index, annotation in enumerate(self._annotations):
            polygon = annotation.to_pixels((width, height))
            min_x, min_y, max_x, max_y = polygon.bounds
            x0 = max(0, math.floor(min_x) - padding)
            y0 = max(0, math.floor(min_y) - padding)
            x1 = min(width, math.ceil(max_x) + padding)
            y1 = min(height, math.ceil(max_y) + padding)
            if x0 >= x1 or y0 >= y1:
                continue

            crop = image[y0:y1, x0:x1].copy()
            if mask:
                points = np.asarray(polygon.exterior.coords, dtype=np.float64)
                points -= (x0, y0)
                stencil = np.zeros(crop.shape[:2], dtype=np.uint8)
                cv2.fillPoly(stencil, [np.round(points).astype(np.int32)], 255)
                crop[stencil == 0] = 0

            path = directory / f"{prefix}_{index:04d}_class_{annotation.class_id}.{extension}"
            if not cv2.imwrite(str(path), crop):
                raise OSError(f"Could not write crop to {path}.")
            saved.append(path)

        return saved


def _as_id_set(class_ids: int | Iterable[int]) -> set[int]:
    if isinstance(class_ids, int):
        return {class_ids}
    return {int(class_id) for class_id in class_ids}
