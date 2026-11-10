"""The :class:`Annotation` value object and polygon trait measurement."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from shapely import affinity
from shapely.geometry import Polygon, box

__all__ = ["TRAIT_NAMES", "UNITS", "Annotation", "measure"]

#: Names of the morphometric traits returned by :func:`measure`, in order.
TRAIT_NAMES: tuple[str, ...] = (
    "area",
    "perimeter",
    "solidity",
    "convexity",
    "circularity",
    "bbox_width",
    "bbox_height",
    "obb_width",
    "obb_length",
)

#: Supported physical units, mapped to the number of units per inch.
UNITS: dict[str, float] = {
    "nm": 25_400_000.0,
    "um": 25_400.0,
    "mm": 25.4,
    "cm": 2.54,
    "m": 0.0254,
    "inch": 1.0,
}


def pixel_size_from_dpi(dpi: float, unit: str) -> float:
    """Return the edge length of one pixel in ``unit`` for a scan at ``dpi``."""
    if unit not in UNITS:
        raise ValueError(f"Unsupported unit {unit!r}; choose one of {sorted(UNITS)}.")
    if not dpi > 0:
        raise ValueError(f"dpi must be positive, got {dpi!r}.")
    return UNITS[unit] / float(dpi)


def measure(polygon: Polygon, *, pixel_size: float = 1.0) -> dict[str, float]:
    """Measure the morphometric traits of a polygon.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon in the coordinate system the measurements should be taken in
        (normally pixels).
    pixel_size : float, default 1.0
        Length of one coordinate unit in physical units. Lengths are
        multiplied by ``pixel_size`` and areas by ``pixel_size ** 2``.

    Returns
    -------
    dict of str to float
        One entry per name in :data:`TRAIT_NAMES`:

        ``area``, ``perimeter``
            Polygon area and exterior + interior boundary length.
        ``solidity``
            ``area / convex_hull_area``.
        ``convexity``
            ``convex_hull_perimeter / perimeter``.
        ``circularity``
            ``4 * pi * area / perimeter ** 2`` (1.0 for a circle).
        ``bbox_width``, ``bbox_height``
            Extent of the axis-aligned bounding box.
        ``obb_width``, ``obb_length``
            Short and long side of the minimum rotated bounding rectangle.

    Notes
    -----
    The shape ratios ``solidity``, ``convexity`` and ``circularity`` are
    dimensionless, but ``convexity`` and ``circularity`` are only meaningful
    when both axes share the same scale. Measure in pixel coordinates, not in
    normalized coordinates of a non-square image.

    Examples
    --------
    >>> from shapely.geometry import box
    >>> traits = measure(box(0, 0, 4, 2))
    >>> traits["area"], traits["obb_length"], traits["obb_width"]
    (8.0, 4.0, 2.0)
    """
    scale = float(pixel_size)
    area = polygon.area
    perimeter = polygon.length
    hull = polygon.convex_hull
    minx, miny, maxx, maxy = polygon.bounds
    obb_width, obb_length = _obb_sides(polygon)

    return {
        "area": area * scale**2,
        "perimeter": perimeter * scale,
        "solidity": area / hull.area if hull.area > 0 else 0.0,
        "convexity": hull.length / perimeter if perimeter > 0 else 0.0,
        "circularity": 4.0 * math.pi * area / perimeter**2 if perimeter > 0 else 0.0,
        "bbox_width": (maxx - minx) * scale,
        "bbox_height": (maxy - miny) * scale,
        "obb_width": obb_width * scale,
        "obb_length": obb_length * scale,
    }


def _obb_sides(polygon: Polygon) -> tuple[float, float]:
    """Return ``(short, long)`` side lengths of the minimum rotated rectangle."""
    rectangle = polygon.minimum_rotated_rectangle
    if not isinstance(rectangle, Polygon) or rectangle.is_empty:
        return math.nan, math.nan
    (x0, y0), (x1, y1), (x2, y2) = rectangle.exterior.coords[:3]
    first = math.hypot(x1 - x0, y1 - y0)
    second = math.hypot(x2 - x1, y2 - y1)
    return min(first, second), max(first, second)


@lru_cache(maxsize=32)
def _inset_unit_box(margin: float) -> Polygon:
    return box(margin, margin, 1.0 - margin, 1.0 - margin)


@dataclass(frozen=True)
class Annotation:
    """One segmented object: a class label, a polygon and a confidence.

    Annotations are immutable. Polygon coordinates are *normalized*: ``x`` and
    ``y`` are fractions of the image width and height, exactly as stored in
    YOLO segmentation label files.

    Parameters
    ----------
    class_id : int
        Integer class label.
    polygon : shapely.geometry.Polygon
        Non-empty polygon in normalized image coordinates.
    confidence : float, optional
        Detection confidence in ``[0, 1]``. ``None`` for ground-truth
        annotations that carry no score.

    See Also
    --------
    Collection : All annotations of one image.

    Examples
    --------
    >>> from shapely.geometry import box
    >>> ann = Annotation(0, box(0.1, 0.1, 0.3, 0.2), confidence=0.9)
    >>> ann.class_id
    0
    >>> ann.traits(image_size=(1000, 1000))["area"]
    20000.0
    """

    class_id: int
    polygon: Polygon
    confidence: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.class_id, bool) or int(self.class_id) != self.class_id:
            raise TypeError(f"class_id must be an integer, got {self.class_id!r}.")
        object.__setattr__(self, "class_id", int(self.class_id))
        if not isinstance(self.polygon, Polygon):
            raise TypeError(
                f"polygon must be a shapely Polygon, got {type(self.polygon).__name__}."
            )
        if self.polygon.is_empty:
            raise ValueError("polygon must not be empty.")
        if self.confidence is not None:
            confidence = float(self.confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"confidence must be in [0, 1], got {confidence!r}.")
            object.__setattr__(self, "confidence", confidence)

    def replace(self, **changes: Any) -> Annotation:
        """Return a copy with the given fields replaced.

        Examples
        --------
        >>> from shapely.geometry import box
        >>> Annotation(0, box(0, 0, 1, 1)).replace(class_id=3).class_id
        3
        """
        return dataclasses.replace(self, **changes)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounds ``(min_x, min_y, max_x, max_y)``, normalized."""
        return self.polygon.bounds

    def to_pixels(self, image_size: tuple[int, int]) -> Polygon:
        """Return the polygon scaled to pixel coordinates.

        Parameters
        ----------
        image_size : tuple of int
            ``(width, height)`` of the image in pixels.
        """
        width, height = image_size
        return affinity.scale(
            self.polygon, xfact=float(width), yfact=float(height), origin=(0, 0)
        )

    def traits(
        self,
        image_size: tuple[int, int] | None = None,
        *,
        pixel_size: float = 1.0,
    ) -> dict[str, float]:
        """Measure the morphometric traits of this annotation.

        Parameters
        ----------
        image_size : tuple of int, optional
            ``(width, height)`` of the image in pixels. When given, traits are
            measured in pixel space. When omitted they are measured on the
            normalized polygon, which is only meaningful for square images.
        pixel_size : float, default 1.0
            Physical size of one pixel, to convert lengths and areas.

        Returns
        -------
        dict of str to float
            See :func:`hires.measure`.
        """
        polygon = self.polygon if image_size is None else self.to_pixels(image_size)
        return measure(polygon, pixel_size=pixel_size)

    def iou(self, other: Annotation) -> float:
        """Intersection-over-union with another annotation's polygon."""
        from .metrics import iou

        return iou(self, other)

    def within_frame(self, margin: float = 0.0) -> bool:
        """Whether the polygon lies inside the image frame.

        Parameters
        ----------
        margin : float, default 0.0
            Normalized inset applied to every side of the frame. An annotation
            closer than ``margin`` to any image edge is not within the frame.

        Examples
        --------
        >>> from shapely.geometry import box
        >>> Annotation(0, box(0.2, 0.2, 0.4, 0.4)).within_frame(0.01)
        True
        >>> Annotation(0, box(0.0, 0.2, 0.4, 0.4)).within_frame(0.01)
        False
        """
        return self.polygon.is_valid and _inset_unit_box(float(margin)).contains(
            self.polygon
        )

    def __repr__(self) -> str:
        confidence = "None" if self.confidence is None else f"{self.confidence:.3f}"
        return (
            f"Annotation(class_id={self.class_id}, confidence={confidence}, "
            f"vertices={len(self.polygon.exterior.coords) - 1})"
        )
