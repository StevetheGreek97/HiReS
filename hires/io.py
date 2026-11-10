"""Reading annotations, class names and images from disk."""

from __future__ import annotations

import os
import threading
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from shapely.geometry import MultiPolygon, Polygon

from .album import Album
from .annotation import Annotation
from .collection import Collection
from .errors import AnnotationFormatError, HiresWarning, require

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "IMAGE_EXTENSIONS",
    "find_images",
    "read_album",
    "read_class_names",
    "read_image",
    "read_image_size",
    "read_txt",
]

#: Image file extensions recognised by HiReS (lower case, with dot).
IMAGE_EXTENSIONS: tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

OnError = Literal["warn", "raise", "ignore"]

_pil_lock = threading.Lock()


# ---------------------------------------------------------------------- #
# Images
# ---------------------------------------------------------------------- #
def read_image_size(path: str | os.PathLike[str]) -> tuple[int, int]:
    """Return ``(width, height)`` of an image without decoding its pixels.

    Works for scans larger than Pillow's decompression-bomb limit.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    from PIL import Image

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    # High-resolution scans routinely exceed PIL's bomb guard; lift it only
    # for the duration of this header read instead of process-wide.
    with _pil_lock:
        limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        try:
            with Image.open(path) as image:
                return image.size
        finally:
            Image.MAX_IMAGE_PIXELS = limit


def read_image(path: str | os.PathLike[str]) -> np.ndarray:
    """Read an image as an 8-bit BGR array of shape ``(height, width, 3)``.

    EXIF orientation is ignored so that pixel coordinates always match the
    stored raster and :func:`read_image_size`.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    OSError
        If the file cannot be decoded as an image.
    """
    import cv2

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise OSError(f"Could not decode image: {path}")
    return image


def find_images(
    source: str | os.PathLike[str],
    *,
    recursive: bool = False,
    extensions: Sequence[str] = IMAGE_EXTENSIONS,
) -> list[Path]:
    """List the image files at ``source``.

    Parameters
    ----------
    source : path-like
        An image file or a directory.
    recursive : bool, default False
        Descend into subdirectories.
    extensions : sequence of str, default :data:`IMAGE_EXTENSIONS`
        Accepted file extensions, case-insensitive.

    Returns
    -------
    list of pathlib.Path
        Sorted image paths; empty if ``source`` is a non-image file.

    Raises
    ------
    FileNotFoundError
        If ``source`` does not exist.
    """
    source = Path(source)
    allowed = {e.lower() for e in extensions}
    if source.is_file():
        return [source] if source.suffix.lower() in allowed else []
    if source.is_dir():
        pattern = "**/*" if recursive else "*"
        return sorted(
            p for p in source.glob(pattern) if p.is_file() and p.suffix.lower() in allowed
        )
    raise FileNotFoundError(f"No such file or directory: {source}")


# ---------------------------------------------------------------------- #
# Annotations
# ---------------------------------------------------------------------- #
def polygon_from_coords(coords: Sequence[tuple[float, float]]) -> Polygon | None:
    """Build a valid polygon from vertices, repairing it when possible.

    Self-intersecting rings are repaired with a zero-width buffer; if that
    splits the shape, the largest part is kept. Returns ``None`` when no
    polygon with positive area can be built.
    """
    if len(coords) < 3:
        return None
    try:
        polygon = Polygon(coords)
    except (ValueError, TypeError):
        return None
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if isinstance(polygon, MultiPolygon):
        if not polygon.geoms:
            return None
        polygon = max(polygon.geoms, key=lambda part: part.area)
    if not isinstance(polygon, Polygon) or polygon.is_empty or polygon.area == 0:
        return None
    return polygon


def parse_line(line: str) -> Annotation:
    """Parse one line of a YOLO segmentation label file.

    The format is ``class_id x1 y1 x2 y2 ... xn yn [confidence]``. A trailing
    value that makes the number of coordinates odd is read as the confidence.

    Raises
    ------
    AnnotationFormatError
        If the line is not a valid annotation.
    """
    tokens = line.split()
    if len(tokens) < 7:
        raise AnnotationFormatError("expected a class id and at least 3 points")
    try:
        class_id = int(tokens[0])
        values = [float(token) for token in tokens[1:]]
    except ValueError as exc:
        raise AnnotationFormatError(str(exc)) from None

    confidence = None
    if len(values) % 2:
        confidence = values.pop()
        if not 0.0 <= confidence <= 1.0:
            raise AnnotationFormatError(f"confidence {confidence} is outside [0, 1]")

    polygon = polygon_from_coords(list(zip(values[::2], values[1::2], strict=True)))
    if polygon is None:
        raise AnnotationFormatError("polygon has no area")
    return Annotation(class_id, polygon, confidence)


def read_txt(
    path: str | os.PathLike[str],
    *,
    name: str | None = None,
    image: str | os.PathLike[str] | None = None,
    image_size: tuple[int, int] | None = None,
    on_error: OnError = "warn",
) -> Collection:
    """Read a YOLO segmentation label file into a :class:`~hires.Collection`.

    Parameters
    ----------
    path : path-like
        Label file with one ``class_id x1 y1 ... xn yn [confidence]`` line per
        object, in normalized coordinates.
    name : str, optional
        Name of the collection. Defaults to the file stem.
    image : path-like, optional
        Image the labels belong to. Its size is read from disk so that
        traits can be measured in pixels.
    image_size : tuple of int, optional
        ``(width, height)`` in pixels, as an alternative to ``image``.
    on_error : {"warn", "raise", "ignore"}, default "warn"
        What to do with lines that cannot be parsed (wrong token count,
        non-numeric values, degenerate polygons): skip them with a
        :class:`~hires.HiresWarning`, raise
        :class:`~hires.AnnotationFormatError`, or skip them silently.

    Returns
    -------
    Collection

    Raises
    ------
    FileNotFoundError
        If ``path`` or ``image`` does not exist.

    See Also
    --------
    read_album : Read a whole directory of label files.
    Collection.to_txt : Write a label file.

    Examples
    --------
    >>> collection = hires.read_txt("labels/scan.txt", image="images/scan.tif")  # doctest: +SKIP
    >>> collection.to_frame()[["class_id", "area"]]  # doctest: +SKIP
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    if on_error not in ("warn", "raise", "ignore"):
        raise ValueError("on_error must be 'warn', 'raise' or 'ignore'.")

    annotations: list[Annotation] = []
    problems: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                annotations.append(parse_line(line))
            except AnnotationFormatError as exc:
                message = f"{path}:{number}: {exc}"
                if on_error == "raise":
                    raise AnnotationFormatError(message) from None
                problems.append(message)

    if problems and on_error == "warn":
        warnings.warn(
            f"Skipped {len(problems)} unreadable line(s) in {path.name} "
            f"(first: {problems[0]}).",
            HiresWarning,
            stacklevel=2,
        )

    if image is not None and image_size is None:
        image_size = read_image_size(image)

    return Collection(
        annotations,
        name=name or path.stem,
        image_path=image,
        image_size=image_size,
    )


def read_album(
    source: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    *,
    image_dir: str | os.PathLike[str] | None = None,
    name: str | None = None,
    image_extensions: Sequence[str] = IMAGE_EXTENSIONS,
    on_error: OnError = "warn",
) -> Album:
    """Read many YOLO label files into an :class:`~hires.Album`.

    Parameters
    ----------
    source : path-like or iterable of path-like
        A directory containing ``*.txt`` label files, or an explicit list of
        label files.
    image_dir : path-like, optional
        Directory with the images. Each label file ``{stem}.txt`` is bound to
        ``{image_dir}/{stem}{ext}`` when such a file exists.
    name : str, optional
        Name of the album. Defaults to the directory name.
    image_extensions : sequence of str, default :data:`IMAGE_EXTENSIONS`
        Extensions tried, in order, when looking for an image.
    on_error : {"warn", "raise", "ignore"}, default "warn"
        Handling of unreadable lines; see :func:`read_txt`.

    Returns
    -------
    Album
        One collection per label file, named after the file stem and sorted
        by file name when ``source`` is a directory.

    Raises
    ------
    FileNotFoundError
        If ``source`` or ``image_dir`` does not exist.

    Examples
    --------
    >>> gt = hires.read_album("labels/", image_dir="images/")  # doctest: +SKIP
    >>> gt.with_scale(dpi=2400, unit="um").to_csv("traits.csv")  # doctest: +SKIP
    """
    if isinstance(source, (str, os.PathLike)):
        directory = Path(source)
        if not directory.is_dir():
            raise FileNotFoundError(f"Annotation directory not found: {directory}")
        paths = sorted(directory.glob("*.txt"))
        name = name or directory.name
    else:
        paths = [Path(p) for p in source]

    if image_dir is not None:
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

    collections = []
    for path in paths:
        image = None
        if image_dir is not None:
            candidates = (image_dir / f"{path.stem}{ext}" for ext in image_extensions)
            image = next((c for c in candidates if c.is_file()), None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", HiresWarning)
            collections.append(read_txt(path, image=image, on_error=on_error))
        for warning in caught:
            warnings.warn(warning.message, warning.category, stacklevel=2)

    return Album(collections, name=name)


def read_class_names(path: str | os.PathLike[str]) -> dict[int, str]:
    """Read ``{class_id: label}`` from a dataset YAML or a YOLO weights file.

    Parameters
    ----------
    path : path-like
        A ``data.yaml`` with a ``names`` entry (list or mapping), or a
        ``.pt`` model file. Reading weights needs the ``yolo`` extra.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    KeyError
        If a YAML file has no ``names`` entry.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Class-name source not found: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if "names" not in data:
            raise KeyError(f"No 'names' entry in {path}.")
        names = data["names"]
    else:
        ultralytics = require("ultralytics", extra="yolo", feature="Reading model weights")
        names = ultralytics.YOLO(str(path)).names

    return normalize_class_names(names)


def normalize_class_names(names: object) -> dict[int, str]:
    """Coerce a list or mapping of class names to ``{int: str}``."""
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    raise TypeError(f"Unsupported class-name container: {type(names).__name__}.")
