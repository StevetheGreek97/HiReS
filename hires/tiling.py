"""Splitting large images and their annotations into overlapping tiles.

A tile is identified by the pixel coordinates ``(x, y)`` of its top-left
corner in the full image. Tiles written to disk are called
``{stem}_{x}_{y}.png``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from shapely import affinity

from .annotation import Annotation
from .collection import Collection

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "Tile",
    "iter_tiles",
    "merge_tiles",
    "parse_tile_name",
    "tile_annotations",
    "tile_grid",
    "tile_image",
]


class Tile(NamedTuple):
    """One tile cut from an image.

    Attributes
    ----------
    x, y : int
        Pixel coordinates of the top-left corner in the full image.
    pixels : numpy.ndarray
        Tile content, zero-padded to the full tile size at the image border.
    """

    x: int
    y: int
    pixels: np.ndarray


def _as_size(size: int | tuple[int, int], what: str = "tile_size") -> tuple[int, int]:
    if isinstance(size, int):
        size = (size, size)
    width, height = (int(v) for v in size)
    if width <= 0 or height <= 0:
        raise ValueError(f"{what} must be positive, got {size!r}.")
    return width, height


def _origins(length: int, tile: int, overlap: int) -> list[int]:
    """Tile origins along one axis: step by ``tile - overlap`` until covered."""
    origins = [0]
    while origins[-1] + tile < length:
        origins.append(origins[-1] + tile - overlap)
    return origins


def tile_grid(
    image_size: tuple[int, int],
    tile_size: int | tuple[int, int] = 1024,
    overlap: int = 150,
) -> list[tuple[int, int]]:
    """Compute the top-left corners of the tiles covering an image.

    Tiles are placed every ``tile_size - overlap`` pixels, starting at the
    origin, until the image is covered. The last tile along each axis may
    extend past the image border.

    Parameters
    ----------
    image_size : tuple of int
        ``(width, height)`` of the image in pixels.
    tile_size : int or tuple of int, default 1024
        Tile ``(width, height)`` in pixels; a single int means a square tile.
    overlap : int, default 150
        Pixels shared by neighbouring tiles. Choose it larger than the
        largest object so every object lies fully inside at least one tile.

    Returns
    -------
    list of tuple of int
        ``(x, y)`` corners, column by column.

    Raises
    ------
    ValueError
        If ``overlap`` is negative or not smaller than the tile size.

    Examples
    --------
    >>> tile_grid((2500, 900), tile_size=1024, overlap=150)
    [(0, 0), (874, 0), (1748, 0)]
    """
    width, height = _as_size(image_size, "image_size")
    tile_w, tile_h = _as_size(tile_size)
    if overlap < 0 or overlap >= min(tile_w, tile_h):
        raise ValueError("overlap must be non-negative and smaller than the tile size.")
    return [
        (x, y)
        for x in _origins(width, tile_w, overlap)
        for y in _origins(height, tile_h, overlap)
    ]


def iter_tiles(
    image: np.ndarray,
    tile_size: int | tuple[int, int] = 1024,
    overlap: int = 150,
) -> Iterator[Tile]:
    """Yield the overlapping tiles of an image array.

    Parameters
    ----------
    image : numpy.ndarray
        Image of shape ``(height, width)`` or ``(height, width, channels)``.
    tile_size : int or tuple of int, default 1024
        Tile ``(width, height)`` in pixels.
    overlap : int, default 150
        Pixels shared by neighbouring tiles.

    Yields
    ------
    Tile
        ``(x, y, pixels)``. Every tile has the full tile size; areas beyond
        the image border are zero.

    See Also
    --------
    tile_grid : The tile positions without pixel data.
    """
    import numpy as np

    height, width = image.shape[:2]
    tile_w, tile_h = _as_size(tile_size)
    for x, y in tile_grid((width, height), (tile_w, tile_h), overlap):
        region = image[y : y + tile_h, x : x + tile_w]
        if region.shape[:2] != (tile_h, tile_w):
            padded = np.zeros((tile_h, tile_w, *image.shape[2:]), dtype=image.dtype)
            padded[: region.shape[0], : region.shape[1]] = region
            region = padded
        yield Tile(x, y, region)


def tile_image(
    image_path: str | os.PathLike[str],
    directory: str | os.PathLike[str],
    *,
    tile_size: int | tuple[int, int] = 1024,
    overlap: int = 150,
) -> list[Path]:
    """Cut an image file into overlapping tiles and save them as PNG.

    Parameters
    ----------
    image_path : path-like
        Image to tile. Bit depth and channels are preserved.
    directory : path-like
        Output directory; created if needed.
    tile_size : int or tuple of int, default 1024
        Tile ``(width, height)`` in pixels.
    overlap : int, default 150
        Pixels shared by neighbouring tiles.

    Returns
    -------
    list of pathlib.Path
        The written files, named ``{stem}_{x}_{y}.png``.

    Raises
    ------
    FileNotFoundError
        If the image does not exist.
    OSError
        If the image cannot be decoded or a tile cannot be written.
    """
    import cv2

    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise OSError(f"Could not decode image: {image_path}")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for tile in iter_tiles(image, tile_size, overlap):
        path = directory / f"{image_path.stem}_{tile.x}_{tile.y}.png"
        if not cv2.imwrite(str(path), tile.pixels):
            raise OSError(f"Could not write tile to {path}.")
        written.append(path)
    return written


_TILE_NAME = re.compile(r"^(?P<stem>.*)_(?P<x>\d+)_(?P<y>\d+)$")


def parse_tile_name(name: str | os.PathLike[str]) -> tuple[str, int, int]:
    """Split a tile file name into ``(stem, x, y)``.

    Examples
    --------
    >>> parse_tile_name("scan_01_874_1748.png")
    ('scan_01', 874, 1748)
    """
    found = _TILE_NAME.match(Path(name).stem)
    if found is None:
        raise ValueError(f"{str(name)!r} is not a tile name of the form '{{stem}}_{{x}}_{{y}}'.")
    return found["stem"], int(found["x"]), int(found["y"])


def tile_annotations(
    collection: Collection,
    *,
    tile_size: int | tuple[int, int] = 1024,
    overlap: int = 150,
    image_size: tuple[int, int] | None = None,
) -> dict[tuple[int, int], Collection]:
    """Split the annotations of an image into per-tile collections.

    The counterpart of :func:`tile_image` for labels, e.g. to build a tiled
    training set. An annotation is assigned to every tile its bounding box
    overlaps and is *not* clipped, so coordinates of objects crossing a tile
    border fall outside ``[0, 1]``.

    Parameters
    ----------
    collection : Collection
        Annotations of the full image.
    tile_size : int or tuple of int, default 1024
        Tile ``(width, height)`` in pixels; use the same value as for the
        images.
    overlap : int, default 150
        Pixels shared by neighbouring tiles.
    image_size : tuple of int, optional
        ``(width, height)`` of the full image. Defaults to
        ``collection.image_size``.

    Returns
    -------
    dict
        ``{(x, y): Collection}`` for every tile of the grid, including empty
        ones. Each collection is named ``{name}_{x}_{y}`` and its polygons
        are normalized to the tile.

    Raises
    ------
    ValueError
        If the image size is unknown.

    Examples
    --------
    >>> tiles = hires.tiling.tile_annotations(collection, tile_size=1024)  # doctest: +SKIP
    >>> for tile in tiles.values():  # doctest: +SKIP
    ...     tile.to_txt(f"labels/{tile.name}.txt")
    """
    image_size = image_size or collection.image_size
    if image_size is None:
        raise ValueError("image_size is required to tile annotations.")
    width, height = image_size
    tile_w, tile_h = _as_size(tile_size)
    pixel_polygons = [(a, a.to_pixels(image_size)) for a in collection]

    tiles: dict[tuple[int, int], Collection] = {}
    for x, y in tile_grid((width, height), (tile_w, tile_h), overlap):
        members = []
        for annotation, polygon in pixel_polygons:
            min_x, min_y, max_x, max_y = polygon.bounds
            if max_x > x and min_x < x + tile_w and max_y > y and min_y < y + tile_h:
                local = affinity.affine_transform(
                    polygon, [1 / tile_w, 0, 0, 1 / tile_h, -x / tile_w, -y / tile_h]
                )
                members.append(annotation.replace(polygon=local))
        name = None if collection.name is None else f"{collection.name}_{x}_{y}"
        tiles[(x, y)] = Collection(members, name=name, image_size=(tile_w, tile_h))
    return tiles


def merge_tiles(
    tiles: Mapping[tuple[int, int], Iterable[Annotation]],
    *,
    tile_size: int | tuple[int, int],
    image_size: tuple[int, int],
    name: str | None = None,
) -> Collection:
    """Combine per-tile annotations into full-image coordinates.

    The inverse coordinate transform of :func:`tile_annotations`. Objects
    seen in several overlapping tiles appear several times in the result;
    remove the duplicates with :meth:`Collection.nms`.

    Parameters
    ----------
    tiles : mapping
        ``{(x, y): annotations}`` with polygons normalized to their tile.
    tile_size : int or tuple of int
        Tile ``(width, height)`` in pixels.
    image_size : tuple of int
        ``(width, height)`` of the full image in pixels.
    name : str, optional
        Name of the merged collection.

    Returns
    -------
    Collection
        All annotations, normalized to the full image.
    """
    width, height = _as_size(image_size, "image_size")
    tile_w, tile_h = _as_size(tile_size)
    scale_x, scale_y = tile_w / width, tile_h / height

    merged = []
    for (x, y), members in tiles.items():
        matrix = [scale_x, 0, 0, scale_y, x / width, y / height]
        for annotation in members:
            merged.append(
                annotation.replace(polygon=affinity.affine_transform(annotation.polygon, matrix))
            )
    return Collection(merged, name=name, image_size=(width, height))
