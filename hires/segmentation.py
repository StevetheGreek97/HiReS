"""Tiled instance segmentation of high-resolution images."""

from __future__ import annotations

import datetime
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .album import Album
from .annotation import Annotation
from .collection import Collection
from .errors import require
from .io import find_images, normalize_class_names, polygon_from_coords, read_image
from .tiling import _as_size, iter_tiles, merge_tiles

if TYPE_CHECKING:
    import numpy as np

__all__ = ["Segmenter", "run_pipeline", "segment"]

logger = logging.getLogger(__name__)


def _to_list(values: Any) -> list:
    """Convert a tensor, array or sequence to a plain Python list."""
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


class Segmenter:
    """Segment large images tile by tile with a YOLO segmentation model.

    Each image is cut into overlapping tiles, the model is run on every tile,
    detections touching a tile border are discarded (the neighbouring tile
    sees them in full), the remaining detections are merged into full-image
    coordinates and duplicates from overlapping tiles are removed with
    polygon non-maximum suppression.

    Create one ``Segmenter`` and reuse it for many images; the model is
    loaded once.

    Parameters
    ----------
    model : path-like or model object
        Path to Ultralytics YOLO segmentation weights (``.pt``), or an
        already loaded model exposing ``predict(...)`` and ``names``. Loading
        from a path needs the ``yolo`` extra
        (``pip install 'hireseg[yolo]'``).
    tile_size : int or tuple of int, default 1024
        Tile ``(width, height)`` in pixels.
    overlap : int, default 150
        Pixels shared by neighbouring tiles. Should exceed the largest
        object, otherwise objects larger than the overlap can be lost at
        tile borders.
    confidence : float, default 0.5
        Minimum detection confidence.
    inference_size : int, default 1024
        Size the model resizes each tile to (Ultralytics ``imgsz``).
    device : str, default "cpu"
        Inference device, e.g. ``"cpu"``, ``"0"`` or ``"cuda:0"``.
    edge_margin : float, default 0.01
        Detections closer than this fraction of the tile (and of the full
        image) to a border are discarded. With ``0`` nothing is discarded,
        so objects cut by a tile border are kept as fragments.
    nms_iou : float, default 0.7
        Polygon IoU above which the less confident of two detections is
        removed.
    batch_size : int, default 8
        Number of tiles passed to the model at once.

    See Also
    --------
    segment : One-call convenience function.
    run_pipeline : Segment files and write all outputs to disk.

    Examples
    --------
    >>> segmenter = hires.Segmenter("DaphnAI.pt", overlap=300, device="0")  # doctest: +SKIP
    >>> collection = segmenter.segment("scan.tif")  # doctest: +SKIP
    >>> collection.with_scale(dpi=2400, unit="um").to_csv("scan_shapes.csv")  # doctest: +SKIP
    """

    def __init__(
        self,
        model: str | os.PathLike[str] | Any,
        *,
        tile_size: int | tuple[int, int] = 1024,
        overlap: int = 150,
        confidence: float = 0.5,
        inference_size: int = 1024,
        device: str = "cpu",
        edge_margin: float = 0.01,
        nms_iou: float = 0.7,
        batch_size: int = 8,
    ) -> None:
        self.tile_size = _as_size(tile_size)
        if overlap < 0 or overlap >= min(self.tile_size):
            raise ValueError("overlap must be non-negative and smaller than the tile size.")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1].")
        if not 0.0 <= nms_iou <= 1.0:
            raise ValueError("nms_iou must be in [0, 1].")
        if not 0.0 <= edge_margin < 0.5:
            raise ValueError("edge_margin must be in [0, 0.5).")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self.overlap = int(overlap)
        self.confidence = float(confidence)
        self.inference_size = int(inference_size)
        self.device = device
        self.edge_margin = float(edge_margin)
        self.nms_iou = float(nms_iou)
        self.batch_size = int(batch_size)

        if isinstance(model, (str, os.PathLike)):
            path = Path(model)
            if not path.is_file():
                raise FileNotFoundError(f"Model weights not found: {path}")
            ultralytics = require("ultralytics", extra="yolo", feature="Segmentation")
            self.model_path: Path | None = path
            self.model = ultralytics.YOLO(str(path))
        else:
            if not hasattr(model, "predict"):
                raise TypeError("model must be a path or an object with a predict() method.")
            self.model_path = None
            self.model = model

    @property
    def class_names(self) -> dict[int, str]:
        """``{class_id: label}`` as stored in the model."""
        return normalize_class_names(getattr(self.model, "names", {}))

    def parameters(self) -> dict[str, Any]:
        """Return the segmentation settings as a plain dict (e.g. for logging)."""
        return {
            "model": None if self.model_path is None else str(self.model_path),
            "tile_size": list(self.tile_size),
            "overlap": self.overlap,
            "confidence": self.confidence,
            "inference_size": self.inference_size,
            "device": self.device,
            "edge_margin": self.edge_margin,
            "nms_iou": self.nms_iou,
            "batch_size": self.batch_size,
        }

    def __repr__(self) -> str:
        return (
            f"Segmenter(model={str(self.model_path) if self.model_path else self.model!r}, "
            f"tile_size={self.tile_size}, overlap={self.overlap}, "
            f"confidence={self.confidence}, device={self.device!r})"
        )

    def predict_tiles(self, tiles: Sequence[np.ndarray]) -> list[list[Annotation]]:
        """Run the model on tile arrays.

        Parameters
        ----------
        tiles : sequence of numpy.ndarray
            8-bit BGR tiles of shape ``(height, width, 3)``.

        Returns
        -------
        list of list of Annotation
            The raw detections of every tile, normalized to the tile.
        """
        detections: list[list[Annotation]] = []
        for start in range(0, len(tiles), self.batch_size):
            batch = list(tiles[start : start + self.batch_size])
            results = self.model.predict(
                batch,
                conf=self.confidence,
                imgsz=self.inference_size,
                device=self.device,
                verbose=False,
            )
            detections.extend(_annotations_from_result(result) for result in results)
        return detections

    def segment(
        self,
        image: str | os.PathLike[str] | np.ndarray,
        *,
        name: str | None = None,
        debug_dir: str | os.PathLike[str] | None = None,
    ) -> Collection:
        """Segment one image.

        Parameters
        ----------
        image : path-like or numpy.ndarray
            Image file, or an 8-bit BGR array of shape ``(height, width, 3)``.
        name : str, optional
            Name of the returned collection. Defaults to the file stem.
        debug_dir : path-like, optional
            If given, intermediate results are written there: the tiles, the
            raw per-tile predictions (labels and overlays) and the merged
            detections before non-maximum suppression.

        Returns
        -------
        Collection
            The detections in full-image normalized coordinates, ordered by
            decreasing confidence, with ``image_size`` (and ``image_path``
            for file input) set.
        """
        if isinstance(image, (str, os.PathLike)):
            image_path: Path | None = Path(image)
            pixels = read_image(image_path)
            name = name or image_path.stem
        else:
            image_path = None
            pixels = image
            if pixels.ndim != 3 or pixels.shape[2] != 3:
                raise ValueError("An image array must have shape (height, width, 3).")
        height, width = pixels.shape[:2]

        tiles = list(iter_tiles(pixels, self.tile_size, self.overlap))
        logger.info("%s: %d x %d px, %d tiles", name or "image", width, height, len(tiles))
        raw = self.predict_tiles([tile.pixels for tile in tiles])

        per_tile = {
            (tile.x, tile.y): [a for a in found if a.within_frame(self.edge_margin)]
            for tile, found in zip(tiles, raw, strict=True)
        }
        merged = merge_tiles(
            per_tile, tile_size=self.tile_size, image_size=(width, height), name=name
        ).filter(lambda a: a.within_frame(self.edge_margin))
        result = merged.nms(self.nms_iou).replace(image_path=image_path)

        if debug_dir is not None:
            self._write_debug(Path(debug_dir), name or "image", tiles, raw, merged)

        logger.info(
            "%s: %d detections, class counts %s",
            name or "image",
            len(result),
            result.class_counts(),
        )
        return result

    def _write_debug(self, directory, name, tiles, raw, merged) -> None:
        import cv2

        from . import plot

        names = self.class_names
        for sub in ("tiles", "tile_labels", "tile_overlays"):
            (directory / sub).mkdir(parents=True, exist_ok=True)
        for tile, found in zip(tiles, raw, strict=True):
            stem = f"{name}_{tile.x}_{tile.y}"
            cv2.imwrite(str(directory / "tiles" / f"{stem}.png"), tile.pixels)
            Collection(found).to_txt(directory / "tile_labels" / f"{stem}.txt")
            if found:
                plot.overlay(
                    tile.pixels,
                    found,
                    class_names=names,
                    save=directory / "tile_overlays" / f"{stem}.png",
                )
        merged.to_txt(directory / f"{name}_merged.txt")
        logger.info("Debug files written to %s", directory)


def _annotations_from_result(result: Any) -> list[Annotation]:
    """Convert one Ultralytics ``Results`` object to annotations."""
    masks = getattr(result, "masks", None)
    if masks is None:
        return []
    classes = _to_list(result.boxes.cls)
    confidences = _to_list(result.boxes.conf)
    annotations = []
    for points, class_id, confidence in zip(masks.xyn, classes, confidences, strict=True):
        polygon = polygon_from_coords([(float(x), float(y)) for x, y in _to_list(points)])
        if polygon is not None:
            annotations.append(
                Annotation(int(class_id), polygon, min(max(float(confidence), 0.0), 1.0))
            )
    return annotations


def segment(
    image: str | os.PathLike[str] | np.ndarray,
    model: str | os.PathLike[str] | Any,
    **kwargs: Any,
) -> Collection:
    """Segment one high-resolution image.

    Convenience wrapper for ``Segmenter(model, **kwargs).segment(image)``.
    To process several images, create a :class:`Segmenter` once and reuse it
    so the model is loaded only once.

    Parameters
    ----------
    image : path-like or numpy.ndarray
        Image file, or an 8-bit BGR array.
    model : path-like or model object
        YOLO segmentation weights or a loaded model.
    **kwargs
        Tiling and inference settings; see :class:`Segmenter`.

    Returns
    -------
    Collection

    Examples
    --------
    >>> collection = hires.segment("scan.tif", "DaphnAI.pt", overlap=300)  # doctest: +SKIP
    >>> len(collection), collection.class_counts()  # doctest: +SKIP
    (87, {0: 61, 1: 26})
    """
    return Segmenter(model, **kwargs).segment(image)


def run_pipeline(
    source: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    model: str | os.PathLike[str] | Any,
    output_dir: str | os.PathLike[str] = "results",
    *,
    recursive: bool = False,
    dpi: float | None = None,
    unit: str | None = None,
    save_overlay: bool = True,
    save_crops: bool = False,
    debug: bool = False,
    on_error: Literal["raise", "skip"] = "raise",
    class_names: Mapping[int, str] | None = None,
    **kwargs: Any,
) -> Album:
    """Segment image files and write labels, tables and overlays to disk.

    The Python equivalent of the ``hires run`` command.

    For every image ``{stem}`` the following is written to ``output_dir``
    (mirroring the sub-directory layout of a directory source):

    ``{stem}.txt``
        Detections as YOLO segmentation labels with confidences.
    ``{stem}_shapes.csv``
        One row per detection with its morphometric traits.
    ``{stem}_annotated.tif``
        The image with the detections drawn on top (``save_overlay``).
    ``{stem}_crops/``
        One masked crop per detection (``save_crops``).
    ``{stem}_debug/``
        Intermediate results (``debug``).

    A ``run_config.yaml`` with the settings is written to ``output_dir``.

    Parameters
    ----------
    source : path-like or iterable of path-like
        An image file, a directory of images, or a list of either.
    model : path-like or model object
        YOLO segmentation weights or a loaded model.
    output_dir : path-like, default "results"
        Directory for all outputs.
    recursive : bool, default False
        Descend into sub-directories of a directory source.
    dpi : float, optional
        Scan resolution; together with ``unit`` it converts the traits in
        the CSV to physical units.
    unit : str, optional
        ``"nm"``, ``"um"``, ``"mm"``, ``"cm"``, ``"m"`` or ``"inch"``.
    save_overlay : bool, default True
        Write the annotated image.
    save_crops : bool, default False
        Write a masked crop of every detection.
    debug : bool, default False
        Write intermediate results.
    on_error : {"raise", "skip"}, default "raise"
        What to do when an image fails: stop, or log the error and continue
        with the next image.
    class_names : mapping of int to str, optional
        Labels for the overlay. Defaults to the names stored in the model.
    **kwargs
        Tiling and inference settings; see :class:`Segmenter`.

    Returns
    -------
    Album
        One collection per successfully processed image.

    Raises
    ------
    FileNotFoundError
        If a source does not exist or contains no images.

    Examples
    --------
    >>> album = hires.run_pipeline(  # doctest: +SKIP
    ...     "scans/", "DaphnAI.pt", "results/", dpi=2400, unit="um", device="0"
    ... )
    >>> album.to_frame().groupby("name")["area"].describe()  # doctest: +SKIP
    """
    import yaml

    from . import plot

    if (dpi is None) != (unit is None):
        raise ValueError("dpi and unit must be given together.")
    if on_error not in ("raise", "skip"):
        raise ValueError("on_error must be 'raise' or 'skip'.")

    sources = (
        [Path(source)]
        if isinstance(source, (str, os.PathLike))
        else [Path(s) for s in source]
    )
    jobs: list[tuple[Path, Path]] = []  # (image, output directory)
    output_dir = Path(output_dir)
    for root in sources:
        for image in find_images(root, recursive=recursive):
            target = output_dir / image.relative_to(root).parent if root.is_dir() else output_dir
            jobs.append((image, target))
    if not jobs:
        raise FileNotFoundError(
            f"No images found in {[str(s) for s in sources]}"
            + ("" if recursive else " (use recursive=True to search sub-directories)")
            + "."
        )

    segmenter = Segmenter(model, **kwargs)
    names = dict(class_names) if class_names is not None else segmenter.class_names

    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_images": len(jobs),
        "settings": {
            "source": [str(s) for s in sources],
            "output_dir": str(output_dir),
            "recursive": recursive,
            "dpi": dpi,
            "unit": unit,
            "save_overlay": save_overlay,
            "save_crops": save_crops,
            "debug": debug,
            **segmenter.parameters(),
        },
    }
    with (output_dir / "run_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    collections = []
    for number, (image, target) in enumerate(jobs, start=1):
        logger.info("[%d/%d] %s", number, len(jobs), image)
        stem = image.stem
        try:
            collection = segmenter.segment(
                image, debug_dir=target / f"{stem}_debug" if debug else None
            )
            if dpi is not None:
                collection = collection.with_scale(dpi, unit)
            collection.to_txt(target / f"{stem}.txt")
            collection.to_csv(target / f"{stem}_shapes.csv")
            if save_overlay:
                plot.overlay(
                    image, collection, class_names=names, save=target / f"{stem}_annotated.tif"
                )
            if save_crops:
                collection.save_crops(target / f"{stem}_crops", prefix=stem)
        except Exception:
            if on_error == "raise":
                raise
            logger.exception("[%d/%d] failed: %s", number, len(jobs), image)
            continue
        collections.append(collection)

    logger.info("Finished: %d of %d images -> %s", len(collections), len(jobs), output_dir)
    return Album(collections, name=output_dir.name)
