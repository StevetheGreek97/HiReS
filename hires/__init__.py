"""HiReS: high-resolution image segmentation and morphometric analysis.

HiReS segments large images tile by tile with a YOLO model, stores the
detections as polygons, measures their shape, and evaluates predictions
against ground truth.

The package is organised like NumPy or SciPy: the everyday objects and
functions live in the top-level namespace, specialised functionality in
submodules.

Top-level namespace
-------------------
Data model
    `Annotation`, `Collection`, `Album`
Input
    `read_txt`, `read_album`, `read_class_names`
Segmentation
    `segment`, `Segmenter`, `run_pipeline`
Measurement
    `measure`, `TRAIT_NAMES`, `UNITS`
Evaluation
    `evaluate` (more in `hires.metrics`)
Class handling
    `ClassMapping`, `build_class_mapping`
Diagnostics
    `show_versions`

Submodules
----------
`hires.metrics`
    Matching predictions to ground truth and detection metrics.
`hires.plot`
    Overlays, comparison images and evaluation charts.
`hires.tiling`
    Splitting images and annotations into tiles and merging them back.
`hires.io`
    Low-level reading helpers.
`hires.analysis`
    Trait-agreement plots (legacy; scheduled for removal).

Examples
--------
>>> import hires
>>> collection = hires.segment("scan.tif", "model.pt")  # doctest: +SKIP
>>> collection.with_scale(dpi=2400, unit="um").to_csv("traits.csv")  # doctest: +SKIP

>>> gt = hires.read_album("labels/", image_dir="images/")  # doctest: +SKIP
>>> pred = hires.read_album("results/", image_dir="images/")  # doctest: +SKIP
>>> hires.metrics.detection_report(gt, pred)  # doctest: +SKIP
"""

from __future__ import annotations

import importlib
import logging
from importlib.metadata import PackageNotFoundError, version

from . import io, metrics, tiling
from .album import Album
from .annotation import TRAIT_NAMES, UNITS, Annotation, measure
from .classes import ClassMapping, build_class_mapping
from .collection import Collection
from .errors import (
    AnnotationFormatError,
    HiresError,
    HiresWarning,
    MissingDependencyError,
)
from .io import read_album, read_class_names, read_txt
from .metrics import evaluate
from .segmentation import Segmenter, run_pipeline, segment

try:
    __version__ = version("hireseg")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

__all__ = [
    "TRAIT_NAMES",
    "UNITS",
    "Album",
    "Annotation",
    "AnnotationFormatError",
    "ClassMapping",
    "Collection",
    "HiresError",
    "HiresWarning",
    "MissingDependencyError",
    "Segmenter",
    "__version__",
    "analysis",
    "build_class_mapping",
    "evaluate",
    "io",
    "measure",
    "metrics",
    "plot",
    "read_album",
    "read_class_names",
    "read_txt",
    "run_pipeline",
    "segment",
    "show_versions",
    "tiling",
]

# A library must not configure logging; applications opt in.
logging.getLogger(__name__).addHandler(logging.NullHandler())

_LAZY_SUBMODULES = {"analysis", "plot"}


def __getattr__(name: str):
    # ``plot`` and ``analysis`` pull in Matplotlib, so load them on first use.
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def show_versions() -> None:
    """Print the versions of HiReS and its dependencies.

    Include the output when reporting a bug.

    Examples
    --------
    >>> hires.show_versions()  # doctest: +SKIP
    hireseg      : 2.0.0
    python       : 3.12.3
    ...
    """
    import platform

    rows = [
        ("hireseg", __version__),
        ("python", platform.python_version()),
        ("platform", platform.platform()),
    ]
    for module in (
        "numpy", "pandas", "shapely", "PIL", "cv2", "matplotlib", "yaml", "ultralytics", "torch",
    ):  # fmt: skip
        try:
            found = getattr(importlib.import_module(module), "__version__", "unknown")
        except ImportError:
            found = "not installed"
        rows.append((module, found))
    for name, value in rows:
        print(f"{name:<12} : {value}")


def __dir__() -> list[str]:
    return sorted(__all__)
