"""Exceptions and warnings raised by HiReS."""

from __future__ import annotations

__all__ = [
    "AnnotationFormatError",
    "HiresError",
    "HiresWarning",
    "MissingDependencyError",
]


class HiresError(Exception):
    """Base class for all errors raised by HiReS."""


class AnnotationFormatError(HiresError, ValueError):
    """An annotation file or line could not be parsed."""


class MissingDependencyError(HiresError, ImportError):
    """An optional dependency required for the requested feature is missing."""


class HiresWarning(UserWarning):
    """Base class for warnings emitted by HiReS."""


def require(module: str, *, extra: str, feature: str):
    """Import an optional dependency or raise a helpful error.

    Parameters
    ----------
    module : str
        Importable module name, e.g. ``"ultralytics"``.
    extra : str
        Name of the ``hireseg`` extra that provides it.
    feature : str
        Short description of the feature that needs it, used in the message.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(
            f"{feature} requires the optional dependency '{module}'. "
            f"Install it with: pip install 'hireseg[{extra}]'"
        ) from exc
