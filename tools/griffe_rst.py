"""Griffe extension that makes numpydoc/reST docstrings render well in MkDocs.

The HiReS docstrings follow the numpydoc standard and use a little
reStructuredText: cross-reference roles (``:func:`hires.measure```),
definition lists, ``See Also`` sections and ``::`` literal blocks.
mkdocstrings renders docstrings as Markdown, so this extension rewrites
those constructs before they are parsed:

* roles become mkdocstrings cross-references, resolved to the *public* path
  of the object (``hires.Collection``, not ``hires.collection.Collection``);
* ``See Also`` entries become a bullet list of cross-references;
* definition lists whose terms are literals become Markdown definition lists;
* a trailing ``::`` becomes ``:``.

It is loaded from ``mkdocs.yml`` and is not part of the installed package.
"""

from __future__ import annotations

import re
from typing import Any

import griffe

__all__ = ["RstToMarkdown", "translate"]

# Names re-exported from the top-level ``hires`` namespace.
_TOP_LEVEL = frozenset(
    {
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
        "build_class_mapping",
        "evaluate",
        "measure",
        "read_album",
        "read_class_names",
        "read_txt",
        "run_pipeline",
        "segment",
        "show_versions",
    }
)

# Submodules that are documented under their own name.
_PUBLIC_MODULES = ("hires.metrics", "hires.plot", "hires.tiling", "hires.io", "hires.analysis")

# First path segment of objects that live in other projects.
_EXTERNAL = frozenset({"pandas", "numpy", "shapely", "matplotlib", "pathlib", "PIL", "cv2"})

_ROLE = re.compile(
    r":(?:py:)?(?P<role>func|class|meth|attr|data|mod|exc|obj):"
    r"`(?P<short>~?)(?P<target>[^`<>]+)`"
)
_SEE_ALSO = re.compile(
    r"^(?P<indent>[ ]*)See Also\n(?P=indent)-{3,}\n(?P<body>(?:(?P=indent)\S[^\n]*(?:\n|$))+)",
    re.MULTILINE,
)
_DEFINITION = re.compile(
    r"^(?P<indent>[ ]*)(?P<term>``[^\n]+)\n(?P<body>(?:(?P=indent)[ ]{4}\S[^\n]*(?:\n|$))+)",
    re.MULTILINE,
)
_LITERAL_BLOCK = re.compile(r"::[ ]*$", re.MULTILINE)


def _resolve(target: str, role: str, module: str, owner: str | None) -> str:
    """Return the public dotted path a reference points to."""
    first = target.split(".", 1)[0]
    if first == "hires" or first in _EXTERNAL:
        return target
    if first in _TOP_LEVEL:
        return f"hires.{target}"
    if role in ("meth", "attr") and owner is not None and "." not in target:
        return f"{owner}.{target}"
    return f"{module}.{target}"


def translate(text: str, *, module: str = "hires", owner: str | None = None) -> str:
    """Rewrite the reStructuredText constructs of a docstring as Markdown.

    Parameters
    ----------
    text : str
        The raw docstring.
    module : str, default "hires"
        Public path of the module the docstring belongs to. Unqualified
        references are resolved relative to it.
    owner : str, optional
        Public path of the class the docstring belongs to, used to resolve
        unqualified method and attribute references.

    Returns
    -------
    str
        The docstring with Markdown cross-references and lists.

    Examples
    --------
    >>> translate("See :func:`match` and :class:`~hires.Collection`.", module="hires.metrics")
    'See [`match`][hires.metrics.match] and [`Collection`][hires.Collection].'
    """

    def link(target: str, role: str = "obj", short: bool = False) -> str:
        target = target.strip()
        label = target.rsplit(".", 1)[-1] if short else target
        return f"[`{label}`][{_resolve(target, role, module, owner)}]"

    def see_also(found: re.Match[str]) -> str:
        indent = found["indent"]
        lines = [f"{indent}See Also", f"{indent}--------"]
        for entry in found["body"].splitlines():
            name, _, description = entry.strip().partition(" : ")
            names = ", ".join(link(part) for part in name.split(","))
            suffix = f": {description}" if description else ""
            lines.append(f"{indent}- {names}{suffix}")
        return "\n".join(lines) + "\n"

    def definition(found: re.Match[str]) -> str:
        indent = found["indent"]
        first, *rest = (line[len(indent) + 4 :] for line in found["body"].splitlines())
        lines = [f"{indent}{found['term']}", f"{indent}:   {first}"]
        lines += [f"{indent}    {line}" for line in rest]
        return "\n" + "\n".join(lines) + "\n\n"

    text = _SEE_ALSO.sub(see_also, text)
    text = _DEFINITION.sub(definition, text)
    text = _ROLE.sub(lambda m: link(m["target"], m["role"], bool(m["short"])), text)
    return _LITERAL_BLOCK.sub(":", text)


def _public_module(path: str) -> str:
    for module in _PUBLIC_MODULES:
        if path == module or path.startswith(f"{module}."):
            return module
    return "hires"


class RstToMarkdown(griffe.Extension):
    """Translate reStructuredText in the docstrings of ``hires`` to Markdown."""

    def on_instance(self, *, obj: griffe.Object, **kwargs: Any) -> None:
        """Rewrite the docstring of a freshly created object."""
        if obj.docstring is None:
            return
        try:
            module = _public_module(obj.module.path)
            parent = obj.parent
        except Exception:  # the object tree is still being built
            module, parent = "hires", None

        owner = None
        if obj.is_class:
            owner = f"{module}.{obj.name}"
        elif parent is not None and parent.is_class:
            owner = f"{module}.{parent.name}"

        obj.docstring.value = translate(obj.docstring.value, module=module, owner=owner)
