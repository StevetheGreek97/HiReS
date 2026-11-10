"""Run the ``>>>`` examples in the documentation and keep their output up to date.

Every Markdown page under ``docs/`` is executed top to bottom in its own
namespace and in its own empty working directory, like a doctest. The text
that follows each ``>>>`` statement is then replaced by what the statement
actually printed, so the output shown in the documentation is never typed
by hand.

Usage::

    python tools/doc_examples.py            # rewrite the output in place
    python tools/doc_examples.py --check    # only report differences
    python tools/doc_examples.py docs/user-guide/io.md

Statements marked ``# doctest: +SKIP`` are not run and their output is left
alone. PNG files that a page writes and embeds as
``assets/examples/{name}.png`` are copied to ``docs/assets/examples/``.

The script is a development tool and is not part of the installed package.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import shutil
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
FIGURES = DOCS / "assets" / "examples"

_PROMPT = re.compile(r"^(?P<indent>\s*)>>> ?(?P<code>.*)$")
_FIGURE = re.compile(r"assets/examples/(?P<name>[\w.-]+\.png)")


def _prepare() -> None:
    """Make output reproducible: same settings as ``docs/conftest.py``."""
    sys.path.insert(0, str(ROOT))
    import matplotlib

    matplotlib.use("Agg")
    import pandas as pd

    import hires

    pd.set_option("display.width", 88, "display.max_columns", 12, "display.precision", 4)
    warnings.simplefilter("error", hires.HiresWarning)


def _run(source: str, namespace: dict) -> str:
    """Execute one interactive statement and return what it printed."""
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        try:
            exec(compile(source, "<example>", "single"), namespace)
        except Exception as error:
            last = traceback.format_exception_only(type(error), error)[-1].rstrip()
            return f"Traceback (most recent call last):\n    ...\n{last}"
    return captured.getvalue().rstrip("\n")


def update(path: Path) -> tuple[str, int]:
    """Return the page with refreshed output, and the number of examples run."""
    import matplotlib.pyplot as plt

    import hires

    lines = path.read_text(encoding="utf-8").split("\n")
    result: list[str] = []
    namespace: dict = {"hires": hires, "__name__": "__main__"}
    executed = 0
    here = Path.cwd()

    with tempfile.TemporaryDirectory() as workdir:
        os.chdir(workdir)
        try:
            i = 0
            while i < len(lines):
                found = _PROMPT.match(lines[i])
                if found is None:
                    result.append(lines[i])
                    i += 1
                    continue

                indent = found["indent"]
                source = [found["code"]]
                result.append(lines[i])
                i += 1
                while i < len(lines) and lines[i].startswith(indent + "..."):
                    source.append(lines[i][len(indent) + 4 :])
                    result.append(lines[i])
                    i += 1

                old: list[str] = []
                while (
                    i < len(lines)
                    and lines[i].strip()
                    and not _PROMPT.match(lines[i])
                    and not lines[i].strip().startswith("```")
                ):
                    old.append(lines[i])
                    i += 1

                code = "\n".join(source) + "\n"
                if "doctest: +SKIP" in code:
                    result.extend(old)
                    continue
                executed += 1
                output = _run(code, namespace)
                for line in output.split("\n") if output else []:
                    result.append(indent + (line.rstrip() or "<BLANKLINE>"))

            for name in {m["name"] for m in _FIGURE.finditer("\n".join(lines))}:
                if Path(name).is_file():
                    FIGURES.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(name, FIGURES / name)
        finally:
            plt.close("all")
            os.chdir(here)
    return "\n".join(result), executed


def main(argv: list[str] | None = None) -> int:
    """Refresh or check the pages given on the command line (default: all)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "pages", nargs="*", type=Path, help="Markdown files (default: all of docs/)"
    )
    parser.add_argument("--check", action="store_true", help="report stale output, change nothing")
    args = parser.parse_args(argv)

    _prepare()
    pages = [p.resolve() for p in args.pages] or sorted(DOCS.rglob("*.md"))
    stale = 0
    for page in pages:
        before = page.read_text(encoding="utf-8")
        after, executed = update(page)
        if not executed:
            continue
        changed = after != before
        stale += changed
        if changed and not args.check:
            page.write_text(after, encoding="utf-8")
        status = ("STALE" if args.check else "updated") if changed else "ok"
        print(f"{status:>8}  {page.relative_to(ROOT)}  ({executed} examples)")
    return 1 if args.check and stale else 0


if __name__ == "__main__":
    sys.exit(main())
