"""Settings for running the ``>>>`` examples of the documentation as tests.

``pytest docs`` executes every Markdown page as a doctest. Each page runs in
an empty temporary directory, so examples may freely write files.
"""

from __future__ import annotations

import doctest
import re

import matplotlib
import pandas as pd
import pytest

import hires

matplotlib.use("Agg")

# An example's expected output ends at the closing ``` of its code block, the
# same rule tools/doc_examples.py uses. Without this, doctest reads the fence
# as part of the output.
doctest.DocTestParser._EXAMPLE_RE = re.compile(
    doctest.DocTestParser._EXAMPLE_RE.pattern.replace(
        r"(?![ ]*>>>)", r"(?![ ]*>>>)(?![ ]*```)", 1
    ),
    doctest.DocTestParser._EXAMPLE_RE.flags,
)

# The ">>>" lines on this page sit inside a sample docstring that shows the
# house style; they are not a runnable example.
collect_ignore = ["development/contributing.md"]


@pytest.fixture(autouse=True)
def _doc_example_environment(doctest_namespace, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.chdir(tmp_path)
    doctest_namespace["hires"] = hires
    with pd.option_context(
        "display.width", 88, "display.max_columns", 12, "display.precision", 4
    ):
        yield
    plt.close("all")
