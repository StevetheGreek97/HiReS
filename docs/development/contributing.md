# Contributing

Contributions are welcome: bug reports, documentation fixes, new features.
The project lives at <https://github.com/StevetheGreek97/HiReS>.

## Reporting bugs and requesting features

Open an [issue](https://github.com/StevetheGreek97/HiReS/issues). For a
bug, include

- what you did, as a minimal piece of code or the exact command,
- what you expected and what happened instead, with the full traceback,
- the output of `hires.show_versions()`.

For a larger change, please open an issue before writing code, so the
design can be discussed first.

## Development setup

```bash
git clone https://github.com/StevetheGreek97/HiReS.git
cd HiReS
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
```

The tests use a stand-in model, so neither `ultralytics` nor a GPU is
needed. Add the `yolo` extra only to run real models.

The makefile wraps the usual commands:

| Command | Does |
|---------|------|
| `make check` | Everything CI runs: `lint` and `test` |
| `make lint` | `ruff check` |
| `make fix` | Apply the safe lint fixes |
| `make test` | Run the test suite |
| `make examples` | Re-run the `>>>` examples in the docs and rewrite their output and figures |
| `make docs` | Build the documentation with `--strict` |
| `make docs-serve` | Serve the documentation at <http://127.0.0.1:8000> with live reload |
| `make build` | Build the sdist and wheel into `dist/` |

## Project layout

```text
hires/
├── __init__.py       public top-level namespace
├── annotation.py     Annotation, measure
├── collection.py     Collection
├── album.py          Album
├── classes.py        ClassMapping
├── io.py             readers
├── segmentation.py   Segmenter, segment, run_pipeline
├── tiling.py         hires.tiling
├── metrics.py        hires.metrics
├── plot.py           hires.plot
├── errors.py         exceptions and warnings
├── cli.py            the hires command
└── analysis/         legacy trait-agreement plots
tests/                pytest suite
docs/                 MkDocs sources
tools/                build helpers that are not part of the package
```

## Code conventions

**Style.** Code is linted with [Ruff](https://docs.astral.sh/ruff/),
configured in `pyproject.toml`. The line length is 100.

**Public API.** A name is public if it is listed in the `__all__` of its
module and documented in the reference. Everyday objects are re-exported
from `hires/__init__.py`; specialised ones stay in their submodule.

**Design rules.** These keep the package predictable; please follow them
in new code.

- Objects of the data model are immutable. Methods return new objects.
- Sizes are `(width, height)`, in pixels.
- Functions comparing two sets of annotations take the ground truth first.
- Everything after the essential arguments is keyword-only.
- Heavy or optional dependencies (`matplotlib`, `cv2`, `pandas`,
  `ultralytics`) are imported inside the functions that need them, so that
  `import hires` stays fast. Optional ones go through
  `hires.errors.require`, which raises a `MissingDependencyError` naming
  the extra to install.
- The library never prints and never configures logging. Use
  `logging.getLogger(__name__)` for progress and `HiresWarning` for
  results that are probably not what the user intended.
- Plotting functions return what they drew, never call `plt.show()`, and
  accept `save=`.

**Type hints** are required for public functions. The package ships a
`py.typed` marker.

## Docstrings

Docstrings follow the
[numpydoc standard](https://numpydoc.readthedocs.io/en/latest/format.html)
and are enforced by Ruff's `D` rules. A public function documents its
parameters, return value and raised exceptions, and should have an example:

```python
def iou(a, b):
    """Intersection-over-union of two polygons.

    Parameters
    ----------
    a, b : Annotation or shapely.geometry.Polygon

    Returns
    -------
    float
        Value in ``[0, 1]``; ``0.0`` when the polygons do not overlap.

    Examples
    --------
    >>> from shapely.geometry import box
    >>> hires.metrics.iou(box(0, 0, 2, 1), box(1, 0, 3, 1))
    0.3333333333333333
```

Cross-references use reStructuredText roles, e.g.
``:func:`hires.measure` `` or ``:meth:`Collection.nms` ``; the extension in
`tools/griffe_rst.py` turns them into links when the reference is built.
Examples that need files or a model are marked `# doctest: +SKIP`.

## Tests

```bash
pytest                         # the whole suite
pytest tests/test_metrics.py   # one file
pytest -k nms                  # tests matching a keyword
```

- Every bug fix comes with a test that fails without the fix.
- Every new public function comes with tests of its behaviour and of its
  error cases.
- Tests must not need network access, a GPU or model weights. Use the
  fixtures in `tests/conftest.py`: small synthetic images and `FakeModel`,
  a stand-in for an Ultralytics model.
- A `HiresWarning` raised during a test is an error. A test that expects
  one must say so with `pytest.warns`.

## Documentation

The documentation is built with [MkDocs](https://www.mkdocs.org/),
[Material](https://squidfunk.github.io/mkdocs-material/) and
[mkdocstrings](https://mkdocstrings.github.io/).

| Section | Purpose | Written |
|---------|---------|---------|
| Getting started | Take a new user to a first result | By hand |
| User guide | Explain concepts and the reasoning behind them | By hand |
| Reference | State exactly what every object does | Generated from the docstrings |

### Examples with output

The user guide follows the NumPy and Shapely manuals: every feature is
shown as an interactive example, the statement after `>>>` and its output
below it.

- Examples must run as they are, without a model or files of the reader.
  Build small objects in the page, or write the files the example needs.
- **Never type output by hand.** Write the statements, then run
  `make examples`. `tools/doc_examples.py` executes each page from top to
  bottom in an empty directory and fills in what was printed. A PNG that
  a page saves and embeds as `assets/examples/{name}.png` is copied into
  `docs/assets/examples/`.
- The pages are also tests: `pytest` runs them as doctests, so an example
  whose output changed fails CI.
- Keep output portable: show `path.name`, not paths; round floats; wrap
  NumPy scalars in `float()`.
- Mark statements that need a real model or data `# doctest: +SKIP`.

A new public object needs a docstring **and** a `::: hires.name` entry on
the matching page in `docs/reference/`; a new feature usually also deserves
a paragraph in the user guide. `make docs` builds in strict mode and fails
on broken links and unresolved cross-references.

## Pull requests

1. Create a branch from `main`.
2. Make the change, with tests and documentation.
3. Add an entry under *Unreleased* in `CHANGELOG.md`.
4. Run `make check` and `make docs`.
5. Open the pull request and describe what it changes and why.

Continuous integration runs the linter, the tests on every supported Python
version, and the documentation build.

## Compatibility policy

HiReS follows [semantic versioning](https://semver.org/).

- The **public API** is everything documented in the
  [reference](../reference/index.md), plus the options, output files and
  exit codes of the `hires` command.
- **Patch** releases (2.0.x) fix bugs.
- **Minor** releases (2.x.0) add features and may deprecate. A deprecated
  feature keeps working, emits a `DeprecationWarning`, and is removed no
  earlier than the next major release.
- **Major** releases may break compatibility, with a migration guide.

Exceptions: `hires.analysis` is legacy code outside this policy, and
numerical results may change in the last digits between releases.

Supported Python versions are those listed in `pyproject.toml`; support for
a version is dropped only in a minor or major release.

## Releasing

For maintainers.

1. Update `version` in `pyproject.toml` and `CITATION.cff`.
2. In `CHANGELOG.md`, rename *Unreleased* to the version, add the date and
   the link at the bottom, and start a new empty *Unreleased* section.
3. Run `make check docs build`.
4. Commit, tag and push:

    ```bash
    git commit -am "Release 2.0.0"
    git tag v2.0.0
    git push origin main --tags
    ```

5. Create a GitHub release from the tag. The `Publish` workflow builds the
   distributions and uploads them to PyPI; the `Docs` workflow deploys the
   documentation.
