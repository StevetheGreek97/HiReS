# Contributing to HiReS

Bug reports, feature requests and pull requests are welcome at
<https://github.com/StevetheGreek97/HiReS>.

The full guide, covering the development setup, code and docstring
conventions, tests, documentation and the release process, is part of the
documentation:

**<https://stevethegreek97.github.io/HiReS/development/contributing.html>**

(source: [`docs/development/contributing.md`](docs/development/contributing.md))

## In short

```bash
git clone https://github.com/StevetheGreek97/HiReS.git
cd HiReS
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"

make check    # lint and tests: what CI runs
make docs     # build the documentation
```

- Open an issue before starting on a larger change.
- Add a test for every bug fix and every new feature.
- Document public functions with numpydoc docstrings.
- Add an entry to `CHANGELOG.md` under *Unreleased*.

When reporting a bug, include the output of
`python -c "import hires; hires.show_versions()"`.
