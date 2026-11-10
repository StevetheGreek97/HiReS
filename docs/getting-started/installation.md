# Installation

HiReS requires **Python 3.10 or newer**. The distribution is called
`hireseg`; the import name and the command are `hires`.

## Install from PyPI

=== "Segmentation and analysis"

    ```bash
    pip install "hireseg[yolo]"
    ```

    Installs HiReS together with [Ultralytics](https://docs.ultralytics.com/)
    (and therefore PyTorch), which is needed to run a model.

=== "Analysis only"

    ```bash
    pip install hireseg
    ```

    Enough to read, measure, filter, plot and evaluate label files. No
    deep-learning stack is installed.

=== "Everything"

    ```bash
    pip install "hireseg[all]"
    ```

## Optional dependencies

The core install is deliberately light. Features that need heavy packages are
grouped in *extras*:

| Extra | Installs | Needed for |
|-------|----------|------------|
| `yolo` | `ultralytics` | [`hires.segment`][hires.segment], [`hires.Segmenter`][hires.Segmenter], [`hires.run_pipeline`][hires.run_pipeline], `hires run`, and reading class names from `.pt` weights |
| `analysis` | `scipy`, `plotnine`, `SkillMetrics` | The legacy trait-agreement plots in [`hires.analysis`](../reference/analysis.md) |
| `all` | `yolo` + `analysis` | Everything above |
| `dev` | `pytest`, `ruff` | Running the tests and the linter |
| `docs` | `mkdocs-material`, `mkdocstrings[python]` | Building this documentation |

Calling a feature whose extra is missing raises
[`MissingDependencyError`][hires.MissingDependencyError] with the exact
`pip` command to run.

## Core dependencies

Installed automatically:

| Package | Minimum version | Used for |
|---------|-----------------|----------|
| NumPy | 1.26 | Arrays, IoU matrices |
| pandas | 2.2 | Trait tables and reports |
| Shapely | 2.0 | Polygon geometry |
| Pillow | 10 | Reading image sizes without decoding |
| opencv-python | 4.8 | Reading, tiling and drawing images |
| Matplotlib | 3.8 | Charts |
| PyYAML | 6.0 | `data.yaml` class names, run configuration |

## GPU support

HiReS runs on the CPU by default. To use a GPU, install a PyTorch build that
matches your CUDA version **before** installing the `yolo` extra, following
the selector on [pytorch.org](https://pytorch.org/get-started/locally/):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "hireseg[yolo]"
```

Then pass `device="0"` (Python) or `--device 0` (command line).

## Install from source

```bash
git clone https://github.com/StevetheGreek97/HiReS.git
cd HiReS
pip install -e ".[all,dev]"
```

See [Contributing](../development/contributing.md) for the development
workflow.

## Verify the installation

```bash
hires --version
```

```python
import hires

hires.show_versions()
```

[`show_versions`][hires.show_versions] prints the versions of HiReS and of
every dependency, including the optional ones. Include its output when you
report a bug.
