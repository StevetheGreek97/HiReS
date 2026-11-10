# API reference

The reference describes every public class, function and constant of HiReS.
It assumes you know the concepts; for background and worked explanations,
read the [user guide](../user-guide/index.md).

Everything listed here is public and covered by the
[compatibility policy](../development/contributing.md#compatibility-policy).
Names that start with an underscore, and anything not listed, are private
and may change without notice.

## Top-level namespace

```python
import hires
```

| Object | Description |
|--------|-------------|
| [`hires.Annotation`](annotation.md) | One polygon with a class and an optional confidence |
| [`hires.Collection`](collection.md) | The annotations of one image |
| [`hires.Album`](album.md) | The collections of one dataset |
| [`hires.read_txt`](io.md), [`hires.read_album`](io.md), [`hires.read_class_names`](io.md) | Readers for label files and class names |
| [`hires.segment`](segmentation.md), [`hires.Segmenter`](segmentation.md), [`hires.run_pipeline`](segmentation.md) | Tiled segmentation |
| [`hires.measure`](measurement.md), [`hires.TRAIT_NAMES`](measurement.md), [`hires.UNITS`](measurement.md) | Morphometric traits |
| [`hires.ClassMapping`](classes.md), [`hires.build_class_mapping`](classes.md) | Translating class ids between schemas |
| [`hires.evaluate`](metrics.md) | Full evaluation report on disk |
| [`hires.HiresError`](errors.md) and subclasses, [`hires.HiresWarning`](errors.md) | Exceptions and warnings |
| [`hires.show_versions`](#hires.show_versions) | Version report for bug reports |
| `hires.__version__` | The installed version, as a string |

## Submodules

Submodules are available as attributes after `import hires`.

| Module | Description |
|--------|-------------|
| [`hires.metrics`](metrics.md) | Matching predictions to ground truth and detection metrics |
| [`hires.plot`](plot.md) | Overlays, comparison images and evaluation charts |
| [`hires.tiling`](tiling.md) | Splitting images and labels into tiles and merging them back |
| [`hires.io`](io.md#low-level-helpers-hiresio) | Low-level image and label reading helpers |
| [`hires.analysis`](analysis.md) | Trait-agreement plots (legacy) |

## Command line

| Page | Description |
|------|-------------|
| [The hires command](cli.md) | `hires run`, `chunk`, `plot` and `evaluate` |
| [Output files](outputs.md) | Every file the commands write |

## Utilities

::: hires.show_versions
