# User guide

The user guide explains the concepts behind HiReS and how its parts fit
together. It is meant to be read; for the exact signature of a function, go
to the [API reference](../reference/index.md).

## Fundamentals

| Page | What you will learn |
|------|---------------------|
| [The data model](data-model.md) | `Annotation`, `Collection` and `Album`; normalized coordinates; immutability |
| [Reading and writing](io.md) | The YOLO segmentation label format, loading files and directories, exporting tables, labels and crops |

## Segmentation and measurement

| Page | What you will learn |
|------|---------------------|
| [Tiled segmentation](segmentation.md) | How an image is tiled, filtered, merged and deduplicated, and how to choose the parameters |
| [Morphometric traits](morphometrics.md) | The definition of every trait and the conversion to physical units |
| [Filtering and class handling](filtering.md) | Selecting annotations, translating class ids between schemas, balancing datasets |

## Quality control

| Page | What you will learn |
|------|---------------------|
| [Evaluation](evaluation.md) | Matching predictions to ground truth and the definition of every metric |
| [Visualisation](plotting.md) | Overlays, comparison images, single-object figures and charts |

## Further topics

| Page | What you will learn |
|------|---------------------|
| [Tiling training data](tiling.md) | Cutting images and their labels into tiles to train a model |
| [Errors, warnings and logging](diagnostics.md) | The exception hierarchy, `HiresWarning`, and how to see progress messages |

## Conventions used in this documentation

```python
import hires
```

is assumed throughout. Submodules are accessed as attributes
(`hires.metrics`, `hires.plot`, `hires.tiling`, `hires.io`); there is no
need to import them separately.

Sizes are always given as `(width, height)`, in that order, in pixels.
Functions that compare two sets of annotations always take the **ground
truth first** and the **predictions second**.
