# HiReS documentation

**Version 2.0** · [Source](https://github.com/StevetheGreek97/HiReS) ·
[Issues](https://github.com/StevetheGreek97/HiReS/issues) ·
[Release notes](development/release-notes.md)

HiReS (*High-Resolution Segmentation*) is a Python library and command-line
tool for instance segmentation and morphometric analysis of images that are
too large for a single neural-network pass. It cuts a scan into overlapping
tiles, runs a YOLO segmentation model on each tile, stitches the detections
back into polygons in full-image coordinates, measures the shape of every
object, and evaluates the result against ground truth.

```python
import hires

collection = hires.segment("scan.tif", "DaphnAI.pt")
collection.with_scale(dpi=2400, unit="um").to_csv("traits.csv")
```

<div class="grid cards" markdown>

-   **Getting started**

    ---

    New to HiReS? Install it and segment your first image in a few minutes.

    [Installation](getting-started/installation.md) ·
    [Quickstart](getting-started/quickstart.md)

-   **User guide**

    ---

    Background and worked explanations: the data model, the tiling
    algorithm, the trait definitions and the evaluation protocol.

    [Open the user guide](user-guide/index.md)

-   **API reference**

    ---

    Every public class, function and constant, with parameters, return
    values and examples. Assumes you know the concepts.

    [Open the reference](reference/index.md)

-   **Command line**

    ---

    The `hires` command: `run`, `chunk`, `plot` and `evaluate`, and the
    files they write.

    [The hires command](reference/cli.md) ·
    [Output files](reference/outputs.md)

</div>

## What HiReS does

Tiled segmentation
:   Images of 10,000 × 10,000 pixels and more are split into overlapping
    tiles so that inference runs on a laptop CPU or a single GPU. Objects cut
    by a tile border are dropped from that tile and recovered from its
    neighbour; duplicates are removed with polygon non-maximum suppression.
    See [Tiled segmentation](user-guide/segmentation.md).

A small, immutable data model
:   [`Annotation`][hires.Annotation] (one polygon),
    [`Collection`][hires.Collection] (one image) and
    [`Album`][hires.Album] (one dataset). Every operation returns a new
    object, so pipelines are chains of method calls with no hidden state.
    See [The data model](user-guide/data-model.md).

Morphometric traits
:   Area, perimeter, solidity, convexity, circularity and the dimensions of
    the axis-aligned and oriented bounding boxes, in pixels or in physical
    units derived from the scan resolution.
    See [Morphometric traits](user-guide/morphometrics.md).

Evaluation against ground truth
:   IoU matching, per-class precision, recall, F1 and average precision,
    confusion matrices, precision-recall curves and per-object comparison
    figures. See [Evaluation](user-guide/evaluation.md).

Model-agnostic
:   Any Ultralytics YOLO segmentation model works, as does any object with
    a compatible `predict()` method. YOLO itself is an optional dependency:
    reading, measuring and evaluating label files needs no deep-learning
    stack.

## How the package is organised

HiReS is laid out like NumPy or SciPy. The objects you use every day live in
the top-level namespace; specialised functionality lives in submodules.

| Namespace | Contents |
|-----------|----------|
| [`hires`](reference/index.md) | Data model, readers, `segment`, `run_pipeline`, `measure`, `evaluate` |
| [`hires.metrics`](reference/metrics.md) | Matching predictions to ground truth and detection metrics |
| [`hires.plot`](reference/plot.md) | Overlays, comparison images and evaluation charts |
| [`hires.tiling`](reference/tiling.md) | Splitting images and labels into tiles and merging them back |
| [`hires.io`](reference/io.md#low-level-helpers-hiresio) | Low-level image and label reading helpers |
| [`hires.analysis`](reference/analysis.md) | Trait-agreement plots (legacy) |

## Citing HiReS

If you use HiReS in your research, please cite:

> Mavrianos S., Teurlincx S., Declerck S.A.J., Otte K.A. (2025). *HiReS: A
> Method for Automated Morphometric Trait Extraction from High-Resolution
> Plankton Images.*

## License

HiReS is distributed under the [MIT License](https://github.com/StevetheGreek97/HiReS/blob/main/LICENSE).
