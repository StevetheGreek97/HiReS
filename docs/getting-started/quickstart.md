# Quickstart

This page takes you from an image to a table of measurements, and then to an
evaluation against manual annotations. It assumes HiReS is installed with the
`yolo` extra (see [Installation](installation.md)) and that you have

- a high-resolution image, here `scan.tif`, and
- YOLO segmentation weights, here `DaphnAI.pt`.

Everything below can be done from Python or from the command line; both are
shown.

## 1. Segment an image

=== "Python"

    ```python
    import hires

    collection = hires.segment("scan.tif", "DaphnAI.pt")

    len(collection)             # number of detected objects
    collection.class_counts()   # {class_id: count}
    ```

=== "Command line"

    ```bash
    hires run --source scan.tif --model DaphnAI.pt --output results/
    ```

[`hires.segment`][hires.segment] cuts the image into overlapping
1024 × 1024 px tiles, runs the model on each, and returns a
[`Collection`][hires.Collection]: an immutable sequence of
[`Annotation`][hires.Annotation] objects, one polygon per detected object.

The most important settings are the tile overlap, which should exceed your
largest object, and the confidence threshold:

```python
collection = hires.segment(
    "scan.tif", "DaphnAI.pt", overlap=300, confidence=0.4, device="0"
)
```

All settings are described under [`hires.Segmenter`][hires.Segmenter].

## 2. Measure

Every annotation can be measured. Attach the scan resolution to get physical
units instead of pixels:

```python
scaled = collection.with_scale(dpi=2400, unit="um")

traits = scaled.to_frame()           # pandas.DataFrame, one row per object
traits[["class_id", "area", "obb_length", "circularity"]].describe()

scaled.to_csv("scan_shapes.csv")
```

The table has one column per trait in [`hires.TRAIT_NAMES`][hires.TRAIT_NAMES]:
`area`, `perimeter`, `solidity`, `convexity`, `circularity`, `bbox_width`,
`bbox_height`, `obb_width` and `obb_length`. Their definitions are given in
[Morphometric traits](../user-guide/morphometrics.md).

On the command line, add `--dpi` and `--unit`:

```bash
hires run -s scan.tif -m DaphnAI.pt -o results/ --dpi 2400 --unit um
```

## 3. Look at the result

```python
import matplotlib.pyplot as plt

rgb = hires.plot.overlay(
    "scan.tif", collection, class_names={0: "D. pulex", 1: "D. galeata"}
)
plt.imshow(rgb)
```

Pass `save="annotated.png"` to write the image instead. The number drawn next
to each object is its position in the collection, so `collection[158]` is the
object labelled `158`.

## 4. Process a whole directory

[`hires.run_pipeline`][hires.run_pipeline] segments every image of a
directory and writes labels, trait tables and overlays to disk. It is what
`hires run` calls.

=== "Python"

    ```python
    album = hires.run_pipeline(
        "scans/", "DaphnAI.pt", "results/", dpi=2400, unit="um"
    )

    album.to_frame().groupby("name")["area"].median()
    ```

=== "Command line"

    ```bash
    hires run -s scans/ -m DaphnAI.pt -o results/ --dpi 2400 --unit um
    ```

For each image `scan.tif` this writes

| File | Content |
|------|---------|
| `results/scan.txt` | Detections as YOLO segmentation labels |
| `results/scan_shapes.csv` | One row per detection with its traits |
| `results/scan_annotated.tif` | The image with the detections drawn on top |
| `results/run_config.yaml` | The settings of the run |

See [Output files](../reference/outputs.md) for the details. The function
returns an [`Album`][hires.Album], a named set of collections, one per image.

## 5. Work with existing label files

You do not need a model to use HiReS. Any YOLO segmentation label file can
be read, filtered, measured and written back:

```python
collection = hires.read_txt("results/scan.txt", image="scan.tif")

animals = (
    collection
    .filter_confidence(0.6)
    .drop_classes(2)
    .with_scale(dpi=2400, unit="um")
    .filter_area(min_area=5e4)
)
animals.to_csv("animals.csv")
```

Passing `image=` lets HiReS read the image size, which it needs to measure in
pixels rather than in normalized coordinates.

## 6. Evaluate against ground truth

With manual annotations in `labels/` and predictions in `results/`:

=== "Python"

    ```python
    gt = hires.read_album("labels/", image_dir="scans/")
    pred = hires.read_album("results/", image_dir="scans/")

    report = hires.evaluate(gt, pred, "evaluation/")
    report.loc["all", ["precision", "recall", "f1", "ap50"]]
    ```

=== "Command line"

    ```bash
    hires evaluate --gt labels/ --pred results/ --images scans/ -o evaluation/
    ```

This writes a per-class detection report, a confusion matrix, a
precision-recall curve, a table of all matched objects and one comparison
image per scan. For finer control use the functions in
[`hires.metrics`](../reference/metrics.md) directly:

```python
matches = hires.metrics.match(gt, pred, iou_threshold=0.5)
matches.summary()

worst = matches.pick("tp", by="worst_iou")
hires.plot.individual(worst, save="worst_match.png")
```

## Where to go next

- [The data model](../user-guide/data-model.md) explains `Annotation`,
  `Collection` and `Album` properly.
- [Tiled segmentation](../user-guide/segmentation.md) explains how tiling
  works and how to choose `tile_size`, `overlap` and `nms_iou`.
- [Evaluation](../user-guide/evaluation.md) defines every metric.
- The [API reference](../reference/index.md) lists everything.
