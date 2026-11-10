# Visualisation

[`hires.plot`](../reference/plot.md) draws annotations, match results and
evaluation charts. Two conventions hold throughout:

- Functions that draw **on an image** return an RGB `numpy.ndarray`.
- Functions that draw a **chart** return the Matplotlib `Axes` or `Figure`.

No function calls `plt.show()`, and every function accepts `save=` to write
the result to a file. Matplotlib is imported only when `hires.plot` is
first used.

## Annotations on an image

```python
import matplotlib.pyplot as plt

rgb = hires.plot.overlay(
    "scan.tif", collection, class_names={0: "D. pulex", 1: "D. galeata"}
)
plt.imshow(rgb)
```

[`overlay`][hires.plot.overlay] fills each polygon in the colour of its
class, draws its bounding box and writes a label `index class confidence`
next to it. The **index** is the position of the annotation in the
collection, and for a collection read from disk, the line number in the
label file. Use it to look an object up: `collection[158]`.

Each element can be switched off:

```python
hires.plot.overlay(
    "scan.tif", collection,
    fill=True, boxes=False, labels=False, alpha=0.35,
    save="outlines_only.png",
)
```

The first argument may also be an 8-bit BGR array, as returned by
[`hires.io.read_image`](../reference/io.md#low-level-helpers-hiresio).
Reading a large scan once and passing the array avoids decoding it again
for every figure.

Colours are taken from `hires.plot.PALETTE` by class id and repeat after
ten classes.

## Predictions versus ground truth

```python
matches = hires.metrics.match(gt, pred)

hires.plot.comparison(
    "scans/C1-1_control.tif",
    matches.for_image("C1-1_control"),
    save="comparison.png",
)
```

[`comparison`][hires.plot.comparison] colours each object by its match
status:

| Colour | Status | Polygon drawn |
|--------|--------|---------------|
| Green | `tp` | Prediction |
| Orange | `misclassified` | Prediction |
| Red | `fp` | Prediction |
| Blue | `fn` | Missed ground truth |

For an album result, select one image with `for_image` first, as above.
`hires.evaluate` writes this image for every scan.

## A single object

[`individual`][hires.plot.individual] crops the image around one match and
draws four panels: the ground-truth mask, the predicted mask, the raw crop,
and both outlines together with the IoU and the match status.

```python
worst = matches.pick("tp", by="worst_iou")
figure = hires.plot.individual(worst, class_names=names, save="worst.png")
```

The image is taken from the match when the albums were read with
`image_dir=`. For many figures of the same scan, read it once:

```python
scan = hires.io.read_image("scans/C1-1_control.tif")
for number, m in enumerate(matches.for_image("C1-1_control").with_status("fn")):
    figure = hires.plot.individual(m, scan, save=f"missed/{number:03d}.png")
    plt.close(figure)
```

!!! warning "Close figures in loops"

    Matplotlib keeps every figure alive until it is closed. In a loop, call
    `plt.close(figure)` after saving, as above.

To draw only the outline of one annotation, without an image, use
[`annotation`][hires.plot.annotation]. With `obb=True` it adds the minimum
rotated rectangle from which body length and width are measured:

```python
hires.plot.annotation(
    collection[0], image_size=collection.image_size, obb=True, bbox=True
)
```

## Evaluation charts

### Confusion matrix

```python
matrix = hires.metrics.confusion_matrix(gt, pred, class_names=names)
ax = hires.plot.confusion_matrix(matrix)                    # row-normalized
ax = hires.plot.confusion_matrix(matrix, normalize=False)   # raw counts
```

### Precision-recall curves

A single curve, or several as a dict of `{label: curve}`; the point with
the best F1 is marked on each:

```python
curves = {
    name: hires.metrics.precision_recall_curve(gt, pred, class_id=class_id)
    for class_id, name in names.items()
}
hires.plot.precision_recall(curves, save="pr_per_class.png")
```

The same works for comparing models: one curve per model, same ground
truth.

### Training curves

[`training_curves`][hires.plot.training_curves] plots the training and
validation losses from the `results.csv` that Ultralytics writes during
training:

```python
hires.plot.training_curves("runs/segment/train/results.csv", patience=50)
```

With `patience`, a vertical line marks the best epoch, assuming training
was ended by early stopping.

## Composing figures

Chart functions accept `ax=`, so they fit into any Matplotlib layout:

```python
figure, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
hires.plot.confusion_matrix(matrix, ax=left, title="Confusion")
hires.plot.precision_recall(curve, ax=right, title="Precision-recall")
figure.savefig("evaluation.pdf")
```

## Working without a display

On a server or in a batch job, select a non-interactive backend before the
first plot:

```python
import matplotlib

matplotlib.use("Agg")
```

The `hires` command needs no such setup.

## See also

- Reference: [`hires.plot`](../reference/plot.md)
- [Evaluation](evaluation.md)
