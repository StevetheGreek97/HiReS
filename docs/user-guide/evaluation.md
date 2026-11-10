# Evaluation

How good are the predictions? HiReS answers this by pairing every predicted
polygon with a ground-truth polygon and counting what is left over. This
page explains the pairing, defines every metric derived from it, and shows
how to inspect individual objects.

All functions live in [`hires.metrics`](../reference/metrics.md) and take
the **ground truth first** and the **predictions second**.

## The short version

```python
gt = hires.read_album("labels/", image_dir="scans/")
pred = hires.read_album("results/", image_dir="scans/")

report = hires.evaluate(gt, pred, "evaluation/")
```

[`hires.evaluate`][hires.evaluate] writes the detection report, the
confusion matrix, the precision-recall curve, a table of all matches and one
comparison image per scan (see
[Output files](../reference/outputs.md#hires-evaluate)) and returns the
detection report. The rest of this page describes the pieces it is built
from.

## Inputs

Both arguments are either two [`Collection`][hires.Collection] objects, for
one image, or two [`Album`][hires.Album] objects, for a dataset. Albums are
paired **by collection name**, which for albums read from disk is the file
stem of the label file.

Collections that exist on only one side are skipped and reported:

```python
matches = hires.metrics.match(gt, pred)
matches.unpaired_gt     # names found only in the ground truth
matches.unpaired_pred   # names found only in the predictions
```

Pass `strict=True` to raise an error instead. An image **without objects**
still needs an (empty) label file on both sides; otherwise its false
positives go unnoticed.

!!! tip "Different class schemas"

    If ground truth and predictions number their classes differently,
    translate one side first; see
    [Translating class ids](filtering.md#translating-class-ids).

## Matching

[`hires.metrics.match`][hires.metrics.match] pairs polygons by
intersection-over-union,

$$
\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|},
$$

computed on the polygons themselves, not on their bounding boxes.

1. All pairs with $\text{IoU} \ge t$ are candidates. The threshold $t$ is
   `iou_threshold`, `0.5` by default.
2. Candidates are accepted greedily in order of decreasing IoU. Once an
   object is part of a match it is not available again, so the matching is
   one-to-one.
3. Whatever remains unmatched becomes a false negative (ground truth) or a
   false positive (prediction).

Every object therefore ends up in exactly one [`Match`][hires.metrics.Match]
with one of four statuses:

| Status | Ground truth | Prediction | Meaning |
|--------|:------------:|:----------:|---------|
| `tp` | ✓ | ✓ | Found, with the right class |
| `misclassified` | ✓ | ✓ | Found, with the wrong class |
| `fp` | – | ✓ | A prediction where there is no object |
| `fn` | ✓ | – | An object the model missed |

### Class-agnostic and class-aware matching

By default, matching ignores the class: an object that was outlined
correctly but labelled wrongly is one `misclassified` match. This separates
*localisation* errors from *classification* errors, which is usually what
you want to know.

With `class_aware=True` only polygons of the same class may match, so the
same object yields one `fn` and one `fp`. The aggregate metrics below are
identical either way; only the bookkeeping differs.

### Choosing the IoU threshold

| `iou_threshold` | Asks |
|-----------------|------|
| `0.5` | Was the object found? The conventional default. |
| `0.7`–`0.8` | Is the outline good enough to measure? Use this when the downstream analysis is morphometric. |
| `0.9` and above | Rarely meaningful: two careful human annotators often agree less than that on small objects. |

## Working with the result

`match` returns a [`MatchResult`][hires.metrics.MatchResult], an immutable
sequence of matches:

```python
matches = hires.metrics.match(gt, pred, iou_threshold=0.5)

matches.counts()
# {'tp': 412, 'misclassified': 9, 'fp': 31, 'fn': 27}

matches.summary()
# n_gt, n_pred, the four counts, precision, recall, f1 and mean_iou

errors = matches.with_status("fp", "fn")       # a smaller MatchResult
one_scan = matches.for_image("C1-1_control")
small = matches.filter(lambda m: m.iou < 0.6)
```

[`to_frame`][hires.metrics.MatchResult.to_frame] turns it into a table with
one row per match, including the nine traits of both objects as `gt_*` and
`pred_*` columns:

```python
table = matches.to_frame(class_names={0: "D. pulex", 1: "D. galeata"})
table.groupby("status")["iou"].describe()
```

Row `i` of the table describes `matches[i]`.

## Metrics

### Precision, recall and F1

$$
P = \frac{TP}{TP + FP}, \qquad
R = \frac{TP}{TP + FN}, \qquad
F_1 = \frac{2PR}{P + R}
$$

Precision is the fraction of predictions that are correct; recall is the
fraction of objects that were found; F1 is their harmonic mean. A metric
whose denominator is zero is reported as `0.0`.

A `misclassified` pair is **both** an error of the ground-truth class,
which lost an object (FN), and of the predicted class, which gained a wrong
one (FP). It is counted accordingly.

### The detection report

```python
report = hires.metrics.detection_report(
    gt, pred, iou_threshold=0.5, class_names={0: "D. pulex", 1: "D. galeata"}
)
```

| Column | Content |
|--------|---------|
| `tp`, `fp`, `fn` | Counts at `iou_threshold` |
| `precision`, `recall`, `f1` | Derived from the counts |
| `ap50` | Average precision at IoU 0.5 |
| `ap50_95` | Average precision, averaged over IoU 0.50, 0.55, …, 0.95 |

There is one row per class and a first row `"all"`. In that row the counts
are summed over the classes and precision, recall and F1 are computed from
the sums (a *micro* average, dominated by frequent classes), while `ap50`
and `ap50_95` are the unweighted mean over classes (mAP, a *macro* average).

### Average precision

Precision and recall describe the model at one confidence threshold.
Average precision (AP) summarises all thresholds: the predictions of a
class are ranked by confidence, precision and recall are computed for every
prefix of the ranking, and AP is the area under the resulting
precision-recall curve, using the usual all-point interpolation in which
precision at recall $r$ is replaced by the best precision at any recall
$\ge r$.

`ap50_95` repeats this for ten IoU thresholds and averages, which rewards
accurate outlines, not only found objects.

!!! note "Difference from COCO"

    HiReS matches polygons once, greedily by IoU and independently of
    confidence, and ranks afterwards. In the COCO protocol, predictions
    claim ground truth in order of confidence. The two agree unless several
    predictions compete for the same object, so values are close to, but
    not necessarily identical with, those printed by Ultralytics.

AP is only meaningful when the predictions were made with a **low
confidence threshold**. Predictions that were discarded by the model cannot
be ranked, and the curve stops early.

### Confusion matrix

```python
matrix = hires.metrics.confusion_matrix(gt, pred, class_names=names)
hires.plot.confusion_matrix(matrix, save="confusion.png")
```

Rows are the ground-truth class and columns the predicted class. An extra
`background` row and column collect the unmatched objects: false negatives
are in the `background` *column* (the object was predicted as nothing),
false positives in the `background` *row* (nothing was predicted as an
object). The plot normalizes each row by default, so the diagonal reads as
per-class recall.

## Choosing a confidence threshold

The confidence threshold trades precision against recall. To choose it from
data rather than by eye, segment a validation set with a low threshold and
compute the curve:

```python
pred = hires.run_pipeline("validation/", "model.pt", "low_conf/", confidence=0.05)
gt = hires.read_album("labels/")

curve = hires.metrics.precision_recall_curve(gt, pred)
best = curve.loc[curve["f1"].idxmax()]
best["confidence"], best["precision"], best["recall"]
```

Each row of the curve tells how the model would score if only predictions
with at least that confidence were kept. The row with the highest F1 is a
reasonable default; pick a higher threshold if false positives are costly
(they contaminate trait distributions) and a lower one if missed objects
are (they bias counts).

```python
hires.plot.precision_recall(curve, save="pr_curve.png")
```

Apply the chosen threshold either at inference (`confidence=`) or
afterwards with
[`filter_confidence`][hires.Collection.filter_confidence], which needs no
second model run.

Use `class_id=` for the curve of a single class. Because tiling removes
border objects before NMS, evaluate on **full images** processed by the
tiled pipeline, not on the validation tiles used during training: only the
former measures what you will get in production.

## Inspecting individual objects

Numbers say how often the model is wrong; pictures say why.

```python
image = hires.plot.comparison(
    "scans/C1-1_control.tif", matches.for_image("C1-1_control"),
    save="comparison.png",
)
```

draws every match on the scan, coloured by status. To look at one object
closely, select a match and pass it to
[`hires.plot.individual`](../reference/plot.md):

```python
typical = matches.pick("tp")                      # median IoU
worst = matches.pick("tp", by="worst_iou")
missed = matches.pick("fn", by="random", seed=0)

hires.plot.individual(worst, save="worst.png")
```

To find a specific object, use the number drawn next to it in an overlay:

```python
m = matches.find(pred_index=158, image="C1-1_control")
m.status, m.iou
```

See [Visualisation](plotting.md) for the figures.

## Trait agreement

Detection metrics say whether objects are found; for morphometrics it also
matters whether the *measurements* agree. The match table holds the traits
of both objects of every pair, so agreement is a few lines of pandas:

```python
table = matches.with_status("tp").to_frame()

ratio = table["pred_area"] / table["gt_area"]
ratio.median()                                  # multiplicative bias
table["pred_area"].corr(table["gt_area"])       # Pearson r
```

Give both albums the same physical scale with
[`with_scale`][hires.Collection.with_scale] before matching to obtain the
traits in physical units.

A ratio with a median away from one indicates a systematic offset, for
example halos included in the predicted masks. It can be corrected by a
constant factor, whereas a low correlation cannot.

The legacy module [`hires.analysis`](../reference/analysis.md) draws
distribution, bias, Bland-Altman, Taylor and target diagrams from the same
table. It needs the `analysis` extra and is scheduled to be replaced:

```python
hires.analysis.bland_altman(table)
```

## See also

- Reference: [`hires.metrics`](../reference/metrics.md),
  [`hires.plot`](../reference/plot.md)
- [Output files of `hires evaluate`](../reference/outputs.md#hires-evaluate)
