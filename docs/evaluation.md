# Evaluation

The evaluation module compares a set of predicted annotations against ground-truth annotations. It produces `Pair` objects (one per detection decision), collects them in a `Bundle`, and supports export to DataFrames and CSV.

```
IoUMatrix  →  CollectionMatchMaker  →  Bundle[Pair, ...]
                      ↑
              AlbumMatchMaker  (many images at once)
```

All classes live under `hires.models.eval`.

---

## Pair

`hires.models.eval.pair.Pair`

An immutable record that represents one matching decision between a predicted annotation (left) and a ground-truth annotation (right).

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `left_ann` | `Annotation \| None` | The predicted annotation (`None` for FN) |
| `right_ann` | `Annotation \| None` | The ground-truth annotation (`None` for FP) |
| `iou` | `float` | Intersection-over-union between the two polygons (0.0 when unmatched) |
| `status` | `str` | Outcome: `"tp"`, `"fp"`, `"fn"`, or `"misclassified"` |

### Status meanings

| Status | Meaning |
|--------|---------|
| `"tp"` | Prediction matched a GT polygon with IoU ≥ threshold and same class |
| `"fp"` | Prediction had no matching GT polygon |
| `"fn"` | GT polygon had no matching prediction |
| `"misclassified"` | Prediction and GT polygon overlap (IoU ≥ threshold) but class ids differ |

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `is_tp` | `bool` | `True` when status is `"tp"` |
| `is_fp` | `bool` | `True` when status is `"fp"` or `"misclassified"` |
| `is_fn` | `bool` | `True` when status is `"fn"` or `"misclassified"` |
| `class_match` | `bool` | `True` when both annotations share the same class id |

### Methods

#### `from_annotations(...)` (classmethod)

Construct a `Pair` from index references into annotation lists. Used internally by `CollectionMatchMaker`.

#### `to_dict()`

Serialize the pair to a flat dict.

```python
pair.to_dict()
# {
#   "left_ann":  <Annotation ...>,
#   "right_ann": <Annotation ...>,
#   "iou":       0.83,
#   "status":    "tp",
# }
```

### Example

```python
from hires.models.eval.pair import Pair

pair = Pair(left_ann=pred_ann, right_ann=gt_ann, iou=0.83, status="tp")

print(pair.is_tp)        # True
print(pair.iou)          # 0.83
print(pair.class_match)  # True if both have same class_id
```

---

## Bundle

`hires.models.eval.bundle.Bundle`

An ordered container of `Pair` objects with filtering and export helpers.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `pairs` | `list[Pair]` | The contained pairs |

### Dunder behaviour

```python
len(bundle)       # number of pairs
bundle[0]         # first pair
bundle[1:5]       # slice → list of pairs
for pair in bundle:  # iterate
```

### Methods

#### `add(pair)` / `extend(pairs)`

Add one or many `Pair` objects.

```python
bundle.add(pair)
bundle.extend([pair_a, pair_b])
```

#### `filter(fn)`

Return a new `Bundle` keeping only pairs where `fn(pair)` returns `True`.

```python
tp_bundle = bundle.filter(lambda p: p.is_tp)
fn_bundle = bundle.filter(lambda p: p.is_fn)
```

#### `filter_by_iou(min_iou, max_iou)`

Filter by IoU range. Either bound can be omitted.

```python
high_iou  = bundle.filter_by_iou(min_iou=0.8)
low_iou   = bundle.filter_by_iou(max_iou=0.3)
mid_range = bundle.filter_by_iou(min_iou=0.5, max_iou=0.8)
```

#### `filter_by_status(statuses)`

Filter by one or more status values.

```python
tps = bundle.filter_by_status("tp")
errors = bundle.filter_by_status(["fp", "fn", "misclassified"])
```

#### `to_records()` / `to_dataframe()` / `to_csv(path)`

Export all pairs to a list of dicts, a pandas DataFrame, or a CSV file.

```python
records = bundle.to_records()
df = bundle.to_dataframe()
bundle.to_csv("evaluation.csv")
```

### Example

```python
from hires.models.eval.bundle import Bundle

# Assume `bundle` was produced by CollectionMatchMaker (see below)

print(len(bundle))  # total pairs

tp_bundle = bundle.filter_by_status("tp")
fp_bundle = bundle.filter_by_status("fp")
fn_bundle = bundle.filter_by_status("fn")

print(f"TP={len(tp_bundle)}  FP={len(fp_bundle)}  FN={len(fn_bundle)}")

precision = len(tp_bundle) / (len(tp_bundle) + len(fp_bundle))
recall    = len(tp_bundle) / (len(tp_bundle) + len(fn_bundle))

# Filter to high-confidence matches only
good = bundle.filter_by_iou(min_iou=0.75)

df = bundle.to_dataframe()
print(df.groupby("status").size())
bundle.to_csv("results.csv")
```

---

## IoUMatrix

`hires.models.eval.iou.IoUMatrix`

Computes pairwise intersection-over-union between all annotations in two `Collection` objects. Uses a Shapely `STRtree` for spatial indexing so only candidate pairs are evaluated.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `left` | `Collection` | Predictions |
| `right` | `Collection` | Ground truth |
| `return_dense` | `bool` | Whether to build a dense NumPy matrix (default `True`) |
| `values` | `dict[tuple[int,int], float]` | Sparse IoU results keyed by `(left_idx, right_idx)` |
| `dense` | `np.ndarray \| None` | Dense `(n_left, n_right)` matrix when `return_dense=True` |

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `shape` | `tuple[int, int]` | `(n_left_annotations, n_right_annotations)` |
| `left_annotations` | `list[Annotation]` | Annotations from the left collection |
| `right_annotations` | `list[Annotation]` | Annotations from the right collection |
| `left_polygons` | `list[Polygon]` | Polygons from the left collection |
| `right_polygons` | `list[Polygon]` | Polygons from the right collection |

### Methods

#### `require_dense()`

Return the dense matrix or raise if it was not built.

```python
matrix = iou_mat.require_dense()  # np.ndarray shape (n_left, n_right)
```

#### `compute_iou(a, b)` (staticmethod)

Compute IoU between two Shapely polygons directly.

```python
from hires.models.eval.iou import IoUMatrix

score = IoUMatrix.compute_iou(polygon_a, polygon_b)
```

#### `to_dict()`

Return a dict with `values`, `dense`, and `shape`.

### Example

```python
from hires.models.collection import Collection
from hires.models.eval.iou import IoUMatrix

preds = Collection.read_txt("results/image.txt")
gt    = Collection.read_txt("ground_truth/image.txt")

iou_mat = IoUMatrix(preds, gt, return_dense=True)

print(iou_mat.shape)          # (n_preds, n_gt)

matrix = iou_mat.require_dense()
print(matrix.max(axis=1))     # best GT match for each prediction
```

---

## CollectionMatchMaker

`hires.models.eval.match_maker.CollectionMatchMaker`

Matches the annotations in two `Collection` objects (predictions vs. ground truth) using greedy IoU matching and produces a list of `Pair` outcomes.

The matching algorithm:
1. Build a full IoU matrix between left (predictions) and right (ground truth).
2. Sort all overlapping pairs by IoU descending.
3. Greedily assign each prediction and GT polygon to at most one match.
4. Unmatched predictions → `"fp"`, unmatched GT polygons → `"fn"`.
5. Matched pairs with mismatched class ids → `"misclassified"`.

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `left` | `Collection` | — | Predictions |
| `right` | `Collection` | — | Ground truth |
| `iou_threshold` | `float` | `0.5` | Minimum IoU to count as a match |
| `class_aware` | `bool` | `False` | Skip misclassified matches (treat as FP+FN instead) |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `pairs_list()` | `list[Pair]` | All pairs as a list |
| `to_records()` | `list[dict]` | Serialized pairs |
| `to_dataframe()` | `DataFrame` | Pairs as a pandas DataFrame |
| `to_bundle()` | `Bundle` | Wrap pairs in a `Bundle` |

### Example

```python
from hires.models.collection import Collection
from hires.models.eval.match_maker import CollectionMatchMaker

preds = Collection.read_txt("results/image.txt")
gt    = Collection.read_txt("ground_truth/image.txt")

matcher = CollectionMatchMaker(preds, gt, iou_threshold=0.5)
bundle  = matcher.to_bundle()

tp = bundle.filter_by_status("tp")
fp = bundle.filter_by_status("fp")
fn = bundle.filter_by_status("fn")

print(f"TP={len(tp)}  FP={len(fp)}  FN={len(fn)}")

precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0.0
recall    = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

df = matcher.to_dataframe()
print(df[["iou", "status"]].head())
```

---

## AlbumMatchMaker

`hires.models.eval.match_maker.AlbumMatchMaker`

Runs `CollectionMatchMaker` across every shared image in two `Album` objects and returns a single flat `Bundle` of all pairs.

Collections are matched by `collection_name`. Images present in only one album are skipped (or raise an error when `strict=True`).

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `left` | `Album` | — | Predictions album |
| `right` | `Album` | — | Ground-truth album |
| `iou_threshold` | `float` | `0.5` | Passed to each `CollectionMatchMaker` |
| `class_aware` | `bool` | `False` | Passed to each `CollectionMatchMaker` |
| `strict` | `bool` | `False` | Raise if the two albums don't have identical names |

### Methods

#### `bundle()`

Run matching on all shared collections and return a flat `Bundle`.

```python
bundle = album_matcher.bundle()
```

### Example

```python
from hires.models.album import Album
from hires.models.eval.match_maker import AlbumMatchMaker

preds_album = Album.from_dir("results/",        image_dir="data/images/", album_name="preds")
gt_album    = Album.from_dir("ground_truth/",   image_dir="data/images/", album_name="gt")

matcher = AlbumMatchMaker(preds_album, gt_album, iou_threshold=0.5)
bundle  = matcher.bundle()

tp = bundle.filter_by_status("tp")
fp = bundle.filter_by_status("fp")
fn = bundle.filter_by_status("fn")

print(f"Dataset totals — TP={len(tp)}  FP={len(fp)}  FN={len(fn)}")

precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0.0
recall    = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0.0

print(f"Precision={precision:.3f}  Recall={recall:.3f}")

# Export full result table
df = bundle.to_dataframe()
bundle.to_csv("dataset_evaluation.csv")

# High-IoU true positives only
strong_tp = bundle.filter_by_iou(min_iou=0.8).filter_by_status("tp")
print(f"Strong TPs (IoU ≥ 0.8): {len(strong_tp)}")
```

---

## Performance

`hires.models.eval.performance.Performance`

Computes detection metrics across two `Album` objects (ground truth and
predictions) by running `CollectionMatchMaker` on each shared image.

### Constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `gt_album` | `Album` | Ground-truth album |
| `pred_album` | `Album` | Predictions album |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `confusion_matrix(iou_threshold=0.5, background="background")` | `DataFrame` | GT (rows) × Pred (cols) confusion matrix; unmatched objects fall into the `background` row/column |
| `per_class_report(iou_threshold=0.5, map_thresholds=None)` | `DataFrame` | Per-class `tp`, `fp`, `fn`, `precision`, `recall`, `f1`, `mAP@0.5`, `mAP@0.5-0.95`, plus an `all` summary row |
| `map_at(iou_threshold=0.5)` | `DataFrame` | Average precision per class at a single IoU threshold |

```python
from hires.models.album import Album
from hires.models.eval.performance import Performance

gt    = Album.from_dir("ground_truth/", image_dir="images/", album_name="gt")
preds = Album.from_dir("results/",      image_dir="images/", album_name="preds")

perf = Performance(gt_album=gt, pred_album=preds)
print(perf.per_class_report(iou_threshold=0.5))
cm = perf.confusion_matrix(iou_threshold=0.5)
```

The confusion matrix can be rendered with
`hires.viz.plot_confusion_matrix(cm, class_names)`.

---

## PairInspector

`hires.models.eval.inspector.PairInspector`

Produces a descriptor-enriched pair table across two `Album` objects. Each row
is one matching decision with the geometry of both the ground-truth (`left_*`)
and predicted (`right_*`) annotation attached.

```python
from hires.models.eval.inspector import PairInspector

inspector = PairInspector(gt_album=gt, pred_album=preds)
pairs = inspector.pairs_df(iou_threshold=0.5)
```

`pairs_df` puts ground truth on the left, so its `status` values are remapped to
`"tp"`, `"missed_gt"` (unmatched GT), `"fp"` (unmatched prediction), and
`"misclassified"`. This table is the input expected by the trait-comparison
functions in `hires.analysis`.

---

## Trait analysis

The `hires.analysis` module turns a `PairInspector.pairs_df` table into
GT-vs-prediction trait comparisons: `distributions`, `bias`, `bland_altman`,
`taylor_plot`, `target_diagram`, and `per_sample_abundance`. These functions
require the optional `plotnine` dependency (and `SkillMetrics` for the Taylor
diagram).

```python
from hires.analysis import distributions, bland_altman

processed, plot = distributions(pairs, class_names={0: "D_pulex", 1: "S_vetulus"})
```

---

## Putting it all together

```python
from hires.models.album import Album
from hires.models.eval.match_maker import AlbumMatchMaker

# 1. Load predictions and ground truth
preds = Album.from_dir("results/",       image_dir="images/", album_name="predictions")
gt    = Album.from_dir("ground_truth/",  image_dir="images/", album_name="ground_truth")

# 2. Match
matcher = AlbumMatchMaker(preds, gt, iou_threshold=0.5, class_aware=False)
bundle  = matcher.bundle()

# 3. Count outcomes
tp = len(bundle.filter_by_status("tp"))
fp = len(bundle.filter_by_status("fp"))
fn = len(bundle.filter_by_status("fn"))
mc = len(bundle.filter_by_status("misclassified"))

# 4. Metrics
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall    = tp / (tp + fn) if (tp + fn) else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

print(f"TP={tp}  FP={fp}  FN={fn}  Misclassified={mc}")
print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

# 5. Save
bundle.to_csv("full_evaluation.csv")
```
