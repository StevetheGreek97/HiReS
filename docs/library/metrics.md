# Metrics

Higher-level helpers that aggregate matched pairs into detection metrics and
descriptor-enriched tables across whole datasets.

---

## Performance

`hires.models.eval.Performance`

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
from hires.models import Album
from hires.models.eval import Performance

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

`hires.models.eval.PairInspector`

Produces a descriptor-enriched pair table across two `Album` objects. Each row
is one matching decision with the geometry of both the ground-truth (`left_*`)
and predicted (`right_*`) annotation attached.

```python
from hires.models.eval import PairInspector

inspector = PairInspector(gt_album=gt, pred_album=preds)
pairs = inspector.pairs_df(iou_threshold=0.5)
```

`pairs_df` puts ground truth on the left, so its `status` values are remapped to
`"tp"`, `"missed_gt"` (unmatched GT), `"fp"` (unmatched prediction), and
`"misclassified"`. This table is the input expected by the
[trait-analysis](analysis.md) functions in `hires.analysis`.
