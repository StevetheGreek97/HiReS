# Evaluation

The evaluation API compares predicted annotations against ground-truth
annotations. It matches them into `Pair` outcomes, aggregates detection metrics,
and feeds trait-level comparisons.

```
IoUMatrix  →  CollectionMatchMaker  →  Bundle[Pair, ...]
                      ↑
              AlbumMatchMaker  (many images at once)
```

Most classes live under `hires.models.eval`; the trait-comparison functions live
under `hires.analysis`.

---

## In this section

| Page | Covers |
|------|--------|
| [Matching & pairs](matching.md) | `Pair`, `Bundle`, `IoUMatrix`, `CollectionMatchMaker`, `AlbumMatchMaker` |
| [Metrics](metrics.md) | `Performance` (precision/recall/F1/mAP, confusion matrix) and `PairInspector` (descriptor-enriched pair table) |
| [Trait analysis](analysis.md) | GT-vs-prediction trait plots in `hires.analysis` |

---

## Putting it all together

```python
from hires.models import Album
from hires.models.eval import AlbumMatchMaker

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

For ready-made per-class precision/recall/F1/mAP tables and a confusion matrix,
use [`Performance`](metrics.md#performance) instead of counting by hand.
