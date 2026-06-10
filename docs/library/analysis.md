# Trait analysis

`hires.analysis`

The analysis module compares ground-truth and predicted **trait distributions**
(area, perimeter, OBB dimensions, …) rather than just counting detections. Every
function takes the descriptor-enriched pair table produced by
[`PairInspector.pairs_df`](metrics.md#pairinspector) and returns a tidy
`DataFrame` (and usually a plot object).

!!! warning "Optional dependencies"
    These functions require **`plotnine`** (`pip install plotnine`).
    `taylor_plot` additionally uses **`SkillMetrics`** (a core dependency).

```python
from hires.models.eval import PairInspector
from hires.analysis import distributions, bland_altman

pairs = PairInspector(gt_album=gt, pred_album=preds).pairs_df(iou_threshold=0.5)

class_names = {0: "d_pulex", 1: "d_galeata", 2: "S_vetulus"}
processed, plot = distributions(pairs, class_names=class_names)
```

---

## Functions

| Function | Returns | Purpose |
|----------|---------|---------|
| `distributions(data, ...)` | `(DataFrame, plot)` | GT vs Pred descriptor histograms / KDE, faceted by species × descriptor |
| `bias(data, ...)` | `DataFrame` | Percent difference `(Pred/GT − 1) × 100` as violin/box plots per species × descriptor |
| `bland_altman(data, ...)` | `(DataFrame, plot)` | Bland–Altman agreement grid (log scale by default) |
| `taylor_plot(data, ...)` | `(DataFrame, figure)` | Normalised Taylor diagram (correlation / std / centred RMSD) |
| `target_diagram(data, ...)` | `(DataFrame, figure)` | Normalised target diagram (bias vs centred RMSD) |
| `per_sample_abundance(data, ...)` | `DataFrame` | Total GT vs predicted object counts per sample |
| `plot_per_sample_abundance(data, ...)` | `(DataFrame, plot)` | Scatter of manual vs predicted abundance per sample |

`target_plot` is an alias of `target_diagram`.

---

## Common arguments

Most functions share these keyword arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `columns` | `None` | Descriptor columns to include (e.g. `["area", "perimeter"]`). `None` = all available |
| `only_tp` | `True` | Restrict to true-positive matched pairs with the same class |
| `align` | `"median"` | Centre/scale prediction values onto GT before plotting (`"median"`, `"mean"`, or `None`) |
| `log` | varies | Work in log space (base 10) where applicable |
| `class_names` | `None` | `{class_id: label}` for nicer facet labels |
| `exclude_classes` | `None` | Class ids/labels to drop |
| `file_names` / `samples` | `None` | Restrict to specific images / samples |
| `sample_n` | `None` | Randomly keep `n` samples (with `random_state`) |
| `show` | `True` | Display the plot |
| `save` | `None` | Path (file or directory) to write the figure to |

### Discovering available descriptors

```python
from hires.analysis import available_descriptor_columns

available_descriptor_columns(pairs)
# ['area', 'perimeter', 'solidity', 'convexity', 'circularity',
#  'bbox_w', 'bbox_h', 'obb_w', 'obb_l']
```

---

## Example

```python
from hires.models import Album
from hires.models.eval import PairInspector
from hires.analysis import distributions, bias, bland_altman, taylor_plot

gt    = Album.from_dir("ground_truth/", image_dir="images/", album_name="gt")
preds = Album.from_dir("results/",      image_dir="images/", album_name="preds")

pairs = PairInspector(gt_album=gt, pred_album=preds).pairs_df(iou_threshold=0.5)
class_names = {0: "d_pulex", 1: "d_galeata", 2: "S_vetulus"}

# Distribution overlays for two traits, saved to disk
distributions(
    pairs,
    columns=["area", "obb_l"],
    class_names=class_names,
    save="figs/distributions.png",
    show=False,
)

# Percentage bias table per species × descriptor
bias_df = bias(pairs, class_names=class_names, show=False)

# Bland–Altman agreement and a Taylor diagram
bland_altman(pairs, class_names=class_names, save="figs/bland_altman.png", show=False)
taylor_plot(pairs, class_names=class_names, save="figs/taylor.png", show=False)
```

!!! note "Input format"
    The functions expect the `left_*`/`right_*` columns from
    `PairInspector.pairs_df`; they convert these to the internal `gt_*`/`pred_*`
    naming automatically (via `hires.analysis.from_pairs_df`). You can also pass
    a table you have already converted.
