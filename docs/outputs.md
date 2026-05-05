# Outputs

For each processed image, HiReS writes output files to `output_dir`. This page describes each file and its contents.

---

## Annotation file

**Path:** `<out>/<image>.txt`

YOLO segmentation format — one object per line:

```
<class_id>  x1 y1  x2 y2  ...  xN yN  [confidence]
```

Coordinates are **normalized** (0–1) relative to the full image dimensions. The optional confidence value is appended when available. These files can be loaded directly with `Collection.read_txt()`.

---

## Overlay image

**Path:** `<out>/<image>_annotated.tif`

The source image with all detected polygon outlines drawn on top, coloured by class. Useful for quick visual quality assurance and publication figures.

---

## Per-object crops

**Path:** `<out>/<image>_crops/<index>_class<id>.tif`

A tight bounding-box crop around each detected object. Written when `save_crops=True` in `Settings` (default `False`).

---

## Shape descriptor table

**Path:** `<out>/<image>_shapes.csv`

One row per detected object. This is the primary output for downstream statistical analysis.

| Column | Unit | Description |
|--------|------|-------------|
| `index` | — | Object index (matches crop filename) |
| `class_id` | — | Class integer assigned by the model |
| `confidence` | — | Detection confidence score (0–1) |
| `area` | px² | Polygon area (Shoelace formula) |
| `perimeter` | px | Polygon perimeter (sum of edge lengths) |
| `circularity` | — | 4π·area / perimeter² (1.0 = perfect circle) |
| `convexity` | — | Convex hull perimeter / polygon perimeter |
| `solidity` | — | Polygon area / convex hull area |
| `obb_width` | px | Oriented bounding box short axis (body width) |
| `obb_height` | px | Oriented bounding box long axis (body length) |
| `obb_angle` | deg | OBB rotation angle |

!!! info "Physical units"
    All measurements are in **pixel units** by default. To convert to physical units (µm, mm, …), set `dpi` and `unit` in `Settings` or call `collection.set_scale(dpi=..., unit=...)` after loading. See [Morphometric Descriptors — Physical unit conversion](morphometrics.md#physical-unit-conversion) for the conversion factors.

### Loading the CSV

```python
import pandas as pd
df = pd.read_csv("results/image_shapes.csv")
print(df[["class_id", "area", "obb_height", "circularity"]].describe())
```

Or load directly through HiReS:

```python
from hires.models.collection import Collection

col = Collection.read_txt("results/image.txt")
df = col.to_df()
df.to_csv("summary.csv", index=False)
```

---

## Debug artifacts

**Path:** `<out>/<image>_debug/`

Written only when `debug=True` in `Settings`. Contains per-tile annotation `.txt` files **before** edge filtering and coordinate unification. Useful for diagnosing:

- Missed detections at tile boundaries
- Edge filter aggressiveness (`edge_threshold`)
- NMS behaviour across overlapping tiles

---

## Compare outputs (`hires compare`)

Running `hires compare` generates colour-coded overlay images and a summary JSON for evaluating predictions against a ground-truth annotation file.

| File | Contents |
|------|----------|
| `<image>_compare_overlay.tif` | All predictions and GT polygons colour-coded by TP / FP / FN |
| `<image>_compare_tp.tif` | True positives: matched predictions with GT outlines |
| `<image>_compare_fp.tif` | False positives: unmatched predictions |
| `<image>_compare_fn.tif` | False negatives: unmatched ground-truth polygons |
| `<image>_compare_summary.json` | `{"tp": N, "fp": N, "fn": N, "matches": [[pred_i, gt_j], ...]}` |

See [Evaluation](evaluation.md) for computing precision, recall, and F1 from these outputs programmatically.
