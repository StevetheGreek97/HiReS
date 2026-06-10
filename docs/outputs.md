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

## Run configuration

**Path:** `<out>/run_config.yaml`

A record of every `Settings` value used for the run, plus a timestamp and the
number of images processed. Useful for reproducibility.

---

## Per-object crops

**Path:** `<out>/<image>_crops/<image>_<index>_class_<id>.png`

A masked crop around each detected object (`index` is zero-padded, e.g. `0000`).
Written only when `save_crops=True` in `Settings` / the `--save-crops` CLI flag
is passed (default `False`).

---

## Shape descriptor table

**Path:** `<out>/<image>_shapes.csv`

One row per detected object. This is the primary output for downstream statistical analysis.

| Column | Unit | Description |
|--------|------|-------------|
| `collection_name` | — | Collection name (image stem) |
| `image_path` | — | Path to the source image |
| `image_width` | px | Source image width |
| `image_height` | px | Source image height |
| `dpi` | — | DPI used for scaling (blank if unset) |
| `unit` | — | Physical unit used for scaling (blank if unset) |
| `class_id` | — | Class integer assigned by the model |
| `confidence` | — | Detection confidence score (0–1) |
| `area` | px² | Polygon area (Shoelace formula) |
| `perimeter` | px | Polygon perimeter (sum of edge lengths) |
| `solidity` | — | Polygon area / convex hull area |
| `convexity` | — | Convex hull perimeter / polygon perimeter |
| `circularity` | — | 4π·area / perimeter² (1.0 = perfect circle) |
| `bbox_width` | px | Axis-aligned bounding box width |
| `bbox_height` | px | Axis-aligned bounding box height |
| `obb_width` | px | Oriented bounding box short axis (body width) |
| `obb_length` | px | Oriented bounding box long axis (body length) |

When `dpi`/`unit` are set, the length columns (`perimeter`, `bbox_*`, `obb_*`)
are reported in the chosen unit and `area` in unit². The `solidity`,
`convexity`, and `circularity` ratios are always dimensionless.

!!! info "Physical units"
    All measurements are in **pixel units** by default. To convert to physical units (µm, mm, …), set `dpi` and `unit` in `Settings` or call `collection.set_scale(dpi=..., unit=...)` after loading. See [Morphometric Descriptors — Physical unit conversion](morphometrics.md#physical-unit-conversion) for the conversion factors.

### Loading the CSV

```python
import pandas as pd
df = pd.read_csv("results/image_shapes.csv")
print(df[["class_id", "area", "obb_length", "circularity"]].describe())
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

Written only when `debug=True` in `Settings`. The debug directory mirrors the
internal stages for the image and contains:

- `chunks/` — the raw tile images
- `pred/` — per-tile prediction overlays
- `filtered/` and `filtered_txt/` — per-tile annotations after edge filtering
- `<image>_unified.txt` / `<image>_unified.png` — merged full-image annotations before NMS

Useful for diagnosing:

- Missed detections at tile boundaries
- Edge filter aggressiveness (`edge_threshold`)
- NMS behaviour across overlapping tiles

---

## Evaluating against ground truth

HiReS does not ship a CLI compare command. To compare predictions against
manual annotations programmatically, use the `hires.models.eval` API — see
[Evaluation](library/evaluation.md) for computing TP/FP/FN, precision, recall, and F1.
