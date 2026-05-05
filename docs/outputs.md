# Outputs

For each processed image, HiReS writes the following files to `output_dir`.

---

## Annotation file

**Path:** `<out>/<image>.txt`

YOLO segmentation format — one object per line:

```
<class_id> x1 y1 x2 y2 ... xN yN [confidence]
```

Coordinates are normalized (0–1) relative to the full image dimensions. The optional confidence column is appended when available.

---

## Overlay image

**Path:** `<out>/<image>_annotated.tif`

The source image with all detected polygon outlines drawn on top, coloured by class. Useful for quick visual QA.

---

## Per-object crops

**Path:** `<out>/<image>_crops/<index>_class<id>.tif`

A tight crop around each detected object with the polygon mask applied. Written when `save_crops=True` (default).

---

## Shape descriptor table

**Path:** `<out>/<image>_shapes.csv`

One row per detected object. Columns:

| Column | Description |
|--------|-------------|
| `index` | Object index (matches crop filename) |
| `class_id` | Class index |
| `confidence` | Detection confidence score |
| `area` | Polygon area |
| `perimeter` | Polygon perimeter |
| `circularity` | 4π·area / perimeter² (1.0 = perfect circle) |
| `solidity` | area / convex hull area (1.0 = fully convex) |
| `obb_width` | Oriented bounding box width (short axis) |
| `obb_height` | Oriented bounding box height (long axis) |
| `obb_angle` | OBB rotation angle (degrees) |

Units are pixels by default. If `dpi` and `unit` are set in `Settings`, the area and length columns are converted to the specified physical unit.

---

## Debug artifacts

**Path:** `<out>/<image>_debug/`

Written only when `debug=True`. Contains per-chunk annotation `.txt` files before edge filtering and merging — useful for diagnosing missed detections or NMS behaviour.

---

## Compare outputs (`hires compare`)

| File | Contents |
|------|----------|
| `<image>_compare_overlay.tif` | All predictions and GT polygons colour-coded by TP/FP/FN |
| `<image>_compare_tp.tif` | True positives: matched predictions with GT outlines |
| `<image>_compare_fp.tif` | False positives: unmatched predictions |
| `<image>_compare_fn.tif` | False negatives: unmatched ground-truth polygons |
| `<image>_compare_summary.json` | `{"tp": N, "fp": N, "fn": N, "matches": [[pred_i, gt_j], ...]}` |
