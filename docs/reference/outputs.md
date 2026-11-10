# Output files

Every file written by the `hires` command and by the Python functions
behind it. `{stem}` is the file name of an image without its extension.

## hires run

Written by `hires run` and [`hires.run_pipeline`][hires.run_pipeline].
With a directory source and `--recursive`, the sub-directory layout of the
source is mirrored below the output directory.

```text
results/
├── run_config.yaml
├── scan_01.txt
├── scan_01_shapes.csv
├── scan_01_annotated.tif
├── scan_01_crops/            (--save-crops)
│   ├── scan_01_0000_class_0.png
│   └── ...
└── scan_01_debug/            (--debug)
```

| File | Content |
|------|---------|
| `{stem}.txt` | The detections as YOLO segmentation labels with confidences |
| `{stem}_shapes.csv` | One row per detection with its morphometric traits |
| `{stem}_annotated.tif` | The image with the detections drawn on top |
| `{stem}_crops/` | One masked crop per detection |
| `{stem}_debug/` | Intermediate results; see [Debug files](#debug-files) |
| `run_config.yaml` | The settings of the run |

### Label file

One line per detection, in the YOLO segmentation format with the confidence
appended:

```text
class_id x1 y1 x2 y2 ... xn yn confidence
```

Coordinates are normalized to the image: `x` is divided by the image width
and `y` by the image height. The line number, counted from zero, is the
index drawn next to each object in the overlay and used in the crop file
names. See [Reading and writing](../user-guide/io.md#the-label-format).

### Shape table

| Column | Content |
|--------|---------|
| `name` | Image stem |
| `image_path` | Path of the image |
| `image_width`, `image_height` | Image size in pixels |
| `dpi`, `unit` | Physical scale; empty without `--dpi` and `--unit` |
| `class_id` | Predicted class |
| `confidence` | Detection score |
| `area`, `perimeter` | In `unit`² and `unit`, or px² and px without a scale |
| `solidity`, `convexity`, `circularity` | Dimensionless shape descriptors |
| `bbox_width`, `bbox_height` | Sides of the axis-aligned bounding box |
| `obb_width`, `obb_length` | Short and long side of the minimum rotated rectangle |

Rows are in the same order as the lines of the label file. The traits are
defined in [Morphometric traits](../user-guide/morphometrics.md).

### Crops

`{stem}_crops/{stem}_{index:04d}_class_{class_id}.png`: the bounding box of
each detection, with the pixels outside the polygon set to black.

### Run configuration

`run_config.yaml` records what was run, so that results stay reproducible:

```yaml
timestamp: '2026-11-10T14:03:22'
n_images: 12
settings:
  source: [scans]
  output_dir: results
  recursive: false
  dpi: 2400.0
  unit: um
  save_overlay: true
  save_crops: false
  debug: false
  # ... followed by the tiling and inference settings of the Segmenter
```

### Debug files

Written to `{stem}_debug/` with `--debug`, `run_pipeline(debug=True)` or
`Segmenter.segment(debug_dir=...)`:

| Path | Content |
|------|---------|
| `tiles/{stem}_{x}_{y}.png` | Every tile; `(x, y)` is its top-left corner in the image |
| `tile_labels/{stem}_{x}_{y}.txt` | Raw predictions of the tile, normalized to the tile, before edge filtering |
| `tile_overlays/{stem}_{x}_{y}.png` | Raw predictions drawn on the tile; only for tiles with detections |
| `{stem}_merged.txt` | Edge-filtered detections in full-image coordinates, before NMS |

How to read them is described in
[Tiled segmentation](../user-guide/segmentation.md#debugging).

## hires chunk

`{output}/{stem}_{x}_{y}.png` for every tile, where `(x, y)` are the pixel
coordinates of the top-left corner of the tile in the image. Tiles at the
right and bottom border are zero-padded to the full tile size. Bit depth
and channels of the source image are preserved.

## hires plot

`{output}/{stem}_annotated.png`: the image with filled polygons, bounding
boxes and a label `index class confidence` per object.

## hires evaluate

Written by `hires evaluate` and [`hires.evaluate`][hires.evaluate]:

```text
evaluation/
├── detection_report.csv
├── confusion_matrix.csv
├── confusion_matrix.png
├── precision_recall.csv
├── precision_recall.png
├── matches.csv
└── comparisons/
    ├── scan_01.png
    └── ...
```

| File | Content |
|------|---------|
| `detection_report.csv` | Per class and overall: `tp`, `fp`, `fn`, `precision`, `recall`, `f1`, `ap50`, `ap50_95` |
| `confusion_matrix.csv` | Counts with rows = ground truth and columns = prediction; the last row and column are `background` |
| `confusion_matrix.png` | The matrix as a heat map, normalized per ground-truth class |
| `precision_recall.csv` | `confidence`, `precision`, `recall` and `f1` for every confidence threshold, all classes pooled |
| `precision_recall.png` | The curve, with the best-F1 point marked |
| `matches.csv` | One row per matched pair, false positive and false negative |
| `comparisons/{name}.png` | The match outcomes drawn on each image; needs `--images` |

### Match table

| Column | Content |
|--------|---------|
| `image` | Collection name |
| `status` | `tp`, `misclassified`, `fp` or `fn` |
| `iou` | IoU of the pair; `0` when unmatched |
| `class_match` | Whether the pair agrees on the class |
| `gt_index`, `pred_index` | Line numbers of the two objects in their label files, from zero |
| `gt_class`, `pred_class` | Class ids |
| `gt_class_name`, `pred_class_name` | Class labels; only with `--names` |
| `gt_confidence`, `pred_confidence` | Confidences; ground truth usually has none |
| `gt_{trait}`, `pred_{trait}` | The nine traits of each object |

Fields of a missing object are empty: a false positive has no `gt_*`
values, a false negative no `pred_*` values. The traits are in pixels when
`--images` is given and in normalized units otherwise.

### Comparison images

| Colour | Status |
|--------|--------|
| Green | True positive |
| Orange | Misclassified: matched, but with a different class |
| Red | False positive |
| Blue | False negative (the missed ground-truth outline) |
