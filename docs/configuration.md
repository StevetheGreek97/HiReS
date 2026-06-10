# Configuration Reference

All pipeline behaviour is controlled through the `Settings` dataclass.

```python
from hires.models import Settings
```

---

## Parameters

### Input / Output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str \| Path` | `"data"` | Path to an image file or directory of images |
| `model_path` | `str` | `"models/DaphnAI.pt"` | Path to YOLO `.pt` weights file |
| `output_dir` | `str` | `"results"` | Directory where all outputs are written |
| `ann` | `str` | `""` | Annotation file (or directory) for `PlottingPipeline` |
| `recursive` | `bool` | `False` | Recurse into subdirectories when `source` is a directory |

### Inference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conf` | `float` | `0.5` | YOLO confidence threshold (0–1) |
| `imgsz` | `int` | `1024` | Image size passed to YOLO inference |
| `device` | `str` | `"cpu"` | Compute device: `"cpu"`, `"cuda:0"`, `"mps"` |

### Chunking

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | `tuple[int, int]` | `(1024, 1024)` | Tile size in pixels `(width, height)` |
| `overlap` | `int` | `150` | Overlap between adjacent tiles in pixels |

### Post-processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `edge_threshold` | `float` | `0.01` | Inward inset of the normalised unit box; a polygon is kept only if it lies fully inside `box(0,0,1,1).buffer(-edge_threshold)`, discarding edge-touching detections |
| `iou_thresh` | `float` | `0.7` | IoU threshold for non-maximum suppression |

### Outputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_crops` | `bool` | `False` | Save per-object crops to `<out>/<image>_crops/` |
| `dpi` | `float \| None` | `None` | Image DPI for converting pixel measurements to physical units |
| `unit` | `str \| None` | `None` | Physical unit for shape descriptors: `"nm"`, `"um"`, `"mm"`, `"cm"`, `"m"`, `"inch"`. When `None`, measurements stay in pixels |

### Debug

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `debug` | `bool` | `False` | Save intermediate per-chunk annotations under `<out>/<image>_debug/` |

### Evaluation

These fields exist on `Settings` for evaluation workflows but are not used by the
`hires run` / `hires chunk` / `hires plot` commands.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `match_iou` | `float` | `0.5` | IoU threshold for matching predictions to ground truth |
| `pred_ann` | `str` | `""` | Path to a prediction annotation file for comparison |
| `gt_ann` | `str` | `""` | Path to a ground-truth annotation file for comparison |

---

## Example

```python
from hires.models import Settings

cfg = Settings(
    source="data/sample.tif",
    model_path="models/model.pt",
    output_dir="results/",
    conf=0.6,
    imgsz=1024,
    device="cuda:0",
    chunk_size=(1024, 1024),
    overlap=200,
    edge_threshold=0.01,
    iou_thresh=0.7,
    save_crops=True,
    dpi=300.0,
    unit="um",
    debug=False,
)
```
