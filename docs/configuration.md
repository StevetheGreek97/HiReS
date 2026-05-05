# Configuration Reference

All pipeline behaviour is controlled through the `Settings` dataclass.

```python
from hires import Settings
```

---

## Parameters

### Input / Output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str \| Path` | — | Path to an image file or directory of images |
| `model_path` | `str \| Path \| None` | `None` | Path to YOLO `.pt` weights file |
| `output_dir` | `str \| Path` | — | Directory where all outputs are written |
| `ann` | `str \| Path \| None` | `None` | Annotation file for `PlottingPipeline` |
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
| `edge_threshold` | `float` | `0.01` | Fraction of perimeter allowed to touch a tile edge before a polygon is discarded |
| `iou_thresh` | `float` | `0.7` | IoU threshold for non-maximum suppression |

### Outputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_crops` | `bool` | `True` | Save per-object crops to `<out>/<image>_crops/` |
| `dpi` | `float \| None` | `None` | Image DPI for converting pixel measurements to physical units |
| `unit` | `str` | `"px"` | Physical unit for shape descriptors: `"px"`, `"nm"`, `"um"`, `"mm"`, `"cm"`, `"m"`, `"inch"` |

### Debug

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `debug` | `bool` | `False` | Save intermediate per-chunk annotations under `<out>/<image>_debug/` |

---

## Example

```python
from hires import Settings

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
