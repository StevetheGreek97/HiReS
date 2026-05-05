# Python API

All public classes are importable from the top-level `hires` package.

```python
from hires import Settings, SegmentationPipeline, ChunkingPipeline, PlottingPipeline
```

---

## SegmentationPipeline

Runs the full pipeline: chunk → predict → filter → merge → NMS → outputs.

```python
from hires import Settings, SegmentationPipeline

cfg = Settings(
    source="data/images/",
    model_path="models/model.pt",
    output_dir="results/",
    conf=0.58,
    imgsz=1024,
    device="cpu",
    chunk_size=(1024, 1024),
    overlap=300,
    edge_threshold=0.01,
    iou_thresh=0.7,
    recursive=False,
    debug=False,
)

SegmentationPipeline(cfg).run()
```

`source` can be a single image path or a directory. When it is a directory, all supported images (`.tif`, `.tiff`, `.png`, `.jpg`) are processed in sequence.

---

## ChunkingPipeline

Tiles images without running inference — useful for preprocessing or inspecting chunk coverage.

```python
from hires import Settings, ChunkingPipeline

cfg = Settings(
    source="data/images/",
    output_dir="chunks/",
    chunk_size=(1024, 1024),
    overlap=150,
    recursive=True,
)

ChunkingPipeline(cfg).run()
```

---

## PlottingPipeline

Renders existing YOLO annotations onto the source image.

```python
from hires import Settings, PlottingPipeline

cfg = Settings(
    source="data/images/image.tif",
    model_path="models/model.pt",
    output_dir="results/",
    ann="results/image.txt",
)

PlottingPipeline(cfg).run()
```

---

## Settings

See the [Configuration reference](configuration.md) for all available parameters and their defaults.

---

## Working with annotations programmatically

```python
from hires.models.parser import parse_yolo_annotations
from hires.models.collection import Collection

collection = Collection(parse_yolo_annotations("results/image.txt", image_width=4096, image_height=4096))

for ann in collection:
    print(ann.area, ann.class_id, ann.confidence)
```

Individual `Annotation` objects expose Shapely polygon geometry directly:

```python
ann.polygon          # shapely.geometry.Polygon
ann.area             # polygon area in pixels²
ann.perimeter        # polygon perimeter in pixels
ann.circularity      # 4π·area / perimeter²
ann.solidity         # area / convex hull area
ann.obb              # OrientedBoundingBox (width, height, angle)
```
