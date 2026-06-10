# Python API

HiReS classes are imported from their submodules:

```python
from hires.models import Settings
from hires.pipeline.seg_pipeline import SegmentationPipeline
from hires.pipeline.chunk_pipeline import PlottingPipeline
from hires.processing.chunker import ImageChunker
```

---

## SegmentationPipeline

`hires.pipeline.seg_pipeline.SegmentationPipeline`

Runs the full pipeline: chunk → predict → filter → merge → NMS → outputs.

```python
from hires.models import Settings
from hires.pipeline.seg_pipeline import SegmentationPipeline

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
    save_crops=True,
    recursive=False,
    debug=False,
)

SegmentationPipeline(cfg).run()
```

`source` can be a single image path or a directory. When it is a directory, all
supported images (`.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`) are processed in
sequence (add `recursive=True` to descend into subdirectories).

---

## Tiling images

There is no dedicated chunking *pipeline* class — the `hires chunk` command and
the chunking stage both use `ImageChunker` directly. To tile images without
running inference:

```python
from hires.processing.chunker import ImageChunker

ImageChunker("data/images/").slice(
    save_folder="chunks/",
    chunk_size=(1024, 1024),
    overlap=150,
)
```

`ImageChunker` accepts a single image path or a directory. Each tile is written
as `{stem}_{x}_{y}.png`.

---

## PlottingPipeline

`hires.pipeline.chunk_pipeline.PlottingPipeline`

Renders existing YOLO annotations onto the source image.

```python
from hires.models import Settings
from hires.pipeline.chunk_pipeline import PlottingPipeline

cfg = Settings(
    source="data/images/image.tif",
    model_path="models/model.pt",
    output_dir="results/",
    ann="results/image.txt",
)

PlottingPipeline(cfg).run()
```

`ann` may be a single `.txt` file or a directory of `.txt` files (matched to
images by stem). If left empty, the pipeline looks for
`<output_dir>/<image_stem>.txt`.

---

## Settings

See the [Configuration reference](configuration.md) for all available parameters and their defaults.

---

## Working with annotations programmatically

Load a YOLO `.txt` file with `Collection.read_txt`. Pass `image_path` so that
pixel-space measurements and physical-unit scaling are available.

```python
from hires.models import Collection

collection = Collection.read_txt(
    "results/image.txt",
    image_path="data/image.tif",
)

for ann in collection:
    print(ann.area, ann.class_id, ann.confidence)
```

Individual `Annotation` objects expose Shapely polygon geometry directly:

```python
ann.polygon                 # shapely.geometry.Polygon (normalized 0–1 coords)
ann.area                    # polygon area (px², or calibrated unit² if scaled)
ann.perimeter               # polygon perimeter (px, or calibrated unit)
ann.circularity             # 4π·area / perimeter²
ann.solidity                # polygon area / convex-hull area
ann.oriented_bounding_box   # OrientedBoundingBox | None
```

The oriented bounding box exposes its corner coordinates and the
`(width, length)` of its two orthogonal sides:

```python
obb = ann.oriented_bounding_box
if obb is not None:
    width, length = obb.width_length
```

See [Data Models](models.md) for the full `Annotation`, `Collection`, and
`Album` reference.
