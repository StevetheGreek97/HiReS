# Collection

`hires.models.collection.Collection`

An ordered container of [`Annotation`](annotation.md) objects for a single image.

```python
from hires.models import Collection
```

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `annotations` | `list[Annotation]` | The annotations |
| `collection_name` | `str \| None` | Usually the image stem |
| `image_path` | `Path \| str \| None` | Path to the source image |
| `dpi` | `float \| None` | DPI applied to all annotations |
| `unit` | `str \| None` | Unit applied to all annotations |
| `image_width` | `int \| None` | Source image width in px (read from `image_path` or set via `set_scale`) |
| `image_height` | `int \| None` | Source image height in px |

---

## Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `class_counts` | `dict[int, int]` | Count of annotations per class id |

---

## Dunder behaviour

```python
len(col)        # number of annotations
col[0]          # first annotation
col[1:5]        # slice → list of annotations
for ann in col: # iterate
```

---

## Methods

### `read_txt(txt_path, ...)` (classmethod)

Load a YOLO-format `.txt` annotation file and return a `Collection`.

```python
from hires.models import Collection

col = Collection.read_txt(
    "results/image.txt",
    collection_name="image",
    image_path="data/image.tif",
)
print(len(col))  # number of detected objects
```

### `add(annotation)` / `extend(annotations)`

Add one or many annotations.

```python
col.add(ann)
col.extend([ann_a, ann_b])
```

### `set_scale(dpi, unit)`

Apply physical scale to all annotations at once. The collection must know the
source image dimensions — load it with `image_path` set (or set `image_width` /
`image_height` directly), otherwise the call warns and leaves measurements
unscaled.

```python
col = Collection.read_txt("results/image.txt", image_path="data/image.tif")
col.set_scale(dpi=300.0, unit="um")
```

### `filter(...)`

Return a new `Collection` containing only matching annotations.

Three mutually exclusive modes:

```python
# by exact value
high_conf = col.filter(by="class_id", value=0)

# by condition function
large = col.filter(by="area", condition=lambda a: a > 500)

# by full predicate on the Annotation object
circular = col.filter(predicate=lambda ann: ann.circularity > 0.8)
```

### `nms(iou_threshold, class_aware)`

Apply non-maximum suppression and return a deduplicated `Collection`. All annotations must have a confidence score.

```python
clean = col.nms(iou_threshold=0.5)
clean = col.nms(iou_threshold=0.5, class_aware=True)  # suppress within same class only
```

### `remap_classes(mapping, resolve=None)`

Return a new `Collection` with class ids remapped. `mapping` is either a plain
`{old_id: new_id}` dict or a `ClassMapping`; `resolve` chooses a target for any
ambiguous (split) class. See [Class remapping](class-remapping.md) for the full
workflow.

```python
remapped = col.remap_classes({0: 1, 2: 1})  # merge old classes 0 and 2 → new class 1
```

### `to_records()` / `to_df()` / `to_csv(path)`

Export annotations as records, a pandas DataFrame, or a CSV file.

```python
records = col.to_records()   # list of dicts
df = col.to_df()             # pandas DataFrame
col.to_csv("shapes.csv")
```

### `to_txt(output_path, include_conf=True)`

Write annotations back to YOLO segmentation `.txt` format.

```python
col.to_txt("results/image.txt")
col.to_txt("results/image.txt", include_conf=False)
```

### `save_crops(out_dir, ...)`

Crop each annotation out of the source image and save the crops.

```python
paths = col.save_crops(
    "crops/",
    use_mask=True,   # mask pixels outside the polygon
    padding=10,      # extra pixels around the bounding box
    ext="png",
)
```

---

## Full example

```python
from hires.models import Collection

# Load from file
col = Collection.read_txt("results/image.txt", image_path="data/image.tif")

print(col.class_counts)          # {0: 42, 1: 7}
print(len(col))                  # 49

# Scale all annotations
col.set_scale(dpi=300.0, unit="um")

# Filter to a specific class
class0 = col.filter(by="class_id", value=0)

# Keep only high-confidence circular objects
filtered = col.filter(
    predicate=lambda ann: ann.confidence > 0.7 and ann.circularity > 0.75
)

# Deduplicate
clean = col.nms(iou_threshold=0.5)

# Export
df = clean.to_df()
print(df[["class_id", "confidence", "area", "circularity"]].head())
clean.to_csv("clean_shapes.csv")
clean.to_txt("clean.txt")

# Save individual crops
clean.save_crops("crops/", use_mask=True, padding=5)
```
