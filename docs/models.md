# Data Models

HiReS represents segmentation results as three nested classes:

```
Album  →  holds many  →  Collection  →  holds many  →  Annotation
```

- An **Annotation** is a single detected polygon.
- A **Collection** is all annotations for one image.
- An **Album** groups collections for a whole dataset.

---

## Annotation

`hires.models.annotation.Annotation`

A single segmented polygon with its class, confidence score, and derived geometry.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `class_id` | `int` | Class index |
| `polygon` | `shapely.Polygon` | Polygon geometry (normalized coords 0–1) |
| `confidence` | `float \| None` | Detection confidence score |
| `dpi` | `float \| None` | Image DPI (set via `set_scale`) |
| `unit` | `str \| None` | Physical unit (set via `set_scale`) |

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `scale` | `float` | Pixel-to-unit conversion factor (1.0 when no DPI set) |
| `area` | `float` | Polygon area (scaled if DPI set) |
| `perimeter` | `float` | Polygon perimeter (scaled if DPI set) |
| `convex_hull` | `Polygon` | Convex hull polygon (pixel coords, unscaled) |
| `convex_hull_area` | `float` | Convex hull area (scaled if DPI set) |
| `convex_hull_perimeter` | `float` | Convex hull perimeter (scaled if DPI set) |
| `solidity` | `float` | area / convex hull area — 1.0 = fully convex |
| `convexity` | `float` | convex hull perimeter / perimeter |
| `circularity` | `float` | 4π·area / perimeter² — 1.0 = perfect circle |
| `bounding_box` | `BoundingBox` | Axis-aligned bounding box |
| `oriented_bounding_box` | `OrientedBoundingBox \| None` | Minimum rotated rectangle |

### Methods

#### `set_scale(dpi, unit)`

Attach physical scale to the annotation so that `area` and `perimeter` are returned in real-world units.

Supported units: `"nm"`, `"um"`, `"mm"`, `"cm"`, `"m"`, `"inch"`.

```python
ann.set_scale(dpi=300.0, unit="um")
print(ann.area)       # area in μm²
print(ann.perimeter)  # perimeter in μm
```

#### `iou(other)`

Compute intersection-over-union with another annotation.

```python
overlap = ann_a.iou(ann_b)  # float between 0.0 and 1.0
```

#### `is_inside_unit_box(threshold=1e-4)`

Returns `True` if the polygon fits within the normalised [0, 1] unit box (with an optional inset margin). Used to filter out edge-touching polygons.

```python
if ann.is_inside_unit_box(threshold=0.01):
    print("polygon is safely inside the tile")
```

#### `plot(...)`

Visualise the polygon with optional bounding-box and OBB overlays.

```python
ann.plot(show=True)                        # polygon only
ann.plot(obb=True, dims=True, show=True)   # polygon + OBB with dimension arrows
ann.plot(box=True, dims=True, show=True)   # polygon + axis-aligned box with dimensions
```

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `obb` | `False` | Draw the oriented bounding box |
| `box` | `False` | Draw the axis-aligned bounding box |
| `dims` | `False` | Annotate width/length on the bounding box (requires `obb` or `box=True`) |
| `padding` | `0.5` | Whitespace fraction around the polygon |
| `ax` | `None` | Existing matplotlib Axes to draw on |
| `show` | `False` | Call `plt.show()` after drawing |
| `tight` | `True` | Call `tight_layout()` |
| `clear_ax` | `False` | Clear the axes before drawing |

#### `to_dict()`

Serialize the annotation's measurements to a flat dictionary.

```python
d = ann.to_dict()
# {
#   'class_id': 0, 'confidence': 0.92,
#   'area': 1234.5, 'perimeter': 145.2,
#   'solidity': 0.97, 'convexity': 0.99, 'circularity': 0.74,
#   'bbox_width': 42.1, 'bbox_height': 38.5,
#   'obb_width': 35.0, 'obb_length': 44.7,
# }
```

### Full example

```python
from shapely.geometry import Polygon
from hires.models.annotation import Annotation

polygon = Polygon([(0.1, 0.1), (0.4, 0.1), (0.4, 0.5), (0.1, 0.5)])
ann = Annotation(class_id=0, polygon=polygon, confidence=0.91)

print(ann.area)         # pixel-based area
print(ann.circularity)  # shape roundness
print(ann.solidity)     # shape convexity

# Apply physical scale
ann.set_scale(dpi=300.0, unit="um")
print(ann.area)         # area in μm²

# Compare two annotations
other = Annotation(class_id=0, polygon=Polygon([(0.3, 0.3), (0.6, 0.3), (0.6, 0.7), (0.3, 0.7)]))
print(ann.iou(other))   # overlap fraction

# Visualise
ann.plot(obb=True, dims=True, show=True)

print(ann)  # human-readable summary
```

---

## Collection

`hires.models.collection.Collection`

An ordered container of `Annotation` objects for a single image.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `annotations` | `list[Annotation]` | The annotations |
| `collection_name` | `str \| None` | Usually the image stem |
| `image_path` | `Path \| str \| None` | Path to the source image |
| `dpi` | `float \| None` | DPI applied to all annotations |
| `unit` | `str \| None` | Unit applied to all annotations |

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `class_counts` | `dict[int, int]` | Count of annotations per class id |

### Dunder behaviour

```python
len(col)        # number of annotations
col[0]          # first annotation
col[1:5]        # slice → list of annotations
for ann in col: # iterate
```

### Methods

#### `read_txt(txt_path, ...)` (classmethod)

Load a YOLO-format `.txt` annotation file and return a `Collection`.

```python
from hires.models.collection import Collection

col = Collection.read_txt(
    "results/image.txt",
    collection_name="image",
    image_path="data/image.tif",
)
print(len(col))  # number of detected objects
```

#### `add(annotation)` / `extend(annotations)`

Add one or many annotations.

```python
col.add(ann)
col.extend([ann_a, ann_b])
```

#### `set_scale(dpi, unit)`

Apply physical scale to all annotations at once.

```python
col.set_scale(dpi=300.0, unit="um")
```

#### `filter(...)`

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

#### `nms(iou_threshold, class_aware)`

Apply non-maximum suppression and return a deduplicated `Collection`. All annotations must have a confidence score.

```python
clean = col.nms(iou_threshold=0.5)
clean = col.nms(iou_threshold=0.5, class_aware=True)  # suppress within same class only
```

#### `remap_classes(mapping)`

Return a new `Collection` with class ids remapped.

```python
remapped = col.remap_classes({0: 1, 2: 1})  # merge old classes 0 and 2 → new class 1
```

#### `to_records()` / `to_df()` / `to_csv(path)`

Export annotations as records, a pandas DataFrame, or a CSV file.

```python
records = col.to_records()   # list of dicts
df = col.to_df()             # pandas DataFrame
col.to_csv("shapes.csv")
```

#### `to_txt(output_path, include_conf=True)`

Write annotations back to YOLO segmentation `.txt` format.

```python
col.to_txt("results/image.txt")
col.to_txt("results/image.txt", include_conf=False)
```

#### `save_crops(out_dir, ...)`

Crop each annotation out of the source image and save the crops.

```python
paths = col.save_crops(
    "crops/",
    use_mask=True,   # mask pixels outside the polygon
    padding=10,      # extra pixels around the bounding box
    ext="png",
)
```

### Full example

```python
from hires.models.collection import Collection

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

---

## Album

`hires.models.album.Album`

A named container of `Collection` objects — one collection per image. All collection names must be unique.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `collections` | `list[Collection]` | The collections |
| `album_name` | `str \| None` | Name for the whole dataset |

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `names` | `list[str]` | Names of all collections |
| `name_map` | `dict[str, Collection]` | Name → Collection lookup dict |

### Dunder behaviour

```python
len(album)              # number of collections
album[0]                # first collection (by index)
album["image_name"]     # collection by name
"image_name" in album   # membership test
for col in album:       # iterate over collections
```

### Methods

#### `from_dir(txt_dir, ...)` (classmethod)

Load all `.txt` files in a directory as one Album. Optionally pair with images.

```python
from hires.models.album import Album

album = Album.from_dir(
    "results/",
    image_dir="data/images/",
    album_name="experiment_01",
)
print(len(album))   # number of images loaded
```

#### `from_paths(txt_paths, ...)` (classmethod)

Load from an explicit list of `.txt` paths.

```python
from pathlib import Path

paths = list(Path("results/").glob("*.txt"))
album = Album.from_paths(paths, image_dir="data/images/", album_name="batch_1")
```

#### `add(collection)` / `extend(collections)`

Add one or more collections (names must be unique).

```python
album.add(col)
album.extend([col_a, col_b])
```

#### `remove(collection_name)`

Remove a collection by name.

```python
album.remove("bad_image")
```

#### `get(collection_name, default=None)`

Safely retrieve a collection by name.

```python
col = album.get("image_01")
col = album.get("missing", default=None)
```

#### `filter(names)`

Return a new `Album` containing only the named collections.

```python
subset = album.filter(["image_01", "image_02"])
```

#### `set_scale(dpi, unit)`

Apply physical scale to every annotation in every collection.

```python
album.set_scale(dpi=300.0, unit="um")
```

#### `class_counts()`

Total annotation counts per class across the whole album.

```python
counts = album.class_counts()
# Counter({0: 312, 1: 47})
```

#### `summary()`

Returns a list of dicts — one per collection — with name, count, image path, and scale info.

```python
import pandas as pd
df = pd.DataFrame(album.summary())
print(df)
```

#### `to_records()` / `to_dataframe()` / `to_csv(path)`

Export every annotation across all collections into a flat table.

```python
df = album.to_dataframe()
print(df.shape)          # (total_annotations, n_columns)
album.to_csv("all_shapes.csv")
```

#### `to_txt(out_dir)`

Write each collection back to a `.txt` file under `out_dir`.

```python
album.to_txt("remapped_annotations/")
```

#### `remap_classes(mapping)`

Return a new `Album` with class ids remapped across all collections.

```python
remapped = album.remap_classes({0: 0, 1: 0, 2: 1})
```

#### `save_crops(out_dir, ...)`

Save crops for every annotation in every collection. Each collection gets its own subfolder.

```python
saved = album.save_crops("all_crops/", use_mask=True, padding=5)
# saved = {"image_01": ["all_crops/image_01/image_01_0000_class_0.png", ...], ...}
```

### Full example

```python
from hires.models.album import Album

# Load a full results directory
album = Album.from_dir(
    "results/",
    image_dir="data/images/",
    album_name="experiment_01",
)

print(album)                  # Album summary
print(album.names)            # ['image_01', 'image_02', ...]
print(album.class_counts())   # Counter({0: 312, 1: 47})

# Scale measurements
album.set_scale(dpi=300.0, unit="um")

# Access a specific image's results
col = album["image_01"]
print(len(col))              # annotations in image_01

# Work with a subset
subset = album.filter(["image_01", "image_03"])

# Flatten to a DataFrame for analysis
df = album.to_dataframe()
print(df.groupby("class_id")["area"].describe())

# Export
album.to_csv("all_shapes.csv")
album.to_txt("remapped/")
album.save_crops("all_crops/", use_mask=True, padding=5)

# Remap class ids and write back
remapped = album.remap_classes({0: 0, 1: 0})  # merge class 1 into 0
remapped.to_txt("merged_annotations/")
```
