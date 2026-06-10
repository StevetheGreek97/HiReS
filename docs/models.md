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
| `image_width` | `int \| None` | Source image width in px (set via `set_scale`); needed to denormalise coordinates for pixel/physical measurements |
| `image_height` | `int \| None` | Source image height in px (set via `set_scale`) |

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
| `image_width` | `int \| None` | Source image width in px (read from `image_path` or set via `set_scale`) |
| `image_height` | `int \| None` | Source image height in px |

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

Apply physical scale to all annotations at once. The collection must know the
source image dimensions — load it with `image_path` set (or set `image_width` /
`image_height` directly), otherwise the call warns and leaves measurements
unscaled.

```python
col = Collection.read_txt("results/image.txt", image_path="data/image.tif")
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

#### `remap_classes(mapping, resolve=None)`

Return a new `Collection` with class ids remapped. `mapping` is either a plain
`{old_id: new_id}` dict or a `ClassMapping`; `resolve` chooses a target for any
ambiguous (split) class. See [Class remapping](#class-remapping) for the full
workflow.

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

#### `remap_classes(mapping, resolve=None)`

Return a new `Album` with class ids remapped across all collections. Accepts a
plain `{old_id: new_id}` dict or a `ClassMapping` (with an optional `resolve` for
ambiguous classes). See [Class remapping](#class-remapping).

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

---

## Class remapping

`hires.models.build_class_mapping` · `hires.models.ClassMapping`

`Collection.remap_classes` and `Album.remap_classes` rewrite the integer
`class_id` of every annotation. There are two ways to describe the remap:

1. **A plain `{old_id: new_id}` dict** — direct integer→integer renames. Any
   `class_id` not present in the dict is left unchanged.
2. **A `ClassMapping`** built with `build_class_mapping()` — a reusable,
   *label-aware* schema translation between two class-name dictionaries. This is
   the robust option when the source and target models use different class names
   and/or a different number of classes.

### `build_class_mapping(old_names, new_names, name_map)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `old_names` | `dict[int, str]` | `{id: label}` of the **source** schema (the IDs currently in your annotations) |
| `new_names` | `dict[int, str]` | `{id: label}` of the **target** schema (the IDs you want to end up with) |
| `name_map` | `dict[str, str \| list[str]]` | `{old_label: new_label}` for a direct rename, or `{old_label: [candidate_new_labels]}` when one old class can map to several new ones |

Returns a `ClassMapping`. It resolves each old label to a new **id** by looking
the chosen label up in `new_names`, so the two schemas can have completely
different ID orderings.

A list value marks an **ambiguous** entry: a single old class that could become
one of several new classes. You decide which one later, per-collection or
per-album, via the `resolve` argument of `remap_classes`. The chosen label must
be one of the declared candidates — picking anything else raises a `ValueError`.

### `ClassMapping`

| Attribute / Method | Description |
|--------------------|-------------|
| `mapping` | `{old_id: new_id}` for direct renames, `{old_id: [candidate_new_ids]}` for ambiguous ones |
| `old_names` / `new_names` | The two schemas it was built from |
| `flatten(resolve)` | Resolve all ambiguous entries to a flat `{old_id: new_id}` dict. `resolve` is `{old_label: chosen_new_label}` |

`ClassMapping` has a readable `repr`, which is handy for inspecting a mapping
before applying it.

### Example 1 — merge + resolve an ambiguous class

A generic detector with two classes (`ballooned`, `Daphnia`) is translated into
a finer-grained species schema. `ballooned` maps 1:1, but `Daphnia` is ambiguous
— it could be any of three species — so it is declared as a list and resolved
when the mapping is applied.

```python
from hires.models import build_class_mapping, Collection

class_names_old = {0: "ballooned", 1: "Daphnia"}
class_names_new = {0: "d_pulex", 1: "d_galeata", 2: "S_vetulus", 3: "ballooned"}

SCHEMA = {
    "ballooned": "ballooned",                          # 1:1 rename
    "Daphnia":   ["S_vetulus", "d_pulex", "d_galeata"],  # ambiguous → resolve later
}

full_mapping = build_class_mapping(class_names_old, class_names_new, SCHEMA)

print(full_mapping)
# ClassMapping(
#   0 ('ballooned') → 3 ('ballooned')
#   1 ('Daphnia') → [2 ('S_vetulus'), 0 ('d_pulex'), 1 ('d_galeata')]
# )

# The SAME mapping is reused across collections, resolving the ambiguous
# 'Daphnia' class to a different species for each sample you know the identity of:
s_vet     = Collection.read_txt("samples/s_vet.txt")
d_pulex   = Collection.read_txt("samples/d_pulex.txt")
d_galeata = Collection.read_txt("samples/d_galeata.txt")

s_vet_remapped     = s_vet.remap_classes(full_mapping,     resolve={"Daphnia": "S_vetulus"})
d_pulex_remapped   = d_pulex.remap_classes(full_mapping,   resolve={"Daphnia": "d_pulex"})
d_galeata_remapped = d_galeata.remap_classes(full_mapping, resolve={"Daphnia": "d_galeata"})
# In every case  class_id 0 (ballooned) → 3;
# class_id 1 (Daphnia) → 2 (S_vetulus) / 0 (d_pulex) / 1 (d_galeata) respectively.
```

Build the ambiguous mapping once, then resolve it per collection (or per album)
— there is no need to rebuild it for each species.

Omitting `resolve` for an ambiguous entry raises a `KeyError` that lists the
candidate labels, so you can never silently mis-assign a split class.

### Example 2 — per-sample relabelling across an Album

When you know the species of each sample up front, you can collapse several
fine-grained labels into the correct one per sample using a same-schema map
(`build_class_mapping(names, names, schema)`). Because every entry is a 1:1
string rename, no `resolve` is needed.

```python
from pathlib import Path
from hires.models import build_class_mapping, Album

# One shared label↔id schema for both sides of the remap.
names = {
    0: "Dg_f_lateral_adult",
    1: "Dp_f_lateral_adult",
    2: "Sv_f_lateral_adult",
    3: "Daphnia_f_lateral_juvenile",
    4: "Sv_f_lateral_juvenile",
    5: "chydoride",
    6: "copepod",
    7: "unidentified_Daphniidae",
}

# Each sample's adults/juveniles are forced to the correct species.
S_VET_SCHEMA = {
    "Dg_f_lateral_adult":         "Sv_f_lateral_adult",
    "Dp_f_lateral_adult":         "Sv_f_lateral_adult",
    "Sv_f_lateral_adult":         "Sv_f_lateral_adult",
    "Daphnia_f_lateral_juvenile": "Sv_f_lateral_juvenile",
    "Sv_f_lateral_juvenile":      "Sv_f_lateral_juvenile",
    "chydoride":                  "chydoride",
    "copepod":                    "copepod",
    "unidentified_Daphniidae":    "unidentified_Daphniidae",
}
D_GAL_SCHEMA = {
    "Dg_f_lateral_adult":         "Dg_f_lateral_adult",
    "Dp_f_lateral_adult":         "Dg_f_lateral_adult",
    "Sv_f_lateral_adult":         "Dg_f_lateral_adult",
    "Daphnia_f_lateral_juvenile": "Daphnia_f_lateral_juvenile",
    "Sv_f_lateral_juvenile":      "Daphnia_f_lateral_juvenile",
    "chydoride":                  "chydoride",
    "copepod":                    "copepod",
    "unidentified_Daphniidae":    "unidentified_Daphniidae",
}
D_PUL_SCHEMA = {
    "Dg_f_lateral_adult":         "Dp_f_lateral_adult",
    "Dp_f_lateral_adult":         "Dp_f_lateral_adult",
    "Sv_f_lateral_adult":         "Dp_f_lateral_adult",
    "Daphnia_f_lateral_juvenile": "Daphnia_f_lateral_juvenile",
    "Sv_f_lateral_juvenile":      "Daphnia_f_lateral_juvenile",
    "chydoride":                  "chydoride",
    "copepod":                    "copepod",
    "unidentified_Daphniidae":    "unidentified_Daphniidae",
}

base_path = Path("data")

# s_vet, d_gal, d_pul are lists of .txt paths for each species' samples.
for schema, species_paths, name in zip(
    [S_VET_SCHEMA, D_GAL_SCHEMA, D_PUL_SCHEMA],
    [s_vet, d_gal, d_pul],
    ["s_vet", "d_gal", "d_pul"],
):
    full_mapping = build_class_mapping(names, names, schema)

    sp_album = Album.from_paths(species_paths, album_name=name)
    print(sp_album.class_counts())            # before remap

    sp_album_remapped = sp_album.remap_classes(full_mapping)
    print(sp_album_remapped.class_counts())   # after remap

    sp_album_remapped.to_txt(out_dir=base_path / "comadapt_model_E06_remaped")
```

`Album.remap_classes` returns a **new** `Album` (the original is untouched) and
applies the same mapping to every collection. `to_txt` then writes one
`<collection_name>.txt` per sample under `out_dir`.

!!! tip "When to use a plain dict vs. `build_class_mapping`"
    Reach for a plain `{old_id: new_id}` dict for quick, in-schema merges
    (`album.remap_classes({0: 0, 1: 0})`). Use `build_class_mapping` when you are
    translating between two named schemas — it validates every label against
    `new_names` and turns split classes into explicit, resolvable choices.
