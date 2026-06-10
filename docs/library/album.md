# Album

`hires.models.album.Album`

A named container of [`Collection`](collection.md) objects — one collection per
image. All collection names must be unique.

```python
from hires.models import Album
```

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `collections` | `list[Collection]` | The collections |
| `album_name` | `str \| None` | Name for the whole dataset |

---

## Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `names` | `list[str]` | Names of all collections |
| `name_map` | `dict[str, Collection]` | Name → Collection lookup dict |

---

## Dunder behaviour

```python
len(album)              # number of collections
album[0]                # first collection (by index)
album["image_name"]     # collection by name
"image_name" in album   # membership test
for col in album:       # iterate over collections
```

---

## Methods

### `from_dir(txt_dir, ...)` (classmethod)

Load all `.txt` files in a directory as one Album. Optionally pair with images.

```python
from hires.models import Album

album = Album.from_dir(
    "results/",
    image_dir="data/images/",
    album_name="experiment_01",
)
print(len(album))   # number of images loaded
```

### `from_paths(txt_paths, ...)` (classmethod)

Load from an explicit list of `.txt` paths.

```python
from pathlib import Path

paths = list(Path("results/").glob("*.txt"))
album = Album.from_paths(paths, image_dir="data/images/", album_name="batch_1")
```

### `add(collection)` / `extend(collections)`

Add one or more collections (names must be unique).

```python
album.add(col)
album.extend([col_a, col_b])
```

### `remove(collection_name)`

Remove a collection by name.

```python
album.remove("bad_image")
```

### `get(collection_name, default=None)`

Safely retrieve a collection by name.

```python
col = album.get("image_01")
col = album.get("missing", default=None)
```

### `filter(names)`

Return a new `Album` containing only the named collections.

```python
subset = album.filter(["image_01", "image_02"])
```

### `set_scale(dpi, unit)`

Apply physical scale to every annotation in every collection.

```python
album.set_scale(dpi=300.0, unit="um")
```

### `class_counts()`

Total annotation counts per class across the whole album.

```python
counts = album.class_counts()
# Counter({0: 312, 1: 47})
```

### `summary()`

Returns a list of dicts — one per collection — with name, count, image path, and scale info.

```python
import pandas as pd
df = pd.DataFrame(album.summary())
print(df)
```

### `to_records()` / `to_dataframe()` / `to_csv(path)`

Export every annotation across all collections into a flat table.

```python
df = album.to_dataframe()
print(df.shape)          # (total_annotations, n_columns)
album.to_csv("all_shapes.csv")
```

### `to_txt(out_dir)`

Write each collection back to a `.txt` file under `out_dir`.

```python
album.to_txt("remapped_annotations/")
```

### `remap_classes(mapping, resolve=None)`

Return a new `Album` with class ids remapped across all collections. Accepts a
plain `{old_id: new_id}` dict or a `ClassMapping` (with an optional `resolve` for
ambiguous classes). See [Class remapping](class-remapping.md).

```python
remapped = album.remap_classes({0: 0, 1: 0, 2: 1})
```

### `save_crops(out_dir, ...)`

Save crops for every annotation in every collection. Each collection gets its own subfolder.

```python
saved = album.save_crops("all_crops/", use_mask=True, padding=5)
# saved = {"image_01": ["all_crops/image_01/image_01_0000_class_0.png", ...], ...}
```

---

## Full example

```python
from hires.models import Album

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
