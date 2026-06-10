# Annotation

`hires.models.annotation.Annotation`

A single segmented polygon with its class, confidence score, and derived geometry.

```python
from hires.models import Annotation
```

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `class_id` | `int` | Class index |
| `polygon` | `shapely.Polygon` | Polygon geometry (normalized coords 0–1) |
| `confidence` | `float \| None` | Detection confidence score |
| `dpi` | `float \| None` | Image DPI (set via `set_scale`) |
| `unit` | `str \| None` | Physical unit (set via `set_scale`) |
| `image_width` | `int \| None` | Source image width in px (set via `set_scale`); needed to denormalise coordinates for pixel/physical measurements |
| `image_height` | `int \| None` | Source image height in px (set via `set_scale`) |

---

## Properties

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

See [Morphometric Descriptors](../morphometrics.md) for the mathematical
definitions, and [Geometry types](data-models.md#geometry-types) for the
bounding-box helpers.

---

## Methods

### `set_scale(dpi, unit)`

Attach physical scale to the annotation so that `area` and `perimeter` are returned in real-world units.

Supported units: `"nm"`, `"um"`, `"mm"`, `"cm"`, `"m"`, `"inch"`.

```python
ann.set_scale(dpi=300.0, unit="um")
print(ann.area)       # area in μm²
print(ann.perimeter)  # perimeter in μm
```

### `iou(other)`

Compute intersection-over-union with another annotation.

```python
overlap = ann_a.iou(ann_b)  # float between 0.0 and 1.0
```

### `is_inside_unit_box(threshold=1e-4)`

Returns `True` if the polygon fits within the normalised [0, 1] unit box (with an optional inset margin). Used to filter out edge-touching polygons.

```python
if ann.is_inside_unit_box(threshold=0.01):
    print("polygon is safely inside the tile")
```

### `plot(...)`

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

### `to_dict()`

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

---

## Full example

```python
from shapely.geometry import Polygon
from hires.models import Annotation

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
