# Data Models

HiReS represents segmentation results as three nested classes:

```
Album  →  holds many  →  Collection  →  holds many  →  Annotation
```

- An **[Annotation](annotation.md)** is a single detected polygon.
- A **[Collection](collection.md)** is all annotations for one image.
- An **[Album](album.md)** groups collections for a whole dataset.

All three are importable from `hires.models`:

```python
from hires.models import Annotation, Collection, Album
```

---

## In this section

| Page | Covers |
|------|--------|
| [Annotation](annotation.md) | A single polygon and its derived geometry / shape descriptors |
| [Collection](collection.md) | A container of annotations for one image — loading, filtering, NMS, export |
| [Album](album.md) | A container of collections for a dataset — batch loading and export |
| [Class remapping](class-remapping.md) | Translate `class_id`s between label schemas with `build_class_mapping` |

---

## Geometry types

`Annotation` exposes two bounding-box helpers, both importable from
`hires.models`.

### `BoundingBox`

Axis-aligned bounds. `Annotation.bounding_box` returns one.

| Member | Type | Description |
|--------|------|-------------|
| `minx`, `miny`, `maxx`, `maxy` | `float` | Bounds |
| `width` / `height` | `float` | Extent along each axis |
| `center` | `tuple[float, float]` | Box centre `(x, y)` |
| `geometry` | `shapely.Polygon` | The box as a Shapely polygon |
| `from_polygon(polygon)` | classmethod | Build from a polygon's bounds |
| `contains(other)` / `covers(other)` | `bool` | Spatial relationship with another box |

### `OrientedBoundingBox`

Minimum-area rotated rectangle. `Annotation.oriented_bounding_box` returns one
(or `None` for an empty/invalid polygon).

| Member | Type | Description |
|--------|------|-------------|
| `coords` | `tuple[tuple[float, float], ...]` | The four corner coordinates |
| `width_length` | `tuple[float, float]` | `(width, length)` — the shorter side then the longer side |
| `from_polygon(polygon)` | classmethod | Fit to a polygon |

```python
obb = ann.oriented_bounding_box
if obb is not None:
    width, length = obb.width_length
```
