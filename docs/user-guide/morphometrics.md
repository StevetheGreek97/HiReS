# Morphometric traits

HiReS derives nine geometric descriptors from the polygon outline of every
object. They correspond to the size and shape traits commonly used in
ecological image analysis. Their names are listed, in order, in
[`hires.TRAIT_NAMES`][hires.TRAIT_NAMES].

| Trait | Symbol | Definition | Range | Unit | Describes |
|-------|--------|------------|-------|------|-----------|
| `area` | $A$ | Enclosed area | $> 0$ | length² | Two-dimensional size |
| `perimeter` | $P$ | Boundary length | $> 0$ | length | Size and outline complexity |
| `solidity` | $S$ | $A / A_\text{hull}$ | $(0, 1]$ | — | Depth of concavities |
| `convexity` | $K$ | $P_\text{hull} / P$ | $(0, 1]$ | — | Smoothness of the outline |
| `circularity` | $C$ | $4 \pi A / P^2$ | $(0, 1]$ | — | Compactness |
| `bbox_width` | | Extent along the image $x$ axis | $> 0$ | length | |
| `bbox_height` | | Extent along the image $y$ axis | $> 0$ | length | |
| `obb_width` | $W$ | Short side of the oriented bounding box | $> 0$ | length | Body width |
| `obb_length` | $L$ | Long side of the oriented bounding box | $> 0$ | length | Body length |

## Computing traits

For a whole collection, as a table:

```python
frame = collection.to_frame()
frame[["class_id", "area", "obb_length", "circularity"]]
```

For a single annotation, as a dict:

```python
traits = collection[0].traits(collection.image_size)
traits["obb_length"]
```

For any Shapely polygon, with [`hires.measure`][hires.measure]:

```python
>>> from shapely.geometry import box
>>> import hires
>>> traits = hires.measure(box(0, 0, 4, 2))
>>> traits["area"], traits["perimeter"], traits["obb_length"], traits["obb_width"]
(8.0, 12.0, 4.0, 2.0)
```

## Definitions

### Area

The surface enclosed by the outline, computed from the vertices
$(x_i, y_i)$, $i = 1 \dots n$, with the shoelace formula:

$$
A = \frac{1}{2} \left| \sum_{i=1}^{n} \left( x_i \, y_{i+1} - x_{i+1} \, y_i \right) \right|,
\qquad (x_{n+1}, y_{n+1}) = (x_1, y_1).
$$

Area is a direct estimate of body size that uses the full geometry of the
object.

### Perimeter

The length of the boundary, the sum of the Euclidean distances between
consecutive vertices:

$$
P = \sum_{i=1}^{n} \sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2}.
$$

Perimeter reflects both size and boundary complexity. It depends on the
resolution of the outline: a more finely sampled, more jagged polygon has a
longer perimeter than a smooth one around the same object.

### Body length and width

The **oriented bounding box** (OBB) is the rectangle of minimum area that
encloses the polygon, at whatever angle achieves that minimum. With side
lengths $l_1$ and $l_2$,

$$
L = \max(l_1, l_2), \qquad W = \min(l_1, l_2).
$$

`obb_length` and `obb_width` are independent of how the organism is
oriented on the scan, which makes them the appropriate measures of body
length and width.

!!! note "Oriented versus axis-aligned"
    `bbox_width` and `bbox_height` are the extents along the image axes.
    They depend on the orientation of the organism: for an elongated animal
    lying diagonally, both overestimate its width and underestimate its
    length. Use them for cropping, not for morphometry.

### Circularity

$$
C = \frac{4 \pi A}{P^2}
$$

Equal to 1 for a circle and smaller for every other shape. It decreases as
an object becomes more elongated or its outline more irregular.

### Convexity

$$
K = \frac{P_\text{hull}}{P}
$$

where $P_\text{hull}$ is the perimeter of the convex hull. Values near 1
indicate a smooth, convex outline; lower values indicate high-frequency
irregularities such as spines or a jagged boundary.

### Solidity

$$
S = \frac{A}{A_\text{hull}}
$$

where $A_\text{hull}$ is the area of the convex hull, the smallest convex
polygon enclosing the object. Low values indicate deep concavities or
appendages.

Convexity and solidity are complementary: convexity responds to roughness
of the boundary, solidity to missing area.

## Units

The traits are computed in whatever coordinate system the polygon is in.
Which one that is depends on the collection:

| The collection has | Lengths in | Areas in |
|--------------------|------------|----------|
| no `image_size` | normalized units | normalized units² |
| `image_size` | pixels | pixels² |
| `image_size`, `dpi` and `unit` | `unit` | `unit`² |

The ratios `solidity`, `convexity` and `circularity` are dimensionless and
do not change with scale.

!!! warning "Always provide the image size"
    In normalized coordinates a non-square image is squeezed into a square.
    `area` and `solidity` survive this (up to a constant factor), but every
    length, and with it `perimeter`, `convexity`, `circularity` and the
    bounding-box dimensions, is distorted. HiReS emits a
    [`HiresWarning`][hires.HiresWarning] when it has to measure this way.
    Load labels with `image=` or `image_dir=` to avoid it.

### Physical units

A scanner resolution in dots per inch fixes the physical size of a pixel.
For a unit with $u$ units per inch,

$$
s = \frac{u}{\text{dpi}}
$$

is the edge length of one pixel. Lengths are multiplied by $s$, areas by
$s^2$.

```python
>>> collection = hires.Collection(
...     [hires.Annotation(0, box(0, 0, 0.5, 0.5))], image_size=(2400, 2400)
... )
>>> scaled = collection.with_scale(dpi=2400, unit="um")
>>> round(scaled.pixel_size, 3)
10.583
```

The supported units are the keys of [`hires.UNITS`][hires.UNITS]:

| `unit` | Meaning | Units per inch |
|--------|---------|----------------|
| `"nm"` | nanometre | 25 400 000 |
| `"um"` | micrometre | 25 400 |
| `"mm"` | millimetre | 25.4 |
| `"cm"` | centimetre | 2.54 |
| `"m"` | metre | 0.0254 |
| `"inch"` | inch | 1 |

`dpi` and `unit` must be given together, and the collection must know its
`image_size`. Calling `with_scale(None, None)` removes the scale again.

!!! note "Resolution of your images"
    The `dpi` must be the resolution at which the image was actually
    acquired. For a microscope, convert the calibration to dots per inch:
    a pixel size of $p$ µm corresponds to $25\,400 / p$ dpi.

## Edge cases

- A polygon with zero perimeter or zero hull area yields `0.0` for the
  affected ratio rather than raising.
- If no oriented bounding box can be constructed, `obb_width` and
  `obb_length` are `nan`.
- For polygons with holes, `area` excludes the holes and `perimeter`
  includes the interior boundaries. Polygons read from label files never
  have holes.

## Validation

HiReS was validated against manually annotated outlines of *Daphnia pulex*,
*Daphnia galeata* and *Simocephalus vetulus* across nine full-resolution
scanner images. The main findings were:

- Automated and manual trait distributions preserved the same broad
  structure and relative ordering.
- A consistent positive bias of 5–19 % was observed: a multiplicative
  scaling offset, not a distortion.
- After centring by the median residual, agreement was near-perfect
  ($r$ = 0.969–0.999).
- Sample-level medians were strongly correlated ($r$ = 0.94–0.96) despite
  the individual-level bias.
- At low sampling depths ($n$ = 10), automated medians outperformed manual
  subsamples in 43–62 % of cases.

The positive bias likely reflects illumination-induced halos around the
organisms that are partially included in the predicted mask. To quantify the
agreement on your own data, see [Evaluation](evaluation.md#trait-agreement).

## See also

- Reference: [Measurement](../reference/measurement.md)
