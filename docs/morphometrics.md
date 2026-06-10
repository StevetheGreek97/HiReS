# Morphometric Descriptors

After segmentation and duplicate removal, HiReS computes a suite of geometric descriptors for each detected object. These descriptors correspond to commonly used size- and shape-based morphometric traits in ecological image analysis.

All measurements are initially reported in **pixel units** relative to the input image resolution. To obtain biologically meaningful measurements (e.g. µm, mm), apply a calibration factor derived from the imaging setup — see [Physical unit conversion](#physical-unit-conversion) below.

---

## Area

Area quantifies the two-dimensional size of an individual by measuring the surface enclosed by its polygon outline. It is computed using the **Shoelace Formula**:

$$A = \frac{1}{2} \left| \sum_{i=1}^{n-1} (x_i y_{i+1} - x_{i+1} y_i) \right|$$

where $(x_i, y_i)$ are the Cartesian coordinates of the $i$-th vertex and $n$ is the total number of vertices. Area provides a direct estimate of body size derived from the full object geometry.

```python
annotation.area  # float, in px² (or calibrated unit² if dpi/unit set)
```

---

## Perimeter

Perimeter measures the total length of the object boundary by summing the Euclidean distances between consecutive vertices along the polygon outline:

$$P = \sum_{i=1}^{n} \sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2}$$

where $(x_i, y_i)$ and $(x_{i+1}, y_{i+1})$ are consecutive polygon vertices (the last vertex connects back to the first). Perimeter captures both object size and boundary complexity.

```python
annotation.perimeter  # float, in px (or calibrated unit if dpi/unit set)
```

---

## Body dimensions (OBB)

To obtain orientation-independent body length and width, HiReS fits a **Minimum Area Bounding Box** (oriented bounding box, OBB) to each polygon. This identifies the rectangle orientation that minimizes enclosed area. The major axis of this box corresponds to the maximum body length; the minor axis gives the maximum width.

$$L = \max(l_1, l_2), \quad W = \min(l_1, l_2)$$

where $l_1$ and $l_2$ are the lengths of the two orthogonal sides of the bounding box.

```python
obb = annotation.oriented_bounding_box
width, length = obb.width_length  # (W, L) in px
```

!!! note "OBB vs axis-aligned bounding box"
    The OBB is rotated to align with the object's principal axis, making it robust to organism orientation. An axis-aligned bounding box (AABB) aligned with the image axes overestimates width and height for non-horizontal organisms.

---

## Circularity

Circularity describes the compactness of an object relative to a perfect circle. A value of 1.0 represents a perfect circle; values decrease as objects become more elongated or geometrically complex.

$$C = \frac{4\pi A}{P^2}$$

where $A$ is the polygon area and $P$ is its perimeter. Circularity is unitless and scale-invariant.

```python
annotation.circularity  # float in (0, 1]
```

---

## Convexity

Convexity describes the smoothness of the object's boundary. It is the ratio between the perimeter of the convex hull and the actual polygon perimeter:

$$K = \frac{P_{\text{convex}}}{P}$$

where $P$ is the polygon perimeter and $P_{\text{convex}}$ is the perimeter of its convex hull. Values close to 1 indicate smooth, convex outlines; lower values reflect high-frequency surface irregularities such as jagged or indented boundaries.

```python
annotation.convexity  # float in (0, 1]
```

---

## Solidity

Solidity measures the overall density of an object's shape by comparing its area to the area of its convex hull:

$$S = \frac{A}{A_{\text{hull}}}$$

where $A$ is the polygon area and $A_{\text{hull}}$ is the area of its convex hull (the smallest convex polygon enclosing the object). Low solidity values indicate the presence of deep concavities or structural indentations.

```python
annotation.solidity  # float in (0, 1]
```

---

## Summary table

| Descriptor | Symbol | Formula | Range | Measures |
|-----------|--------|---------|-------|---------|
| Area | $A$ | Shoelace | > 0 | 2D size |
| Perimeter | $P$ | Cumulative Euclidean | > 0 | Boundary length |
| OBB length | $L$ | max OBB side | > 0 | Body length (orientation-independent) |
| OBB width | $W$ | min OBB side | > 0 | Body width (orientation-independent) |
| Circularity | $C$ | $4\pi A / P^2$ | (0, 1] | Compactness vs. circle |
| Convexity | $K$ | $P_{\text{hull}} / P$ | (0, 1] | Boundary smoothness |
| Solidity | $S$ | $A / A_{\text{hull}}$ | (0, 1] | Fill of convex hull |

---

## Physical unit conversion

All descriptors are computed on polygon coordinates stored in normalized image space (0–1). When `dpi` and `unit` are set on an `Annotation` (or applied via `Collection.set_scale()`), the library converts measurements to physical units using:

$$\text{scale} = \frac{\text{unit\_factor}}{\text{dpi}}$$

| Unit | Factor (pixels per inch) |
|------|--------------------------|
| `nm` | 25,400,000 |
| `um` | 25,400 |
| `mm` | 25.4 |
| `cm` | 2.54 |
| `m` | 0.0254 |
| `inch` | 1.0 |

Length measurements are multiplied by `scale`; area measurements are multiplied by `scale²`.

```python
from hires.models.collection import Collection

col = Collection.read_txt("results/image.txt")
col.set_scale(dpi=1200, unit="um")  # 1200 dpi flatbed scanner, report in µm

for ann in col:
    print(f"Area: {ann.area:.1f} µm²  |  Length: {ann.oriented_bounding_box.width_length[1] * ann.scale:.1f} µm")
```

---

## Accessing descriptors

All descriptors are available as properties on `Annotation` objects and are exported automatically to the `_shapes.csv` output file:

```python
ann = collection[0]

print(ann.area)        # px² (or calibrated)
print(ann.perimeter)   # px
print(ann.circularity) # dimensionless
print(ann.convexity)   # dimensionless
print(ann.solidity)    # dimensionless

obb = ann.oriented_bounding_box
if obb:
    w, l = obb.width_length
    print(f"L={l:.2f} px, W={w:.2f} px")
```

Use `ann.to_dict()` to get all descriptors as a dictionary, or `collection.to_df()` for a pandas DataFrame across all annotations.

---

## Validation

HiReS was validated against manually annotated outlines of *Daphnia pulex*, *Daphnia galeata*, and *Simocephalus vetulus* across 9 full-resolution scanner images. Key findings:

- Automated and manual trait distributions preserved the same broad structure and relative ordering
- A consistent positive bias of 5–19% was observed (multiplicative scaling offset, not distortion)
- After centering by the median residual, agreement was near-perfect (r = 0.969–0.999)
- Sample-level medians were strongly correlated (r = 0.94–0.96) despite individual-level bias
- At low sampling depths (n = 10), automated medians outperformed manual subsamples in 43–62% of cases

The positive bias likely reflects illumination-induced halos around organisms being partially included in the predicted segmentation mask, causing slight overestimation of true boundaries.
