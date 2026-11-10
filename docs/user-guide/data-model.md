# The data model

HiReS represents segmentation results with three classes that nest inside
each other:

| Class | Represents | Behaves like |
|-------|------------|--------------|
| [`Annotation`][hires.Annotation] | One object: a class id, a polygon and a confidence | A frozen dataclass |
| [`Collection`][hires.Collection] | All annotations of one image, plus the image metadata | A read-only sequence of `Annotation` |
| [`Album`][hires.Album] | A dataset: a named set of collections | A read-only sequence of `Collection`, also indexable by name |

All three are **immutable**. A method that "changes" something returns a new
object and leaves the original untouched, so results can be chained,
compared and reused without defensive copies.

## Annotation

An annotation is one segmented object.

```python
>>> from shapely.geometry import box
>>> import hires
>>> ann = hires.Annotation(0, box(0.1, 0.1, 0.3, 0.2), confidence=0.9)
>>> ann
Annotation(class_id=0, confidence=0.900, vertices=4)
```

`class_id`
:   Integer class label, as used by the model.

`polygon`
:   A [`shapely.geometry.Polygon`](https://shapely.readthedocs.io/en/stable/reference/shapely.Polygon.html).
    The whole Shapely API is available on it.

`confidence`
:   Detection score in `[0, 1]`, or `None` for ground-truth annotations
    that carry no score.

### Normalized coordinates

Polygon coordinates are **normalized**: `x` is a fraction of the image width
and `y` a fraction of the image height, with the origin in the top-left
corner and `y` pointing down. This is exactly how YOLO segmentation label
files store them, so reading and writing involves no conversion.

```python
>>> ann.bounds
(0.1, 0.1, 0.3, 0.2)
```

An annotation does not know the size of its image. To obtain pixel
coordinates, pass the size explicitly:

```python
>>> ann.to_pixels((1000, 1000)).bounds
(100.0, 100.0, 300.0, 200.0)
```

!!! warning "Measure in pixels, not in normalized coordinates"
    Normalized coordinates stretch a non-square image into a unit square.
    Lengths, and the ratios `convexity` and `circularity` that depend on
    them, are therefore only meaningful after scaling to pixels. A
    [`Collection`][hires.Collection] that knows its `image_size` does this
    for you.

### Immutability

```python
>>> ann.class_id = 1
Traceback (most recent call last):
    ...
dataclasses.FrozenInstanceError: cannot assign to field 'class_id'
>>> ann.replace(class_id=1)
Annotation(class_id=1, confidence=0.900, vertices=4)
```

## Collection

A collection holds the annotations of one image together with what is known
about that image.

```python
>>> collection = hires.Collection(
...     [ann, hires.Annotation(1, box(0.5, 0.5, 0.7, 0.6), confidence=0.6)],
...     name="scan",
...     image_size=(2000, 1000),
... )
>>> collection
Collection(name='scan', n=2, image_size=(2000, 1000))
```

In practice collections come from [`hires.segment`][hires.segment] or
[`hires.read_txt`][hires.read_txt] rather than from the constructor.

### Metadata

| Attribute | Meaning | Needed for |
|-----------|---------|------------|
| `name` | Identifier, usually the image file stem | Pairing ground truth with predictions; file names on export |
| `image_size` | `(width, height)` in pixels | Measuring in pixels |
| `image_path` | Location of the image | Crops, comparison figures |
| `dpi`, `unit` | Scan resolution and physical unit | Measuring in physical units |

All of them are optional. HiReS degrades gracefully: without an
`image_size`, traits are measured on the normalized polygons and a
[`HiresWarning`][hires.HiresWarning] tells you so.

Metadata is attached with `with_*` methods, each returning a new collection:

```python
>>> collection.with_scale(dpi=2400, unit="um")
Collection(name='scan', n=2, image_size=(2000, 1000), dpi=2400, unit='um')
```

[`with_image`][hires.Collection.with_image] binds an image file and reads
its size from disk; [`replace`][hires.Collection.replace] sets any attribute
directly.

### A sequence of annotations

A collection supports `len`, iteration, indexing, slicing, `in` and `+`:

```python
>>> len(collection)
2
>>> collection[0] is ann
True
>>> [a.class_id for a in collection]
[0, 1]
>>> collection[:1]
Collection(name='scan', n=1, image_size=(2000, 1000))
```

An integer index returns an `Annotation`; a slice returns a `Collection`
with the same metadata. `collection + annotations` appends any iterable of
annotations and keeps the metadata of the left operand.

### Transformations return new collections

```python
>>> confident = collection.filter_confidence(0.8)
>>> confident
Collection(name='scan', n=1, image_size=(2000, 1000))
>>> len(collection)
2
```

Because every method returns a collection, operations chain naturally:

```python
result = (
    collection
    .filter_confidence(0.6)
    .keep_classes([0, 1])
    .nms(0.7)
    .with_scale(dpi=2400, unit="um")
)
```

The available transformations are covered in
[Filtering and class handling](filtering.md).

## Album

An album is an ordered set of collections with unique names, typically one
per image of a dataset.

```python
>>> album = hires.Album([collection, hires.Collection(name="empty")], name="demo")
>>> album
Album(name='demo', collections=2, annotations=2)
>>> album.names
['scan', 'empty']
```

Collections are looked up by position or by name:

```python
>>> album["scan"] is collection
True
>>> album[0] is collection
True
>>> "scan" in album
True
>>> album.get("missing") is None
True
```

A slice returns an album. [`select`][hires.Album.select] picks collections by
name, and [`map`][hires.Album.map] applies any function to every collection.

### Dataset-wide operations

Most `Collection` methods have an `Album` counterpart that applies the
operation to every collection:

```python
>>> album.class_counts()
{0: 1, 1: 1}
>>> album.keep_classes(0)
Album(name='demo', collections=2, annotations=1)
```

Note that filtering never removes a collection, even if it becomes empty. An
image without objects is still an image, and it matters for evaluation.

### Unique names

Names identify collections, so they must be unique and non-empty. If several
collections share a name, the later ones are renamed `name__1`, `name__2`,
… and a [`HiresWarning`][hires.HiresWarning] is emitted.

## Equality

Two collections are equal when their annotations and all metadata are equal;
two albums when their names and collections are. Collections and albums are
not hashable.

## See also

- Reference: [`hires.Annotation`](../reference/annotation.md),
  [`hires.Collection`](../reference/collection.md),
  [`hires.Album`](../reference/album.md)
- [Reading and writing](io.md)
