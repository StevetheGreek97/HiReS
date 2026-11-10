# Filtering and class handling

Raw model output usually needs cleaning before analysis: low-confidence
detections, debris, clumps and unwanted classes have to go. Every operation
on this page returns a new [`Collection`][hires.Collection] or
[`Album`][hires.Album]; nothing is modified in place.

The examples use a collection of five squares on a 1000 × 1000 px image:
three of class 0, one of class 1, and one tiny object of class 2.

```python
>>> from shapely.geometry import box
>>> import hires
>>> def square(x, y, side, class_id, confidence):
...     return hires.Annotation(class_id, box(x, y, x + side, y + side), confidence)
>>> collection = hires.Collection(
...     [
...         square(0.10, 0.10, 0.20, class_id=0, confidence=0.95),
...         square(0.50, 0.10, 0.20, class_id=0, confidence=0.80),
...         square(0.10, 0.50, 0.30, class_id=0, confidence=0.45),
...         square(0.60, 0.60, 0.10, class_id=1, confidence=0.70),
...         square(0.90, 0.90, 0.02, class_id=2, confidence=0.60),
...     ],
...     name="scan",
...     image_size=(1000, 1000),
... )
>>> collection
Collection(name='scan', n=5, image_size=(1000, 1000))
>>> collection.class_counts()
{0: 3, 1: 1, 2: 1}
```

## Selecting annotations

### By confidence

`filter_confidence` keeps annotations with at least the given confidence:

```python
>>> confident = collection.filter_confidence(0.6)
>>> confident
Collection(name='scan', n=4, image_size=(1000, 1000))
>>> [a.confidence for a in confident]
[0.95, 0.8, 0.7, 0.6]
```

Annotations **without** a confidence, that is ground truth, are always
kept.

### By area

`filter_area` removes debris (too small) and clumps (too large). The bounds
are inclusive, and either may be omitted:

```python
>>> collection.filter_area(min_area=1_000)
Collection(name='scan', n=4, image_size=(1000, 1000))
>>> collection.filter_area(min_area=1_000, max_area=50_000)
Collection(name='scan', n=3, image_size=(1000, 1000))
```

The bounds are in the units the collection measures in. Here that is px²;
after [`with_scale`][hires.Collection.with_scale] it is the physical unit
squared, so set the scale *before* filtering by area:

```python
>>> scaled = collection.with_scale(dpi=2400, unit="um")
>>> scaled.filter_area(min_area=1e6)     # at least 1 mm²
Collection(name='scan', n=4, image_size=(1000, 1000), dpi=2400, unit='um')
```

### By class

Both methods accept one class id or several:

```python
>>> collection.keep_classes([0, 1]).class_counts()
{0: 3, 1: 1}
>>> collection.drop_classes(2).class_counts()
{0: 3, 1: 1}
```

### By any condition

[`filter`][hires.Collection.filter] takes a function that receives an
annotation and returns true to keep it:

```python
>>> collection.filter(lambda a: a.polygon.centroid.x < 0.5)
Collection(name='scan', n=2, image_size=(1000, 1000))
>>> collection.filter(lambda a: a.within_frame(0.09))
Collection(name='scan', n=4, image_size=(1000, 1000))
```

The second example keeps objects that are at least 9 % of the image away
from every border; see
[`Annotation.within_frame`][hires.Annotation.within_frame].

### Chaining

Because every method returns a collection, a cleaning recipe reads top to
bottom:

```python
>>> clean = (
...     collection
...     .filter_confidence(0.6)
...     .drop_classes(2)
...     .filter_area(min_area=1_000)
... )
>>> clean
Collection(name='scan', n=3, image_size=(1000, 1000))
>>> len(collection)   # the original is untouched
5
```

### On an album

The same methods exist on [`Album`][hires.Album] and apply to every
collection:

```python
>>> album = hires.Album([collection, collection.replace(name="scan_2")], name="demo")
>>> album.filter_confidence(0.6).drop_classes(2)
Album(name='demo', collections=2, annotations=6)
```

For anything else, [`map`][hires.Album.map] applies a function to every
collection, and [`select`][hires.Album.select] picks collections by name:

```python
>>> album.map(lambda c: c.filter(lambda a: a.within_frame(0.09)))
Album(name='demo', collections=2, annotations=8)
>>> album.select(["scan_2"])
Album(name='demo', collections=1, annotations=5)
```

Filtering never removes a collection from an album, even an empty one: an
image without objects is still an image, and it matters for evaluation.

## Counting classes

```python
>>> collection.class_counts()
{0: 3, 1: 1, 2: 1}
>>> album.class_counts()
{0: 6, 1: 2, 2: 2}
```

## Removing duplicates

[`nms`][hires.Collection.nms] applies polygon non-maximum suppression: of
two annotations that overlap with an IoU above the threshold, the less
confident one is removed. It is the procedure the segmentation pipeline
uses to merge overlapping tiles (see
[Tiled segmentation](segmentation.md#5-non-maximum-suppression)).

```python
>>> doubled = collection + [square(0.11, 0.10, 0.20, class_id=1, confidence=0.50)]
>>> len(doubled)
6
>>> round(doubled[0].iou(doubled[5]), 2)
0.9
>>> len(doubled.nms(iou_threshold=0.5))
5
```

The added square nearly coincides with the first one and is suppressed.
With `class_aware=True`, only annotations of the same class suppress each
other, so here both survive:

```python
>>> len(doubled.nms(iou_threshold=0.5, class_aware=True))
6
```

Every annotation needs a confidence, so NMS cannot be applied to ground
truth.

## Translating class ids

Datasets evolve. A model may have been trained with the classes
`{0: "ballooned", 1: "Daphnia"}` while newer ground truth distinguishes
`{0: "d_pulex", 1: "d_galeata", 2: "ballooned"}`. To compare the two, the
class ids of one side must be translated.

### Simple renumbering

For a fixed one-to-one change, a dict is enough. Ids that are not in the
dict are left unchanged:

```python
>>> collection.class_counts()
{0: 3, 1: 1, 2: 1}
>>> collection.remap_classes({0: 5, 1: 0}).class_counts()
{0: 1, 2: 1, 5: 3}
```

### Translating between named schemas

When the schemas are given by name, build a
[`ClassMapping`][hires.ClassMapping] with
[`hires.build_class_mapping`][hires.build_class_mapping]. You describe the
translation with **labels**; HiReS works out the ids:

```python
>>> mapping = hires.build_class_mapping(
...     old_names={0: "ballooned", 1: "Daphnia"},
...     new_names={0: "d_pulex", 1: "d_galeata", 2: "ballooned"},
...     name_map={"ballooned": "ballooned", "Daphnia": ["d_pulex", "d_galeata"]},
... )
```

`name_map` must have an entry for every old label. A string is a direct
rename. A list declares an **ambiguous** entry: the old class was split,
and which of the candidates applies depends on the data.

Ambiguous entries are resolved when the mapping is applied, with
`resolve={old_label: chosen_new_label}`:

```python
>>> mapping.flatten({"Daphnia": "d_galeata"})
{0: 2, 1: 1}
>>> old = hires.Collection([square(0.1, 0.1, 0.2, class_id=1, confidence=0.9)])
>>> old.remap_classes(mapping, resolve={"Daphnia": "d_galeata"}).class_counts()
{1: 1}
>>> old.remap_classes(mapping, resolve={"Daphnia": "d_pulex"}).class_counts()
{0: 1}
```

This is the typical use: the same `ClassMapping` serves the whole dataset,
and each group of images, say one monoculture per species, is resolved
differently. Forgetting to resolve an ambiguous entry is an error:

```python
>>> old.remap_classes(mapping)
Traceback (most recent call last):
    ...
KeyError: "'Daphnia' is ambiguous ['d_pulex', 'd_galeata']; choose one with resolve."
```

The names themselves can be read from a `data.yaml` or from model weights
with [`hires.read_class_names`][hires.read_class_names].

## Balancing a dataset

Training sets are often dominated by one class.
[`Album.balance`][hires.Album.balance] reduces the imbalance by dropping
collections (images or tiles) that contain **only** majority classes.

Take six tiles: five contain a single object of class 0, one contains an
object of class 1.

```python
>>> tiles = [
...     hires.Collection([square(0.1, 0.1, 0.2, 0, None)], name=f"tile_{i}")
...     for i in range(5)
... ]
>>> tiles.append(hires.Collection([square(0.1, 0.1, 0.2, 1, None)], name="tile_5"))
>>> dataset = hires.Album(tiles, name="tiles")
>>> dataset.class_counts()
{0: 5, 1: 1}
>>> balanced = dataset.balance(majority_classes=[0], max_ratio=2.0, seed=0)
>>> balanced
Album(name='tiles', collections=3, annotations=3)
>>> balanced.class_counts()
{0: 2, 1: 1}
```

Majority-only collections, which include empty ones, are removed in random
order until

$$
n_\text{majority} \le r_\max \cdot n_\text{minority},
$$

counted in annotations. Collections containing at least one minority
annotation are always kept, so the target ratio may not be reachable. The
result is shuffled; fix `seed` for a reproducible selection.

## See also

- Reference: [`hires.Collection`](../reference/collection.md),
  [`hires.Album`](../reference/album.md),
  [Class handling](../reference/classes.md)
