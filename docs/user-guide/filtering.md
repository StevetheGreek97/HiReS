# Filtering and class handling

Raw model output usually needs cleaning before analysis: low-confidence
detections, debris, clumps and unwanted classes have to go. Every operation
on this page returns a new [`Collection`][hires.Collection] or
[`Album`][hires.Album]; nothing is modified in place.

## Selecting annotations

### By confidence

```python
confident = collection.filter_confidence(0.8)
```

Keeps annotations with a confidence of at least the given value. Annotations
**without** a confidence, i.e. ground truth, are kept.

### By area

```python
scaled = collection.with_scale(dpi=2400, unit="um")
animals = scaled.filter_area(min_area=5e4, max_area=2e6)
```

Removes debris (too small) and clumps (too large). The bounds are inclusive,
and either may be omitted. They are interpreted in the units the collection
measures in: µm² in the example above, px² for a collection with an image
size but no scale, and normalized units otherwise. Set the scale *before*
filtering by area.

### By class

```python
daphnia = collection.keep_classes([0, 1])
no_debris = collection.drop_classes(2)
```

Both accept a single class id or an iterable of ids.

### By an arbitrary condition

[`filter`][hires.Collection.filter] takes any function of an annotation:

```python
round_ones = collection.filter(
    lambda a: a.traits(collection.image_size)["circularity"] > 0.8
)
inside = collection.filter(lambda a: a.within_frame(0.02))
```

### On an album

The same methods exist on [`Album`][hires.Album] and apply to every
collection. For anything else, use [`map`][hires.Album.map]:

```python
clean = album.filter_confidence(0.6).drop_classes(2)
deduplicated = album.map(lambda c: c.nms(0.7))
```

To select whole collections, use [`select`][hires.Album.select] with a list
of names, or slice the album.

## Counting classes

```python
collection.class_counts()   # {0: 61, 1: 26}
album.class_counts()        # summed over all collections
```

## Removing duplicates

[`nms`][hires.Collection.nms] applies polygon non-maximum suppression, the
same procedure the segmentation pipeline uses to merge overlapping tiles
(see [Tiled segmentation](segmentation.md#5-non-maximum-suppression)):

```python
deduplicated = collection.nms(iou_threshold=0.5)
per_class = collection.nms(iou_threshold=0.5, class_aware=True)
```

With `class_aware=True` only annotations of the same class suppress each
other. Every annotation needs a confidence, so NMS cannot be applied to
ground truth.

## Translating class ids

Datasets evolve. A model may have been trained with the classes
`{0: "ballooned", 1: "Daphnia"}` while newer ground truth distinguishes
`{0: "d_pulex", 1: "d_galeata", 2: "ballooned"}`. To compare the two, the
class ids of one side must be translated.

### Simple renumbering

For a fixed one-to-one change, a dict is enough. Ids that are not in the
dict are left unchanged:

```python
remapped = collection.remap_classes({0: 2, 1: 0})
```

### Translating between named schemas

When the schemas are given by name, build a
[`ClassMapping`][hires.ClassMapping] with
[`hires.build_class_mapping`][hires.build_class_mapping]. You describe the
translation with **labels**; HiReS works out the ids:

```python
>>> import hires
>>> mapping = hires.build_class_mapping(
...     old_names={0: "ballooned", 1: "Daphnia"},
...     new_names={0: "d_pulex", 1: "d_galeata", 2: "ballooned"},
...     name_map={"ballooned": "ballooned", "Daphnia": ["d_pulex", "d_galeata"]},
... )
```

`name_map` must have an entry for every old label. A string is a direct
rename. A list declares an **ambiguous** entry: the old class was split, and
which of the candidates applies depends on the data.

Ambiguous entries are resolved when the mapping is applied, with
`resolve={old_label: chosen_new_label}`:

```python
>>> mapping.flatten({"Daphnia": "d_galeata"})
{0: 2, 1: 1}
```

```python
galeata_scans = old_album.select(galeata_names)
remapped = galeata_scans.remap_classes(mapping, resolve={"Daphnia": "d_galeata"})
```

This is the typical use: the same `ClassMapping` is reused for the whole
dataset, and each group of images, say one monoculture per species, is
resolved differently. Forgetting to resolve an ambiguous entry raises a
`KeyError`; choosing a label that was not declared as a candidate raises a
`ValueError`.

The names themselves can be read from a `data.yaml` or from model weights
with [`hires.read_class_names`][hires.read_class_names].

## Balancing a dataset

Training sets are often dominated by one class. 
[`Album.balance`][hires.Album.balance] reduces the imbalance by dropping
collections (images or tiles) that contain **only** majority classes:

```python
balanced = album.balance(majority_classes=[0], max_ratio=3.0, seed=42)
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
