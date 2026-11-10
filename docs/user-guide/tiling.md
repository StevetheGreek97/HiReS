# Tiling training data

A model that is applied to tiles should be trained on tiles. This page
shows how to cut annotated full-resolution images into a training set with
the same geometry the segmentation pipeline uses at inference. The
functions live in [`hires.tiling`](../reference/tiling.md).

You do not need this page to *run* a model: `hires.segment` tiles and
merges internally.

```python
>>> import numpy as np
>>> import cv2
>>> from shapely.geometry import box
>>> import hires
```

## The tile grid

Tiles are placed every `tile_size - overlap` pixels, starting at the
top-left corner, until the image is covered. Each tile is identified by the
pixel coordinates `(x, y)` of its top-left corner in the full image.
[`tile_grid`][hires.tiling.tile_grid] returns these corners for an image of
a given `(width, height)`:

```python
>>> hires.tiling.tile_grid((2500, 900), tile_size=1024, overlap=150)
[(0, 0), (874, 0), (1748, 0)]
>>> hires.tiling.tile_grid((300, 200), tile_size=128, overlap=28)
[(0, 0), (0, 100), (100, 0), (100, 100), (200, 0), (200, 100)]
```

The second grid, three columns by two rows of 128 px tiles placed every
100 px, is used in the rest of this page. Its last column and row extend
past the image; the part outside is filled with zeros, so all tiles have
the same size.

## Tiling images

[`iter_tiles`][hires.tiling.iter_tiles] yields the tiles of an image array:

```python
>>> image = np.full((200, 300, 3), 200, dtype=np.uint8)
>>> for tile in hires.tiling.iter_tiles(image, tile_size=128, overlap=28):
...     print(tile.x, tile.y, tile.pixels.shape)
0 0 (128, 128, 3)
0 100 (128, 128, 3)
100 0 (128, 128, 3)
100 100 (128, 128, 3)
200 0 (128, 128, 3)
200 100 (128, 128, 3)
```

[`tile_image`][hires.tiling.tile_image] does the same for an image file
and writes one PNG per tile, named `{stem}_{x}_{y}.png`. Bit depth and
channels are preserved. The command-line equivalent is `hires chunk`.

```python
>>> _ = cv2.imwrite("scan.png", image)
>>> paths = hires.tiling.tile_image("scan.png", "tiles/images", tile_size=128, overlap=28)
>>> [path.name for path in paths]
['scan_0_0.png', 'scan_0_100.png', 'scan_100_0.png', 'scan_100_100.png', 'scan_200_0.png', 'scan_200_100.png']
```

The position of a tile is recoverable from its file name:

```python
>>> hires.tiling.parse_tile_name("scan_200_100.png")
('scan', 200, 100)
```

## Tiling annotations

[`tile_annotations`][hires.tiling.tile_annotations] is the counterpart for
labels. Take a collection with two objects: a small one near the top-left
corner, and a larger one in the middle of the image.

```python
>>> collection = hires.Collection(
...     [
...         hires.Annotation(0, box(0.05, 0.10, 0.15, 0.25)),   # 30 x 30 px
...         hires.Annotation(1, box(0.37, 0.30, 0.60, 0.55)),   # 69 x 50 px
...     ],
...     name="scan",
...     image_size=(300, 200),
... )
>>> tiles = hires.tiling.tile_annotations(collection, tile_size=128, overlap=28)
>>> for position, tile in tiles.items():
...     print(position, tile)
(0, 0) Collection(name='scan_0_0', n=2, image_size=(128, 128))
(0, 100) Collection(name='scan_0_100', n=1, image_size=(128, 128))
(100, 0) Collection(name='scan_100_0', n=1, image_size=(128, 128))
(100, 100) Collection(name='scan_100_100', n=1, image_size=(128, 128))
(200, 0) Collection(name='scan_200_0', n=0, image_size=(128, 128))
(200, 100) Collection(name='scan_200_100', n=0, image_size=(128, 128))
```

The result maps every tile position to a [`Collection`][hires.Collection]
named `{name}_{x}_{y}`, the same stem as the tile image, with polygons
normalized to the tile. Empty tiles are included.

The collection needs an `image_size`; read labels together with their
image to get one (`hires.read_txt(path, image=...)`).

### Objects on a tile border

An annotation is assigned to **every** tile its bounding box overlaps and
is **not clipped**. The second object lies across the borders of several
tiles. In tile `(0, 0)` it sticks out on the right, so its coordinates
exceed 1; tile `(100, 0)` contains it completely:

```python
>>> [round(v, 2) for v in tiles[(0, 0)][1].bounds]
[0.87, 0.47, 1.41, 0.86]
>>> [round(v, 2) for v in tiles[(100, 0)][0].bounds]
[0.09, 0.47, 0.62, 0.86]
```

For training, drop these partial objects with
[`within_frame`][hires.Annotation.within_frame], which is true for
annotations that lie completely inside their tile:

```python
>>> complete = {
...     position: tile.filter(lambda a: a.within_frame())
...     for position, tile in tiles.items()
... }
>>> for position, tile in complete.items():
...     print(position, len(tile))
(0, 0) 1
(0, 100) 0
(100, 0) 1
(100, 100) 0
(200, 0) 0
(200, 100) 0
```

Each object is now complete in at least one tile. This is the right choice
for two reasons. Ultralytics rejects label files with coordinates outside
`[0, 1]`. And it mirrors inference, where detections touching a tile
border are discarded because the neighbouring tile sees the object in full
(see [Edge filtering](segmentation.md#3-edge-filtering)).

!!! warning "The overlap must exceed the largest object"

    The second object is 69 × 50 px and the overlap only 28 px. It
    survived because it happens to fit inside tile `(100, 0)`. An object
    is guaranteed to be complete in some tile only if the overlap is at
    least as large as the object. With too small an overlap, large objects
    silently vanish from the training set.

## Building a training set

Putting it together for a directory of scans and their label files:

```python
>>> album = hires.read_album("labels/", image_dir="scans/")  # doctest: +SKIP
>>> tiled = []  # doctest: +SKIP
>>> for collection in album:  # doctest: +SKIP
...     hires.tiling.tile_image(collection.image_path, "dataset/images/")
...     tiles = hires.tiling.tile_annotations(collection)
...     tiled.extend(t.filter(lambda a: a.within_frame()) for t in tiles.values())
>>> dataset = hires.Album(tiled, name="dataset")  # doctest: +SKIP
>>> dataset.to_txt("dataset/labels/", confidence=False)  # doctest: +SKIP
```

`confidence=False` writes plain YOLO labels without a trailing confidence
column, as training expects. With the example of this page:

```python
>>> dataset = hires.Album(complete.values(), name="dataset")
>>> dataset
Album(name='dataset', collections=6, annotations=2)
>>> written = dataset.to_txt("tiles/labels", confidence=False)
>>> sorted(path.name for path in written)
['scan_0_0.txt', 'scan_0_100.txt', 'scan_100_0.txt', 'scan_100_100.txt', 'scan_200_0.txt', 'scan_200_100.txt']
```

Scans of sparse samples produce many empty tiles. To thin out tiles that
contain only the dominant class, or nothing, balance the album before
writing it; see [Balancing a dataset](filtering.md#balancing-a-dataset).

!!! warning "Split by scan, not by tile"

    Overlapping tiles of one scan share pixels. Splitting them at random
    into training and validation sets leaks validation data into training
    and inflates the validation metrics. Assign whole scans to one set.

## Merging tiles back

[`merge_tiles`][hires.tiling.merge_tiles] is the inverse coordinate
transform. It takes `{(x, y): annotations}` and returns one collection in
full-image coordinates:

```python
>>> merged = hires.tiling.merge_tiles(
...     complete, tile_size=128, image_size=(300, 200), name="scan"
... )
>>> merged
Collection(name='scan', n=2, image_size=(300, 200))
>>> [tuple(round(v, 2) for v in a.bounds) for a in merged]
[(0.05, 0.1, 0.15, 0.25), (0.37, 0.3, 0.6, 0.55)]
```

Both objects are back at their original positions. An object that is
complete in several overlapping tiles appears once per tile in the merged
result. For predictions, remove such duplicates with
[`Collection.nms`][hires.Collection.nms]; this is what the segmentation
pipeline does.

To merge tile label files from disk, recover the positions from the file
names:

```python
>>> from pathlib import Path
>>> from_disk = {}
>>> for path in sorted(Path("tiles/labels").glob("scan_*.txt")):
...     _, x, y = hires.tiling.parse_tile_name(path)
...     from_disk[(x, y)] = hires.read_txt(path)
>>> hires.tiling.merge_tiles(from_disk, tile_size=128, image_size=(300, 200))
Collection(name=None, n=2, image_size=(300, 200))
```

## See also

- Reference: [`hires.tiling`](../reference/tiling.md)
- [Tiled segmentation](segmentation.md) for the inference-time algorithm
