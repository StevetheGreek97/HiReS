# Tiling training data

A model that is applied to tiles should be trained on tiles. This page
shows how to cut annotated full-resolution images into a training set with
the same geometry the segmentation pipeline uses at inference. The
functions live in [`hires.tiling`](../reference/tiling.md).

You do not need this page to *run* a model: `hires.segment` tiles and
merges internally.

## The tile grid

Tiles are placed every `tile_size - overlap` pixels, starting at the
top-left corner, until the image is covered:

```python
>>> hires.tiling.tile_grid((2500, 900), tile_size=1024, overlap=150)
[(0, 0), (874, 0), (1748, 0)]
```

Each tile is identified by the pixel coordinates `(x, y)` of its top-left
corner in the full image. The last tile along each axis may extend past the
image; the part outside is filled with zeros, so all tiles have the same
size.

## Tiling images

```python
paths = hires.tiling.tile_image(
    "scans/scan_01.tif", "dataset/images/", tile_size=1024, overlap=150
)
```

writes one PNG per tile, named `{stem}_{x}_{y}.png`, and returns the paths.
Bit depth and channels are preserved. The command-line equivalent is
`hires chunk`.

The position is recoverable from the file name:

```python
>>> hires.tiling.parse_tile_name("scan_01_874_1748.png")
('scan_01', 874, 1748)
```

To process tiles in memory instead, iterate over an array:

```python
image = hires.io.read_image("scans/scan_01.tif")
for tile in hires.tiling.iter_tiles(image, tile_size=1024, overlap=150):
    tile.x, tile.y, tile.pixels.shape
```

## Tiling annotations

[`tile_annotations`][hires.tiling.tile_annotations] is the counterpart for
labels. It needs the image size, so read the labels together with their
image:

```python
collection = hires.read_txt("labels/scan_01.txt", image="scans/scan_01.tif")

tiles = hires.tiling.tile_annotations(collection, tile_size=1024, overlap=150)
```

The result maps every tile position to a
[`Collection`][hires.Collection] named `{name}_{x}_{y}`, the same stem as
the tile image, with polygons normalized to the tile. Empty tiles are
included.

### Objects on a tile border

An annotation is assigned to **every** tile its bounding box overlaps and
is **not clipped**. An object that crosses a tile border therefore appears
in two tiles, each time with some coordinates outside `[0, 1]`.

For training, drop these partial objects:

```python
complete = {
    position: tile.filter(lambda a: a.within_frame())
    for position, tile in tiles.items()
}
```

This is the right choice for two reasons. Ultralytics rejects label files
with coordinates outside `[0, 1]`. And it mirrors inference, where
detections touching a tile border are discarded because the neighbouring
tile sees the object in full (see
[Edge filtering](segmentation.md#3-edge-filtering)). As at inference, this
only works if the overlap is larger than the largest object; otherwise
large objects are complete in no tile and vanish from the training set.

## Building a training set

Putting it together for a directory of scans:

```python
from pathlib import Path

album = hires.read_album("labels/", image_dir="scans/")
tiled = []

for collection in album:
    hires.tiling.tile_image(collection.image_path, "dataset/images/")
    tiles = hires.tiling.tile_annotations(collection)
    tiled.extend(t.filter(lambda a: a.within_frame()) for t in tiles.values())

dataset = hires.Album(tiled, name="dataset")
dataset.to_txt("dataset/labels/", confidence=False)
```

`confidence=False` writes plain YOLO labels without a trailing confidence
column, as training expects.

Scans of sparse samples produce many empty tiles. To thin out tiles that
contain only the dominant class, or nothing, balance the album before
writing it (see [Balancing a dataset](filtering.md#balancing-a-dataset)),
replacing the last line above with:

```python
balanced = dataset.balance(majority_classes=[0], max_ratio=3.0, seed=42)
balanced.to_txt("dataset/labels/", confidence=False)
```

Tile images without a label file can then be deleted, or kept as background
examples.

!!! warning "Split by scan, not by tile"

    Overlapping tiles of one scan share pixels. Splitting them at random
    into training and validation sets leaks validation data into training
    and inflates the validation metrics. Assign whole scans to one set.

## Merging tiles back

[`merge_tiles`][hires.tiling.merge_tiles] is the inverse coordinate
transform. It takes `{(x, y): annotations}` and returns one collection in
full-image coordinates:

```python
merged = hires.tiling.merge_tiles(
    tiles, tile_size=1024, image_size=collection.image_size, name="scan_01"
)
```

Objects seen in several overlapping tiles appear several times in the
result. For predictions, remove the duplicates with
[`Collection.nms`][hires.Collection.nms]; this is what the segmentation
pipeline does.

To merge tile label files from disk, recover the positions from their
names:

```python
tiles = {}
for path in Path("tile_labels/").glob("scan_01_*.txt"):
    _, x, y = hires.tiling.parse_tile_name(path)
    tiles[(x, y)] = hires.read_txt(path)
```

## See also

- Reference: [`hires.tiling`](../reference/tiling.md)
- [Tiled segmentation](segmentation.md) for the inference-time algorithm
