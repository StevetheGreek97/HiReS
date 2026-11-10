# Reading and writing

HiReS reads and writes **YOLO segmentation label files** and exports
measurements as tables. The examples on this page create their own small
files, so they can be run as they are.

```python
>>> from pathlib import Path
>>> import hires
```

## The label format

A label file is a plain-text file with one object per line:

```text
class_id x1 y1 x2 y2 ... xn yn [confidence]
```

`class_id`
:   Integer class label.

`x1 y1 ... xn yn`
:   The polygon vertices in normalized coordinates: `x` as a fraction of
    the image width, `y` as a fraction of the image height. At least three
    points are required. The polygon is closed implicitly.

`confidence`
:   Optional detection score in `[0, 1]`. Every vertex contributes two
    numbers, so a trailing value that makes the count odd is the
    confidence.

This is the format written by Ultralytics with
`save_txt=True, save_conf=True`, and without the confidence, the format
expected for training. By convention the label file of `scan.tif` is called
`scan.txt`.

Here is a file with a prediction (confidence 0.87) and a ground-truth
object (no confidence):

```python
>>> Path("labels").mkdir()
>>> _ = Path("labels/scan.txt").write_text(
...     "0 0.1 0.1 0.3 0.1 0.3 0.2 0.1 0.2 0.87\n"
...     "1 0.5 0.5 0.7 0.5 0.6 0.8\n"
... )
```

## Reading one file

[`hires.read_txt`][hires.read_txt] returns a
[`Collection`][hires.Collection] named after the file:

```python
>>> collection = hires.read_txt("labels/scan.txt")
>>> collection
Collection(name='scan', n=2)
>>> collection[0]
Annotation(class_id=0, confidence=0.870, vertices=4)
>>> collection[1]
Annotation(class_id=1, confidence=None, vertices=3)
```

### Telling HiReS the image size

Label files do not record how large the image is, and without that
measurements cannot be expressed in pixels. There are two ways to supply
it. If the image is at hand, pass it; only the file header is read, so this
is fast even for very large scans:

```python
>>> import numpy as np
>>> import cv2
>>> Path("images").mkdir()
>>> _ = cv2.imwrite("images/scan.png", np.zeros((1000, 2000, 3), dtype=np.uint8))
>>> collection = hires.read_txt("labels/scan.txt", image="images/scan.png")
>>> collection
Collection(name='scan', n=2, image_size=(2000, 1000))
>>> collection.image_path.name
'scan.png'
```

Otherwise give the size directly, as `(width, height)`:

```python
>>> hires.read_txt("labels/scan.txt", image_size=(2000, 1000))
Collection(name='scan', n=2, image_size=(2000, 1000))
```

`Collection.from_txt` is an alias of `read_txt`.

### Invalid lines

A line may be unreadable: too few numbers, text where a number should be, a
confidence outside `[0, 1]`, or a polygon without area. The `on_error`
argument decides what happens.

| `on_error` | Behaviour |
|------------|-----------|
| `"warn"` (default) | Skip the line and emit one [`HiresWarning`][hires.HiresWarning] per file |
| `"raise"` | Raise [`AnnotationFormatError`][hires.AnnotationFormatError] |
| `"ignore"` | Skip the line silently |

```python
>>> _ = Path("broken.txt").write_text(
...     "0 0.1 0.1 0.3 0.1 0.3 0.2\n"
...     "1 0.5 0.5 oops\n"
... )
>>> hires.read_txt("broken.txt", on_error="ignore")
Collection(name='broken', n=1)
>>> hires.read_txt("broken.txt", on_error="raise")
Traceback (most recent call last):
    ...
hires.errors.AnnotationFormatError: broken.txt:2: expected a class id and at least 3 points
```

### Polygon repair

Self-intersecting outlines, such as a "bow tie", are common in model
output. They are repaired on reading; if the repair splits the outline into
several parts, the largest part is kept:

```python
>>> _ = Path("bowtie.txt").write_text("0 0.0 0.0 0.4 0.4 0.4 0.0 0.0 0.2\n")
>>> repaired = hires.read_txt("bowtie.txt")[0]
>>> repaired.polygon.is_valid
True
>>> round(repaired.polygon.area, 4)
0.0533
```

## Reading a directory

[`hires.read_album`][hires.read_album] reads every `*.txt` file of a
directory, sorted by name, into an [`Album`][hires.Album] named after the
directory:

```python
>>> _ = Path("labels/scan_2.txt").write_text("0 0.2 0.2 0.4 0.2 0.4 0.6 0.2 0.6\n")
>>> album = hires.read_album("labels")
>>> album
Album(name='labels', collections=2, annotations=3)
>>> album.names
['scan', 'scan_2']
>>> album["scan_2"]
Collection(name='scan_2', n=1)
```

With `image_dir`, each `{stem}.txt` is bound to the image
`{image_dir}/{stem}` with the extension `.tif`, `.tiff`, `.png`, `.jpg` or
`.jpeg`, tried in that order. Label files without a matching image are
still loaded, just without an image size:

```python
>>> album = hires.read_album("labels", image_dir="images")
>>> album["scan"].image_size
(2000, 1000)
>>> album["scan_2"].image_size is None
True
```

An explicit list of files works too:

```python
>>> hires.read_album(["labels/scan.txt", "bowtie.txt"], name="selection")
Album(name='selection', collections=2, annotations=3)
```

`Album.from_dir` is an alias of `read_album`.

## Reading class names

Label files contain only class ids. The names live in the dataset
`data.yaml`, or inside the model weights:

```python
>>> _ = Path("data.yaml").write_text("names:\n  0: d_pulex\n  1: d_galeata\n")
>>> hires.read_class_names("data.yaml")
{0: 'd_pulex', 1: 'd_galeata'}
>>> hires.read_class_names("DaphnAI.pt")  # doctest: +SKIP
```

Reading names from weights needs the `yolo` extra. The resulting
`{class_id: label}` dict is what the plotting and evaluation functions
accept as `class_names=`.

## Exporting measurements

[`to_frame`][hires.Collection.to_frame] returns one row per annotation:

```python
>>> frame = collection.to_frame()
>>> list(frame.columns[:8])
['name', 'image_path', 'image_width', 'image_height', 'dpi', 'unit', 'class_id', 'confidence']
>>> frame[["class_id", "confidence", "area", "perimeter", "circularity"]]
   class_id  confidence     area  perimeter  circularity
0         0        0.87  40000.0  1000.0000       0.5027
1         1         NaN  60000.0  1121.1103       0.5999
```

| Column | Content |
|--------|---------|
| `name` | Collection name |
| `image_path` | Path of the image, if known |
| `image_width`, `image_height` | Image size in pixels, if known |
| `dpi`, `unit` | Physical scale, if set |
| `class_id` | Class label |
| `confidence` | Detection score, empty for ground truth |
| `area` … `obb_length` | The nine traits in [`hires.TRAIT_NAMES`][hires.TRAIT_NAMES] |

The units of the trait columns depend on what the collection knows (see
[Units](morphometrics.md#units)):

| The collection has | Lengths in | Areas in |
|--------------------|------------|----------|
| no `image_size` | normalized units (with a warning) | normalized units² |
| `image_size` | pixels | pixels² |
| `image_size`, `dpi` and `unit` | `unit` | `unit`² |

`to_csv` writes the same table to a file, and `to_records` returns it as a
list of dicts:

```python
>>> collection.to_csv("scan_shapes.csv")
>>> len(Path("scan_shapes.csv").read_text().splitlines())   # header + 2 rows
3
>>> collection.to_records()[1]["area"]
60000.0
```

The album methods of the same name concatenate all collections and add a
leading `album` column. To measure the whole album in pixels, every
collection needs its image, so we first add the one that was missing:

```python
>>> _ = cv2.imwrite("images/scan_2.png", np.zeros((500, 500, 3), dtype=np.uint8))
>>> album = hires.read_album("labels", image_dir="images")
>>> album.to_frame()[["album", "name", "class_id", "area"]]
    album    name  class_id     area
0  labels    scan         0  40000.0
1  labels    scan         1  60000.0
2  labels  scan_2         0  20000.0
```

`summary` gives one row per collection:

```python
>>> album.summary()[["name", "n_annotations", "image_width", "image_height"]]
     name  n_annotations  image_width  image_height
0    scan              2         2000          1000
1  scan_2              1          500           500
```

## Writing label files

```python
>>> collection.to_txt("out/scan.txt")
>>> print(Path("out/scan.txt").read_text())
0 0.100000 0.100000 0.300000 0.100000 0.300000 0.200000 0.100000 0.200000 0.870000
1 0.500000 0.500000 0.700000 0.500000 0.600000 0.800000
```

Coordinates are written with six decimals. With `confidence=False` the
confidence is left out, which turns predictions into training labels:

```python
>>> collection.to_txt("out/scan_training.txt", confidence=False)
>>> print(Path("out/scan_training.txt").read_text())
0 0.100000 0.100000 0.300000 0.100000 0.300000 0.200000 0.100000 0.200000
1 0.500000 0.500000 0.700000 0.500000 0.600000 0.800000
```

An album writes one `{name}.txt` per collection into a directory:

```python
>>> written = album.to_txt("out/album")
>>> sorted(path.name for path in written)
['scan.txt', 'scan_2.txt']
```

Only polygon exteriors are written; the format cannot represent holes.

## Saving crops

[`save_crops`][hires.Collection.save_crops] writes one image per
annotation, cropped to its bounding box. The collection must know its
image:

```python
>>> crops = collection.save_crops("crops", padding=10)
>>> [path.name for path in crops]
['scan_0000_class_0.png', 'scan_0001_class_1.png']
>>> cv2.imread(str(crops[0])).shape
(120, 420, 3)
```

Files are named `{prefix}_{index:04d}_class_{class_id}.png`, where the
prefix defaults to the collection name. Pixels outside the polygon are
blacked out unless `mask=False`. `Album.save_crops` writes one
sub-directory per collection.

## Images

The low-level helpers in
[`hires.io`](../reference/io.md#low-level-helpers-hiresio) are public:

```python
>>> [path.name for path in hires.io.find_images("images")]
['scan.png', 'scan_2.png']
>>> hires.io.read_image_size("images/scan.png")
(2000, 1000)
>>> hires.io.read_image("images/scan.png").shape
(1000, 2000, 3)
```

Three conventions are worth knowing:

- Sizes are `(width, height)`; array shapes are `(height, width, channels)`,
  as in NumPy.
- Images are read as **8-bit BGR** arrays, the OpenCV convention. Functions
  in [`hires.plot`](../reference/plot.md) *return* RGB arrays, ready for
  Matplotlib.
- **EXIF orientation is ignored.** Pixel coordinates always refer to the
  stored raster, so that labels, image sizes and pixels agree.

## See also

- Reference: [Input and output](../reference/io.md)
- [The data model](data-model.md)
