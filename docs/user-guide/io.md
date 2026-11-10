# Reading and writing

HiReS reads and writes **YOLO segmentation label files** and exports
measurements as tables.

## The label format

A label file is a plain-text file with one object per line:

```text
class_id x1 y1 x2 y2 ... xn yn [confidence]
```

`class_id`
:   Integer class label.

`x1 y1 ... xn yn`
:   The polygon vertices, in normalized coordinates (fractions of the image
    width and height). At least three points are required. The polygon is
    closed implicitly.

`confidence`
:   Optional detection score in `[0, 1]`. Because every vertex contributes
    two numbers, a trailing value that makes the count odd is read as the
    confidence.

```text
0 0.412300 0.215400 0.418800 0.221000 0.415100 0.229700 0.409000 0.222100 0.874512
1 0.630200 0.701900 0.641000 0.708300 0.636400 0.719600
```

The first line is a prediction with a confidence of 0.87; the second is a
ground-truth object without one. This is the format written by Ultralytics
with `save_txt=True, save_conf=True`, and the format expected for training.

By convention the label file of `scan.tif` is called `scan.txt`.

## Reading one file

```python
collection = hires.read_txt("labels/scan.txt", image="images/scan.tif")
```

[`hires.read_txt`][hires.read_txt] returns a
[`Collection`][hires.Collection] named after the file stem. The `image`
argument is optional but recommended: HiReS reads the image **size** from the
file header (the pixels are not decoded, so this is fast even for very large
scans) and can then measure in pixels. If the image is not at hand, pass
`image_size=(width, height)` instead.

`Collection.from_txt` is an alias of the same function.

### Invalid lines

Lines can be unreadable: too few tokens, non-numeric values, a confidence
outside `[0, 1]`, or a polygon without area. The `on_error` argument decides
what happens:

| `on_error` | Behaviour |
|------------|-----------|
| `"warn"` (default) | Skip the line and emit one [`HiresWarning`][hires.HiresWarning] per file |
| `"raise"` | Raise [`AnnotationFormatError`][hires.AnnotationFormatError] with file name and line number |
| `"ignore"` | Skip the line silently |

### Polygon repair

Self-intersecting outlines (a "bow tie") are common in model output. They
are repaired with a zero-width buffer; if the repair splits the outline into
several parts, the largest part is kept. Only outlines that cannot yield a
polygon with positive area are treated as invalid.

## Reading a directory

```python
gt = hires.read_album("labels/", image_dir="images/")
```

[`hires.read_album`][hires.read_album] reads every `*.txt` file of a
directory, sorted by name, into an [`Album`][hires.Album] named after the
directory. With `image_dir`, each `{stem}.txt` is bound to
`{image_dir}/{stem}{ext}`, trying the extensions `.tif`, `.tiff`, `.png`,
`.jpg` and `.jpeg` in that order. Label files without a matching image are
still loaded, just without an image size.

An explicit list of files works too:

```python
album = hires.read_album(["a/scan1.txt", "b/scan2.txt"], name="selection")
```

`Album.from_dir` is an alias of `read_album`.

## Reading class names

Label files contain only class ids. The names live in the dataset
`data.yaml` or inside the model weights:

```python
class_names = hires.read_class_names("data.yaml")     # {0: "d_pulex", 1: "d_galeata"}
class_names = hires.read_class_names("DaphnAI.pt")    # needs the yolo extra
```

The resulting `{class_id: label}` dict is accepted as `class_names=` by the
plotting and evaluation functions.

## Exporting measurements

```python
frame = collection.to_frame()       # pandas.DataFrame
collection.to_csv("scan_shapes.csv")
records = collection.to_records()   # list of dicts
```

One row per annotation, with these columns:

| Column | Content |
|--------|---------|
| `name` | Collection name |
| `image_path` | Path of the image, if known |
| `image_width`, `image_height` | Image size in pixels, if known |
| `dpi`, `unit` | Physical scale, if set |
| `class_id` | Class label |
| `confidence` | Detection score, empty for ground truth |
| `area` … `obb_length` | The nine traits in [`hires.TRAIT_NAMES`][hires.TRAIT_NAMES] |

The units of the trait columns depend on the metadata of the collection:

| The collection has | Lengths in | Areas in |
|--------------------|------------|----------|
| no `image_size` | normalized units (with a warning) | normalized units² |
| `image_size` | pixels | pixels² |
| `image_size`, `dpi` and `unit` | `unit` | `unit`² |

The album methods of the same name concatenate all collections and add a
leading `album` column:

```python
album.with_scale(dpi=2400, unit="um").to_csv("all_traits.csv")
album.summary()   # one row per collection: size and image metadata
```

## Writing label files

```python
collection.to_txt("out/scan.txt")
album.to_txt("out/")                     # one {name}.txt per collection
collection.to_txt("gt/scan.txt", confidence=False)
```

Coordinates are written with six decimals. Use `confidence=False` to produce
training labels from predictions. Only polygon exteriors are written; the
format cannot represent holes.

## Saving crops

```python
paths = collection.save_crops("crops/", padding=10)
```

[`save_crops`][hires.Collection.save_crops] writes one image per
annotation, cropped to its bounding box and named
`{prefix}_{index:04d}_class_{class_id}.png`. By default pixels outside the
polygon are blacked out (`mask=True`). The collection must be bound to an
image (`image_path`). `Album.save_crops` writes one sub-directory per
collection.

## Images

The low-level helpers in [`hires.io`](../reference/io.md#low-level-helpers-hiresio)
are public:

```python
hires.io.find_images("scans/", recursive=True)   # sorted list of image paths
hires.io.read_image_size("scan.tif")             # (width, height), header only
hires.io.read_image("scan.tif")                  # 8-bit BGR array
```

Two conventions are worth knowing:

- Images are read as **8-bit BGR** arrays, the OpenCV convention. Functions
  in [`hires.plot`](../reference/plot.md) *return* RGB arrays, ready for
  Matplotlib.
- **EXIF orientation is ignored.** Pixel coordinates always refer to the
  stored raster, so that labels, image sizes and pixels agree.
