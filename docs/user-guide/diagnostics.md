# Errors, warnings and logging

HiReS reports problems in three ways, following the conventions of the
scientific Python stack:

| Mechanism | Used for | You see it |
|-----------|----------|------------|
| Exceptions | The operation cannot be completed | Always |
| Warnings | The operation completed, but probably not as intended | Once per location, by default |
| Logging | Progress and routine information | Only if you enable it |

```python
>>> from pathlib import Path
>>> from shapely.geometry import box
>>> import hires
```

## Exceptions

Errors specific to HiReS derive from [`HiresError`][hires.HiresError], so
one `except` clause catches them all:

```python
>>> _ = Path("broken.txt").write_text("0 0.1 0.1 oops\n")
>>> try:
...     hires.read_txt("broken.txt", on_error="raise")
... except hires.HiresError as error:
...     print("Could not load labels:", error)
Could not load labels: broken.txt:1: expected a class id and at least 3 points
```

| Exception | Raised when | Also a |
|-----------|-------------|--------|
| [`AnnotationFormatError`][hires.AnnotationFormatError] | A label file or line cannot be parsed | `ValueError` |
| [`MissingDependencyError`][hires.MissingDependencyError] | A feature needs an optional dependency that is not installed | `ImportError` |

Because of the second base class, code that already catches `ValueError`
or `ImportError` keeps working:

```python
>>> issubclass(hires.AnnotationFormatError, ValueError)
True
>>> issubclass(hires.MissingDependencyError, ImportError)
True
```

Everything else uses the built-in exception that describes the problem:

| Exception | Typical cause |
|-----------|---------------|
| `ValueError` | An argument is out of range or inconsistent |
| `TypeError` | An argument has the wrong type |
| `FileNotFoundError` | An image, label file or directory does not exist |
| `KeyError`, `LookupError` | A collection name, class label or match does not exist |

```python
>>> collection = hires.Collection(
...     [hires.Annotation(0, box(0.1, 0.1, 0.3, 0.3), confidence=0.9)],
...     name="scan",
...     image_size=(1000, 1000),
... )
>>> collection.with_scale(dpi=2400, unit="furlong")
Traceback (most recent call last):
    ...
ValueError: Unsupported unit 'furlong'; choose one of ['cm', 'inch', 'm', 'mm', 'nm', 'um'].
>>> hires.tiling.tile_grid((2000, 2000), tile_size=512, overlap=512)
Traceback (most recent call last):
    ...
ValueError: overlap must be non-negative and smaller than the tile size.
>>> hires.metrics.match(collection, hires.Album([collection]))
Traceback (most recent call last):
    ...
TypeError: gt and pred must be two Collection objects or two Album objects, got Collection and Album.
>>> hires.read_txt("missing.txt")
Traceback (most recent call last):
    ...
FileNotFoundError: Annotation file not found: missing.txt
>>> hires.Album([collection])["other"]
Traceback (most recent call last):
    ...
KeyError: "No collection named 'other' in the album."
```

### Missing optional dependencies

The message of a `MissingDependencyError` names the extra to install:

```text
hires.errors.MissingDependencyError: Segmentation requires the optional
dependency 'ultralytics'. Install it with: pip install 'hireseg[yolo]'
```

See [Installation](../getting-started/installation.md#optional-dependencies)
for what each extra provides.

### Choosing between failing and continuing

Functions that process many items let you decide what a single bad item
does:

| Function | Parameter | Choices |
|----------|-----------|---------|
| [`read_txt`][hires.read_txt], [`read_album`][hires.read_album] | `on_error` | `"warn"` (default): skip invalid lines and warn; `"raise"`; `"ignore"` |
| [`run_pipeline`][hires.run_pipeline] | `on_error` | `"raise"` (default): stop at the first failing image; `"skip"`: log it and continue |

The `hires run` command uses `"skip"` and signals skipped images with its
[exit code](../reference/cli.md#exit-codes).

## Warnings

HiReS warns with [`HiresWarning`][hires.HiresWarning], a subclass of
`UserWarning`, when an operation succeeded but the result is probably not
what you intended. To show the messages on this page, we collect them
instead of letting Python print them:

```python
>>> import warnings
>>> def warnings_of(function):
...     with warnings.catch_warnings(record=True) as caught:
...         warnings.simplefilter("always")
...         function()
...     return [str(w.message) for w in caught]
```

**Lines were skipped while reading a label file.** The file was probably
not written in the YOLO segmentation format, or is damaged:

```python
>>> warnings_of(lambda: hires.read_txt("broken.txt"))
['Skipped 1 unreadable line(s) in broken.txt (first: broken.txt:1: expected a class id and at least 3 points).']
```

**Traits were measured without an image size.** The values are then in
normalized units, which distort the shape of objects in non-square images.
Read the labels with `image=` or `image_dir=`, or use
[`with_image`][hires.Collection.with_image]:

```python
>>> no_size = collection.replace(image_size=None)
>>> warnings_of(no_size.to_frame)
["Collection 'scan' has no image_size; traits are measured in normalized coordinates. Load it with an image to measure in pixels."]
```

**Two collections of an album share a name.** The later one is renamed:

```python
>>> warnings_of(lambda: hires.Album([collection, collection]))
["1 collection(s) with duplicate names were renamed (e.g. 'scan' -> 'scan__1')."]
```

While developing a pipeline, it is good practice to turn these warnings
into errors, so that nothing slips through:

```python
>>> with warnings.catch_warnings():
...     warnings.simplefilter("error", hires.HiresWarning)
...     no_size.to_frame()
Traceback (most recent call last):
    ...
hires.errors.HiresWarning: Collection 'scan' has no image_size; traits are measured in normalized coordinates. Load it with an image to measure in pixels.
```

To apply this to a whole program, put
`warnings.simplefilter("error", hires.HiresWarning)` at its top. To silence
a warning you have understood, use `"ignore"` in the same way:

```python
>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore", hires.HiresWarning)
...     frame = no_size.to_frame()
>>> len(frame)
1
```

Under pytest, warnings are turned into errors in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
filterwarnings = ["error::hires.HiresWarning"]
```

## Logging

As a library, HiReS never prints and never configures logging. Progress
messages are sent to the standard `logging` module under the logger name
`hires`, and are discarded unless the application enables them:

```python
>>> import logging
>>> logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
```

After that, a pipeline run reports what it is doing:

```text
INFO hires.segmentation: [1/2] scans/scan_01.tif
INFO hires.segmentation: scan_01: 14040 x 9921 px, 192 tiles
INFO hires.segmentation: scan_01: 87 detections, class counts {0: 61, 1: 26}
INFO hires.segmentation: [2/2] scans/scan_02.tif
...
INFO hires.segmentation: Finished: 2 of 2 images -> results
```

To enable the messages of HiReS without those of other libraries, set the
level on its logger only:

```python
>>> logging.getLogger("hires").setLevel(logging.INFO)
```

| Level | Messages |
|-------|----------|
| `INFO` | Progress: image being processed, tiles, detections, files written |
| `WARNING` | Something was skipped, e.g. collections present in only one album during evaluation |
| `ERROR` | An image failed and was skipped (`on_error="skip"`), with the traceback |

Loggers are named after the module that emits the message
(`hires.segmentation`, `hires.metrics`), so they can be tuned separately.
The `hires` command enables `INFO` itself.

Ultralytics has its own, separate logging. HiReS calls the model with
`verbose=False`, so per-tile output is suppressed.

## Reporting a bug

Include the output of [`hires.show_versions`][hires.show_versions], which
lists the versions of HiReS, Python and every dependency:

```python
>>> hires.show_versions()  # doctest: +SKIP
hireseg      : 2.0.0
python       : 3.12.3
platform     : Linux-6.8.0-x86_64-with-glibc2.39
numpy        : 2.1.3
pandas       : 2.2.3
shapely      : 2.0.6
...
```

together with the full traceback and, if possible, a small file that
reproduces the problem. Bugs are tracked at
<https://github.com/StevetheGreek97/HiReS/issues>.

## See also

- Reference: [Exceptions and warnings](../reference/errors.md)
