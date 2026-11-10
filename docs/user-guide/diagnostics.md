# Errors, warnings and logging

HiReS reports problems in three ways, following the conventions of the
scientific Python stack:

| Mechanism | Used for | You see it |
|-----------|----------|------------|
| Exceptions | The operation cannot be completed | Always |
| Warnings | The operation completed, but probably not as intended | Once per location, by default |
| Logging | Progress and routine information | Only if you enable it |

## Exceptions

Errors specific to HiReS derive from [`HiresError`][hires.HiresError], so
one `except` clause catches them all:

```python
try:
    collection = hires.read_txt("labels/scan.txt", on_error="raise")
except hires.HiresError as error:
    print(f"Could not load labels: {error}")
```

| Exception | Raised when | Also a |
|-----------|-------------|--------|
| [`AnnotationFormatError`][hires.AnnotationFormatError] | A label file or line cannot be parsed | `ValueError` |
| [`MissingDependencyError`][hires.MissingDependencyError] | A feature needs an optional dependency that is not installed | `ImportError` |

Because of the second base class, code that already catches `ValueError`
or `ImportError` keeps working.

Everything else uses the built-in exception that describes the problem:

| Exception | Typical cause |
|-----------|---------------|
| `ValueError` | An argument is out of range, e.g. `overlap` not smaller than the tile size, or `dpi` given without `unit` |
| `TypeError` | An argument has the wrong type, e.g. comparing a `Collection` with an `Album` |
| `FileNotFoundError` | An image, label file or directory does not exist |
| `KeyError`, `LookupError` | A collection name, class label or match does not exist |

### Missing optional dependencies

The message of a `MissingDependencyError` names the extra to install:

```text
Segmentation requires the optional dependency 'ultralytics'. Install it with: pip install 'hireseg[yolo]'
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
`UserWarning`, in two situations:

- **Lines were skipped while reading a label file.** The message gives the
  number of lines and the first problem. The file was probably not written
  in the YOLO segmentation format, or is damaged.
- **Traits were measured without an image size.** The values are then in
  normalized units, which distort the shape of objects in non-square
  images. Read the labels with `image=` or `image_dir=`, or use
  [`with_image`][hires.Collection.with_image].

While developing a pipeline, it is good practice to turn them into errors,
so that nothing slips through:

```python
import warnings

warnings.simplefilter("error", hires.HiresWarning)
```

To silence a warning you have understood, for a block of code only:

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore", hires.HiresWarning)
    frame = collection.to_frame()
```

Under pytest, the equivalent of the first snippet is

```toml
[tool.pytest.ini_options]
filterwarnings = ["error::hires.HiresWarning"]
```

## Logging

As a library, HiReS never prints and never configures logging. Progress
messages are sent to the standard `logging` module under the logger name
`hires`, and are discarded unless the application enables them:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

After that, a pipeline run reports each image, its size and tile count, and
the number of detections per class.

To enable the messages of HiReS without those of other libraries:

```python
logging.basicConfig()
logging.getLogger("hires").setLevel(logging.INFO)
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

Include the output of

```python
hires.show_versions()
```

which lists the versions of HiReS, Python and every dependency, together
with the full traceback and, if possible, a small file that reproduces the
problem. Bugs are tracked at
<https://github.com/StevetheGreek97/HiReS/issues>.

## See also

- Reference: [Exceptions and warnings](../reference/errors.md)
