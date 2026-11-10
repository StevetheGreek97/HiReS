# Migrating from 1.x

HiReS 2.0 is a rewrite of the Python API. The command line is largely
unchanged; Python code written against 1.x needs to be updated. This page
maps the old names to the new ones.

## What changed, in short

- **One flat namespace.** Everything common is imported from `hires`
  directly. The subpackages `hires.models`, `hires.pipeline`,
  `hires.processing`, `hires.operations` and `hires.viz` are gone.
- **Functions instead of pipeline classes.** The `Settings` dataclass and the
  `SegmentationPipeline` / `PlottingPipeline` classes are replaced by
  functions that take keyword arguments.
- **Immutable objects.** `Annotation`, `Collection` and `Album` can no longer
  be modified in place. Methods that used to mutate now return a new object.
- **Traits are computed on demand.** An annotation no longer carries its
  image size or scale; the collection does.
- **YOLO is optional.** `ultralytics` moved to the `yolo` extra; `scipy`,
  `plotnine` and `SkillMetrics` moved to the `analysis` extra; `tqdm` is no
  longer used.
- **New:** `hires evaluate`, [`hires.metrics`](../reference/metrics.md),
  [`hires.plot`](../reference/plot.md), a dedicated exception hierarchy and
  standard-library logging.

## Running the pipeline

=== "1.x"

    ```python
    from hires.models import Settings
    from hires.pipeline.seg_pipeline import SegmentationPipeline

    cfg = Settings(
        source="data/images/",
        model_path="models/model.pt",
        output_dir="results/",
        conf=0.58,
        chunk_size=(1024, 1024),
        overlap=300,
        iou_thresh=0.7,
        save_crops=True,
    )
    SegmentationPipeline(cfg).run()
    ```

=== "2.0"

    ```python
    import hires

    album = hires.run_pipeline(
        "data/images/",
        "models/model.pt",
        "results/",
        confidence=0.58,
        tile_size=(1024, 1024),
        overlap=300,
        nms_iou=0.7,
        save_crops=True,
    )
    ```

### `Settings` fields

| 1.x `Settings` field | 2.0 argument | Of |
|----------------------|--------------|----|
| `source` | `source` | [`run_pipeline`][hires.run_pipeline] |
| `model_path` | `model` | `run_pipeline`, [`Segmenter`][hires.Segmenter] |
| `output_dir` | `output_dir` | `run_pipeline` |
| `recursive` | `recursive` | `run_pipeline` |
| `conf` | `confidence` | `Segmenter` |
| `imgsz` | `inference_size` | `Segmenter` |
| `device` | `device` | `Segmenter` |
| `chunk_size` | `tile_size` | `Segmenter` |
| `overlap` | `overlap` | `Segmenter` |
| `edge_threshold` | `edge_margin` | `Segmenter` |
| `iou_thresh` | `nms_iou` | `Segmenter` |
| `save_crops` | `save_crops` | `run_pipeline` |
| `dpi`, `unit` | `dpi`, `unit` | `run_pipeline`, [`Collection.with_scale`][hires.Collection.with_scale] |
| `debug` | `debug` | `run_pipeline` |
| `ann` | — | Pass annotations to [`hires.plot.overlay`][hires.plot.overlay] |
| `match_iou` | `iou_threshold` | [`hires.metrics.match`][hires.metrics.match] and friends |
| `gt_ann`, `pred_ann` | — | Read both sides with [`read_album`][hires.read_album] |

`run_pipeline` forwards any `Segmenter` argument, so all of them can be
passed in one call.

!!! note "Chunks and tiles"
    The Python API now says *tile* where 1.x said *chunk*. The command line
    keeps the old spelling (`hires chunk`, `--chunk-size`).

## Other 1.x entry points

| 1.x | 2.0 |
|-----|-----|
| `ImageChunker(src).slice(save_folder, chunk_size, overlap)` | [`hires.tiling.tile_image(image, directory, tile_size=..., overlap=...)`][hires.tiling.tile_image], once per image |
| `PlottingPipeline(cfg).run()` | [`hires.plot.overlay(image, collection, save=...)`][hires.plot.overlay] |
| `Collection.read_txt(path, image_path=...)` | [`hires.read_txt(path, image=...)`][hires.read_txt] or `Collection.from_txt` |
| `Album.from_dir(dir, image_dir=..., album_name=...)` | [`hires.read_album(dir, image_dir=..., name=...)`][hires.read_album] or `Album.from_dir` |
| `collection.set_scale(dpi=..., unit=...)` (in place) | `collection = collection.with_scale(dpi=..., unit=...)` |
| `collection.to_df()` | [`collection.to_frame()`][hires.Collection.to_frame] |
| `ann.to_dict()` | [`ann.traits(image_size)`][hires.Annotation.traits] |
| `PairInspector(gt_album=gt, pred_album=pred).pairs_df(iou_threshold=t)` | [`hires.metrics.match(gt, pred, iou_threshold=t).to_frame()`][hires.metrics.match] |

## Measurements

In 1.x the traits were properties of the annotation. In 2.0 they are computed
by [`Annotation.traits`][hires.Annotation.traits] (or the stand-alone
[`hires.measure`][hires.measure]) and returned as a dict:

=== "1.x"

    ```python
    ann.area
    ann.circularity
    width, length = ann.oriented_bounding_box.width_length
    ```

=== "2.0"

    ```python
    traits = ann.traits(collection.image_size, pixel_size=collection.pixel_size)
    traits["area"]
    traits["circularity"]
    traits["obb_width"], traits["obb_length"]
    ```

In practice you rarely measure a single annotation;
[`Collection.to_frame`][hires.Collection.to_frame] measures all of them with
the right image size and scale.

## Tables

- The identifier column of trait tables is now `name` (1.x:
  `collection_name`).
- The match table returned by
  [`MatchResult.to_frame`][hires.metrics.MatchResult.to_frame] uses the
  prefixes `gt_` and `pred_` (1.x `pairs_df`: `left_` and `right_`).

## Debug output

The layout of `{stem}_debug/` changed; see
[Output files](../reference/outputs.md#debug-files).

## Command line

`hires run`, `hires chunk` and `hires plot` accept the same options as
before. New in 2.0:

- `hires evaluate` compares predictions with ground truth.
- `hires run --batch-size` sets the number of tiles per model call.
- `hires --version`.
- Exit codes are meaningful; see
  [The hires command](../reference/cli.md#exit-codes).
