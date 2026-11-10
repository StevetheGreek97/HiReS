# The `hires` command

Installing HiReS adds the `hires` command. It exposes the most common
workflows without writing Python.

```text
hires [--version] [--help] COMMAND [options]
```

| Command | Purpose | Python equivalent |
|---------|---------|-------------------|
| [`hires run`](#hires-run) | Segment images and write labels, tables and overlays | [`hires.run_pipeline`][hires.run_pipeline] |
| [`hires chunk`](#hires-chunk) | Cut images into overlapping tiles | [`hires.tiling.tile_image`][hires.tiling.tile_image] |
| [`hires plot`](#hires-plot) | Draw existing label files on their images | [`hires.plot.overlay`][hires.plot.overlay] |
| [`hires evaluate`](#hires-evaluate) | Compare predictions with ground truth | [`hires.evaluate`][hires.evaluate] |

`hires COMMAND --help` prints the options of a command with their defaults
and is always authoritative for the installed version.

!!! note "Terminology"

    The command line speaks of *chunks* and the Python API of *tiles*. They
    are the same thing; the option names are kept from HiReS 1.x.

## hires run

Run the full pipeline on one image or on every image in a directory:
tile → predict → filter edges → merge → NMS → write outputs. Needs the
`yolo` extra.

```bash
hires run -s scans/ -m DaphnAI.pt -o results/ --dpi 2400 --unit um
```

| Option | Default | Description |
|--------|---------|-------------|
| `-s`, `--source PATH` | required | Image file or directory |
| `-m`, `--model PATH` | `models/DaphnAI.pt` | YOLO segmentation weights |
| `-o`, `--output DIR` | `results` | Root output directory |
| `-r`, `--recursive` | off | Search the source directory recursively; the sub-directory layout is mirrored in the output |
| `--conf FLOAT` | `0.5` | Minimum detection confidence |
| `--imgsz INT` | `1024` | Size the model resizes each chunk to |
| `--device STR` | `cpu` | Inference device: `cpu`, `0`, `cuda:0`, … |
| `--batch-size INT` | `8` | Chunks per model call |
| `--chunk-size W H` | `1024 1024` | Chunk width and height in pixels |
| `--overlap INT` | `150` | Pixels shared by adjacent chunks |
| `--edge-threshold FLOAT` | `0.01` | Detections closer than this fraction of a chunk to its border are discarded |
| `--iou-thresh FLOAT` | `0.7` | Polygon IoU above which the less confident of two detections is removed |
| `--save-crops` | off | Write a masked crop of every detection |
| `--dpi FLOAT` | none | Scan resolution; with `--unit`, traits are written in physical units |
| `--unit UNIT` | none | `nm`, `um`, `mm`, `cm`, `m` or `inch`; requires `--dpi` |
| `--debug` | off | Write intermediate files |

An image that fails is logged and skipped; the remaining images are still
processed. The meaning of the tiling and inference options is explained in
[Tiled segmentation](../user-guide/segmentation.md#choosing-the-parameters);
the files written are listed in [Output files](outputs.md#hires-run).

## hires chunk

Cut one image, or every image in a directory, into overlapping PNG tiles
named `{stem}_{x}_{y}.png`.

```bash
hires chunk -s scans/ -o chunks/ --chunk-size 1024 1024 --overlap 150
```

| Option | Default | Description |
|--------|---------|-------------|
| `-s`, `--source PATH` | required | Image file or directory |
| `-o`, `--output DIR` | `chunks` | Output directory |
| `-r`, `--recursive` | off | Search the source directory recursively |
| `--chunk-size W H` | `1024 1024` | Chunk width and height in pixels |
| `--overlap INT` | `150` | Pixels shared by adjacent chunks |

To tile the labels of an annotated image as well, use
[`hires.tiling.tile_annotations`][hires.tiling.tile_annotations] from
Python; see [Tiling training data](../user-guide/tiling.md).

## hires plot

Draw existing YOLO-format label files on their images and write
`{stem}_annotated.png` to the output directory.

```bash
hires plot -s scans/ --ann labels/ -o plots/ -m data.yaml
```

| Option | Default | Description |
|--------|---------|-------------|
| `-s`, `--source PATH` | required | Image file or directory |
| `--ann PATH` | the output directory | A `.txt` file, or a directory of `.txt` files matched to images by file stem |
| `-m`, `--model PATH` | `models/DaphnAI.pt` | Model weights (`.pt`) or a data YAML file; read for class names only |
| `-o`, `--output DIR` | `results` | Output directory |
| `-r`, `--recursive` | off | Search the source directory recursively |

Without `--ann`, the label file of each image is expected at
`{output}/{stem}.txt`, which is where `hires run` writes it.

## hires evaluate

Compare a directory of predicted label files with a directory of
ground-truth label files. Files are paired by name. The per-class detection
report is printed and the full report is written to the output directory.

```bash
hires evaluate --gt labels/ --pred results/ --images scans/ -n data.yaml
```

| Option | Default | Description |
|--------|---------|-------------|
| `--gt DIR` | required | Directory of ground-truth `.txt` files |
| `--pred DIR` | required | Directory of predicted `.txt` files, e.g. the output of `hires run` |
| `--images DIR` | none | Directory of the images; enables measurements in pixels and comparison images |
| `-n`, `--names PATH` | none | Model weights (`.pt`) or a data YAML file; read for class names only |
| `-o`, `--output DIR` | `evaluation` | Output directory |
| `--iou FLOAT` | `0.5` | Minimum IoU for a prediction to match a ground-truth object |
| `--no-images` | off | Do not write comparison images |

Label files present on only one side are skipped with a warning. The
metrics are defined in [Evaluation](../user-guide/evaluation.md); the files
written are listed in [Output files](outputs.md#hires-evaluate).

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Partial success: `hires run` skipped at least one image that failed, or `hires plot` found no label file for at least one image |
| `2` | Error: invalid arguments, a missing file or directory, nothing to process, or a missing optional dependency |

## Logging

The command line prints progress messages at level `INFO` to standard
error, in the form `[HH:MM:SS] LEVEL | message`. Only the report table of
`hires evaluate` goes to standard output, so it can be redirected to a file.
