# CLI Reference

Once installed, HiReS exposes a `hires` command.

```
hires <command> [options]
```

Run `hires <command> --help` for the full, authoritative option list of any command.

---

## Commands

| Command | Description |
|---------|-------------|
| `hires run` | Full segmentation pipeline |
| `hires chunk` | Split images into tiles only |
| `hires plot` | Render annotation overlays |

---

## `hires run`

Runs the complete pipeline: chunk → predict → filter edges → unify → NMS → visualise → crops + CSV.

```bash
hires run -s data/ -m models/DaphnAI.pt -o results/
```

| Flag | Description | Default |
|------|-------------|---------|
| `-s`, `--source` | Image file or directory | **required** |
| `-m`, `--model` | Path to YOLO `.pt` weights | `models/DaphnAI.pt` |
| `-o`, `--output` | Output directory | `results` |
| `-r`, `--recursive` | Recurse into subdirectories | `False` |
| `--conf` | Confidence threshold | `0.5` |
| `--imgsz` | Inference image size (px) | `1024` |
| `--device` | `cpu`, `0`, `cuda:0`, … | `cpu` |
| `--chunk-size` | Tile size: `width height` | `1024 1024` |
| `--overlap` | Overlap between tiles (px) | `150` |
| `--edge-threshold` | Edge-touch filter inset | `0.01` |
| `--iou-thresh` | NMS IoU threshold | `0.7` |
| `--save-crops` | Save a masked crop per detection | `False` |
| `--dpi` | Scan resolution in DPI — enables physical measurements | `None` |
| `--unit` | Physical unit for descriptors: `nm`, `um`, `mm`, `cm`, `m`, `inch` | `None` |
| `--debug` | Save chunk-level debug artifacts | `False` |

Examples:

```bash
hires run -s image.tif
hires run -s images/ -o results/ --conf 0.4 --device 0
hires run -s images/ --chunk-size 2048 2048 --overlap 256 --iou-thresh 0.6
hires run -s images/ --dpi 1200 --unit um
hires run -s images/ -r --debug
```

---

## `hires chunk`

Splits an image (or directory) into overlapping tiles without running inference.

```bash
hires chunk -s image.tif -o chunks/ --overlap 150
```

| Flag | Description | Default |
|------|-------------|---------|
| `-s`, `--source` | Image file or directory | **required** |
| `-o`, `--output` | Output directory for chunk images | `chunks` |
| `--chunk-size` | Tile size: `width height` | `1024 1024` |
| `--overlap` | Overlap between tiles (px) | `150` |

Chunk images are written as `{stem}_{x}_{y}.png`, where `x`/`y` are the top-left
pixel offsets of the tile in the original image.

---

## `hires plot`

Renders YOLO-format polygon annotations onto the source image.

```bash
hires plot -s image.tif --ann results/image.txt -o results/ -m models/DaphnAI.pt
```

| Flag | Description | Default |
|------|-------------|---------|
| `-s`, `--source` | Image file or directory to annotate | **required** |
| `-m`, `--model` | YOLO weights (`.pt`) or `data.yaml` — used for class names only | `models/DaphnAI.pt` |
| `-o`, `--output` | Output directory for annotated images | `results` |
| `--ann` | Annotation `.txt` file or directory of `.txt` files. When a directory is given, files are matched to images by stem name. | `""` |
| `-r`, `--recursive` | Recurse into subdirectories | `False` |

If `--ann` is omitted, `hires plot` looks for `<output>/<image_stem>.txt`.

Output: `<output>/<image_stem>_annotated.png`
