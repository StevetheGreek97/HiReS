# CLI Reference

Once installed, HiReS exposes a `hires` command.

```
hires <command> [options]
```

---

## Commands

| Command | Description |
|---------|-------------|
| `hires run` | Full segmentation pipeline |
| `hires chunk` | Split images into tiles only |
| `hires plot` | Render annotation overlays |
| `hires compare` | Evaluate predictions against ground truth |

---

## `hires run`

Runs the complete pipeline: chunk → predict → filter → merge → NMS → save outputs.

```bash
hires run --source data/ --model models/model.pt --out results/
```

| Flag | Description | Default |
|------|-------------|---------|
| `--source` | Image file or directory | required |
| `--model` | Path to YOLO `.pt` weights | required |
| `--out` | Output directory | required |
| `--conf` | Confidence threshold | `0.5` |
| `--imgsz` | Inference image size (px) | `1024` |
| `--device` | `cpu`, `cuda:0`, `mps` | `cpu` |
| `--chunk-size` | Tile size: `width height` | `1024 1024` |
| `--overlap` | Overlap between tiles (px) | `150` |
| `--edge-thr` | Edge-touch filter threshold | `0.01` |
| `--iou-thr` | NMS IoU threshold | `0.7` |
| `--recursive` | Recurse into subdirectories | `False` |
| `--debug` | Save chunk-level debug artifacts | `False` |

---

## `hires chunk`

Splits an image (or directory) into overlapping tiles without running inference.

```bash
hires chunk --source image.tif --out chunks/ --overlap 150
```

| Flag | Description | Default |
|------|-------------|---------|
| `--source` | Image file or directory | required |
| `--out` | Output directory | required |
| `--chunk-size` | Tile size: `width height` | `1024 1024` |
| `--overlap` | Overlap between tiles (px) | `150` |
| `--recursive` | Recurse into subdirectories | `False` |

> **Note:** `--chunk-size` is currently not wired through the CLI due to a known bug. Use the Python API (`Settings(chunk_size=...)`) to override the default.

---

## `hires plot`

Renders YOLO-format polygon annotations onto the source image.

```bash
hires plot --image image.tif --ann results/image.txt --out results/ --model model.pt
```

| Flag | Description |
|------|-------------|
| `--image` | Source image path |
| `--ann` | YOLO annotation `.txt` file |
| `--out` | Output directory |
| `--model` | YOLO weights or `data.yaml` (used for class names) |

Output: `<out>/<image>_annotated.tif`

---

## `hires compare`

Compares a prediction file against a ground-truth file and writes color-coded overlays plus a summary JSON.

```bash
hires compare --pred pred.txt --gt gt.txt --img image.tif --model model.pt --out results/
```

| Flag | Description | Default |
|------|-------------|---------|
| `--pred` | Prediction annotation file | required |
| `--gt` | Ground-truth annotation file | required |
| `--img` / `--image` | Source image path | required |
| `--model` | YOLO weights or `data.yaml` | required |
| `--out` | Output directory | `results` |
| `--iou-thr` | IoU threshold for TP matching | `0.5` |

**Outputs:**

| File | Contents |
|------|----------|
| `<image>_compare_overlay.tif` | Combined TP/FP/FN overlay |
| `<image>_compare_tp.tif` | Matched predictions with GT outlines |
| `<image>_compare_fp.tif` | Unmatched predictions |
| `<image>_compare_fn.tif` | Unmatched ground-truth polygons |
| `<image>_compare_summary.json` | Counts and matched index pairs |
