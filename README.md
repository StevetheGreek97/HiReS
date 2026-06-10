# HiReS
**High-Resolution Image Segmentation and Analysis Pipeline**

HiReS is a modular Python package and command-line tool for automated image segmentation and analysis.
It targets high-resolution microscopy or biological datasets and combines **YOLO-based instance segmentation**
with **geometry-aware postprocessing** (Shapely).

HiReS makes it easy to:
- Split large `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg` images into overlapping chunks.
- Run YOLO segmentation on each chunk.
- Merge predictions into full-image coordinates.
- Filter edge-touching polygons and apply IoU-based NMS.
- Generate overlays, object crops, and shape-descriptor tables.

Full documentation: <https://stevethegreek97.github.io/HiReS/>

---

## Key Features
- Chunking with overlap for arbitrarily large images (no GPU required).
- YOLO segmentation per chunk.
- Polygon edge filtering, unification, and IoU-based NMS.
- Overlay rendering for quick QA.
- Per-object crops and shape descriptors (CSV).
- Optional physical-unit conversion (µm, mm, …) from scan DPI.
- A `hires.models.eval` API for comparing predictions to ground truth.

---

## Installation

```bash
# Clone and install
git clone https://github.com/StevetheGreek97/HiReS.git
cd HiReS
pip install -e .
```

Install from PyPI (distribution name is `HiReSeg`):
```bash
pip install HiReSeg
```

The import name is `hires`.

**Requirements:** Python ≥ 3.10

**GPU inference:** Install PyTorch separately from the official PyTorch site, then pass `--device cuda:0` (or `--device 0`).

**Optional dependency:** `plotnine` (and `SkillMetrics`, a core dependency) for the trait-analysis plots in `hires.analysis`.

---

## Command-Line Interface (CLI)

Once installed, HiReS provides a terminal command called `hires`.

### General usage
```bash
hires <command> [options]
```

### Available Commands
| Command | Description |
|----------|-------------|
| `hires chunk` | Split images into overlapping chunks |
| `hires run` | Run the full segmentation pipeline (file or directory) |
| `hires plot` | Render segmentation overlays |

Run `hires <command> --help` for the authoritative option list.

---

### `hires chunk`
Split an image (or directory of images) into overlapping chunks.

```bash
hires chunk --source raw_image.tif --output chunks/ --chunk-size 1024 1024 --overlap 150
```

| Flag | Description | Default |
|------|-------------|---------|
| `-s`, `--source` | Path to a single image or directory | **required** |
| `-o`, `--output` | Output directory for chunks | `chunks` |
| `--chunk-size` | Chunk size in pixels: width height | `1024 1024` |
| `--overlap` | Overlap in pixels between chunks | `150` |

---

### `hires run`
Run the complete segmentation pipeline on one image or a folder.

```bash
hires run --source data/ --model models/DaphnAI.pt --output results/
```

**Pipeline steps:** chunk → predict → filter edges → unify → NMS → visualise → crops + CSV.

| Flag | Description | Default |
|------|-------------|---------|
| `-s`, `--source` | Image file or directory of images | **required** |
| `-m`, `--model` | Path to YOLO model (`.pt`) | `models/DaphnAI.pt` |
| `-o`, `--output` | Output directory | `results` |
| `-r`, `--recursive` | Recurse into subdirectories | `False` |
| `--conf` | Model confidence threshold | `0.5` |
| `--imgsz` | Inference image size | `1024` |
| `--device` | Compute device: `cpu`, `0`, `cuda:0`, … | `cpu` |
| `--chunk-size` | Chunk size (width height) | `1024 1024` |
| `--overlap` | Chunk overlap (pixels) | `150` |
| `--edge-threshold` | Border-touch filtering inset | `0.01` |
| `--iou-thresh` | IoU threshold for NMS | `0.7` |
| `--save-crops` | Save a masked crop per detection | `False` |
| `--dpi` | Scan resolution in DPI — enables physical measurements | `None` |
| `--unit` | Physical unit: `nm`, `um`, `mm`, `cm`, `m`, `inch` | `None` |
| `--debug` | Save intermediate debug artifacts under `<output>/<image>_debug/` | `False` |

---

### `hires plot`
Overlay YOLO-format segmentation polygons on the original image.

```bash
hires plot --source raw_image.tif --ann results/raw_image.txt --output results/ --model models/DaphnAI.pt
```

| Flag | Description | Default |
|------|-------------|---------|
| `-s`, `--source` | Path to the input image or directory | **required** |
| `-m`, `--model` | YOLO weights or a `data.yaml` file (class names only) | `models/DaphnAI.pt` |
| `-o`, `--output` | Output directory (writes `<image>_annotated.png` inside) | `results` |
| `--ann` | Annotation `.txt` file or directory of `.txt` files | `""` |
| `-r`, `--recursive` | Recurse into subdirectories | `False` |

If `--ann` is omitted, `hires plot` looks for `<output>/<image_stem>.txt`.

---

## Outputs

For each input image, `hires run` writes:
- `<output>/<image>.txt` → YOLO-style segmentation annotations (normalized polygon coords, optional confidence).
- `<output>/<image>_annotated.tif` → segmentation overlay image.
- `<output>/<image>_shapes.csv` → shape descriptors (area, perimeter, solidity, convexity, circularity, bbox and OBB width/length).
- `<output>/run_config.yaml` → the settings used for the run.
- `<output>/<image>_crops/` → per-object masked crops (only with `--save-crops`).

---

## Python API

Basic segmentation pipeline:

```python
from hires.models import Settings
from hires.pipeline.seg_pipeline import SegmentationPipeline

cfg = Settings(
    source="data/images/",
    model_path="models/DaphnAI.pt",
    output_dir="results/",
    conf=0.58,
    imgsz=1024,
    device="cpu",
    chunk_size=(1024, 1024),
    overlap=300,
    edge_threshold=0.01,
    iou_thresh=0.7,
    save_crops=True,
    recursive=False,
    debug=False,
)

SegmentationPipeline(cfg).run()
```

Chunking only:

```python
from hires.processing.chunker import ImageChunker

ImageChunker("data/images/").slice(
    save_folder="chunks/",
    chunk_size=(1024, 1024),
    overlap=150,
)
```

Plotting only:

```python
from hires.models import Settings
from hires.pipeline.chunk_pipeline import PlottingPipeline

cfg = Settings(
    source="data/images/",
    model_path="models/DaphnAI.pt",
    output_dir="results/",
    ann="results/",
)

PlottingPipeline(cfg).run()
```

See the [documentation](https://stevethegreek97.github.io/HiReS/) for the data
model (`Annotation`, `Collection`, `Album`) and evaluation APIs.

---

## Project Structure

```
hires/
├── models/            # Annotation, Collection, Album, Settings, parser, eval/
├── pipeline/          # BasePipeline, SegmentationPipeline, PlottingPipeline
├── processing/        # ImageChunker, AnnotationChunker, YOLOSegPredictor
├── operations/        # unify_collections (chunk → full-image transform)
├── viz/               # SegmentationPlotter, performance plots
├── analysis/          # GT-vs-prediction trait analysis (optional plotnine)
└── cli.py             # Command-line interface
```

---

## Dependencies

Core dependencies (from `pyproject.toml`):
- **numpy** ≥ 2.1
- **pandas** ≥ 2.2
- **matplotlib** ≥ 3.9
- **opencv-python** ≥ 4.10
- **shapely** ≥ 2.1
- **ultralytics** ≥ 8.3
- **tqdm** ≥ 4.66
- **pyyaml** ≥ 6.0
- **scipy** ≥ 1.14
- **SkillMetrics** ≥ 1.2

Optional:
- **torch** (GPU inference; install via the official PyTorch site)
- **plotnine** (trait-analysis plots in `hires.analysis`)

---

## Author

**Stylianos Mavrianos**
University of Hamburg
stylianosmavrianos@gmail.com

---

## License

Licensed under the **MIT License**.
See the `LICENSE` file for details.
