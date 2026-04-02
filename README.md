# HiReS
**High-Resolution Image Segmentation and Analysis Pipeline**

HiReS is a modular Python package and command-line tool for automated image segmentation and analysis.
It targets high-resolution microscopy or biological datasets and combines **YOLO-based instance segmentation**
with **geometry-aware postprocessing** (Shapely).

HiReS makes it easy to:
- Split large `.tif`, `.tiff`, `.png`, `.jpg` images into overlapping chunks.
- Run YOLO segmentation on each chunk.
- Merge predictions into full-image coordinates.
- Filter edge-touching polygons and apply IoU-based NMS.
- Generate overlays, object crops, and shape-descriptor tables.

---

## Key Features
- Chunking with overlap for large images.
- YOLO segmentation per chunk.
- Polygon filtering, unification, and NMS.
- Overlay rendering for quick QA.
- Per-object crops and shape descriptors (CSV).
- CLI, Python API, and Streamlit UI (optional).

---

## Installation

```bash
# Clone and install
git clone https://github.com/StevetheGreek97/HiReS.git
cd HiReS
pip install -e .
```

Install from PyPI (package name is `HiReSeg`):
```bash
pip install HiReSeg
```

Import name is `HiReS`.

**Requirements:** Python ≥ 3.10

**GPU inference:** Install PyTorch separately from the official PyTorch site.

**Extra runtime deps used by the code:**
- `opencv-python` (chunking and plotting)
- `streamlit` (UI)

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

---

### `hires chunk`
Split an image (or directory of images) into evenly sized chunks.

```bash
hires chunk --source raw_image.tif --out chunks/ --chunk-size 1024 1024 --overlap 150
```

**Arguments:**
| Flag | Description | Default |
|------|-------------|---------|
| `--source` | Path to a single image or directory | — |
| `--out` | Output directory for chunks | — |
| `--chunk-size` | Chunk size in pixels: width height | `1024 1024` |
| `--overlap` | Overlap in pixels between chunks | `150` |
| `--recursive` | Recurse into subdirectories | `False` |

Note: `hires chunk` currently ignores `--chunk-size` due to a CLI wiring bug and
uses the default `Settings.chunk_size` (1024x1024). Use the Python API with
`Settings(chunk_size=...)` to override.

---

### `hires run`
Run the complete segmentation pipeline on one image or a folder.

```bash
hires run --source data/ --model models/DaphnAI.pt --out results/
```

**Pipeline steps:**
1. Chunk input images
2. Predict segmentations using YOLO
3. Filter polygons touching image edges
4. Merge chunks into full-image coordinates
5. Apply IoU-based polygon NMS
6. Save final annotations + overlay
7. Save crops and shape descriptors

**Arguments:**
| Flag | Description | Default |
|------|-------------|---------|
| `--source` | Image file or directory of images | — |
| `--model` | Path to YOLO model (.pt) | — |
| `--out` | Output directory | — |
| `--conf` | Model confidence threshold | `0.5` |
| `--imgsz` | Inference image size | `1024` |
| `--device` | Compute device: `cpu`, `cuda:0`, or `mps` | `cpu` |
| `--chunk-size` | Chunk size (width height) | `1024 1024` |
| `--overlap` | Chunk overlap (pixels) | `150` |
| `--edge-thr` | Border-touch filtering threshold | `1e-2` |
| `--iou-thr` | IoU threshold for NMS | `0.7` |
| `--recursive` | Recurse into subdirectories | `False` |
| `--debug` | Parsed by CLI but not wired; set `Settings(debug=True)` in Python API | `False` |
| `--workers` | Parsed by CLI but not used; directory processing is sequential | `1` |

---

### `hires plot`
Overlay YOLO-format segmentation polygons on the original image.

```bash
hires plot --image raw_image.tif --ann results/raw_image.txt --out results/ --model models/DaphnAI.pt
```

**Arguments:**
| Flag | Description |
|------|-------------|
| `--image` | Path to the input image |
| `--ann` | YOLO-format annotation file |
| `--out` | Output directory (writes `<image>_annotated.tif` inside) |
| `--model` | Required: YOLO weights or a `data.yaml` file used for class names |

---

## Outputs

For each input image, HiReS writes:
- `<out>/<image>.txt` → YOLO-style segmentation annotations (normalized polygon coords, optional confidence).
- `<out>/<image>_annotated.tif` → segmentation overlay image.
- `<out>/<image>_crops/` → per-object crops with mask.
- `<out>/<image>_shapes.csv` → shape descriptors (area, perimeter, solidity, circularity, OBB width/height, etc.).

---

## Python API

Basic segmentation pipeline:

```python
from HiReS import Settings, SegmentationPipeline

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
    recursive=False,
    debug=False,
)

SegmentationPipeline(cfg).run()
```

Chunking only:

```python
from HiReS import Settings, ChunkingPipeline

cfg = Settings(
    source="data/images/",
    output_dir="chunks/",
    chunk_size=(1024, 1024),
    overlap=150,
    recursive=True,
)

ChunkingPipeline(cfg).run()
```

Plotting only:

```python
from HiReS import Settings, PlottingPipeline

cfg = Settings(
    source="data/images/",
    model_path="models/DaphnAI.pt",
    output_dir="results/",
    ann="results/example.txt",
)

PlottingPipeline(cfg).run()
```

---

## Streamlit UI (Optional)

```bash
pip install streamlit
streamlit run HiReS/ui/Welcome.py
```

Note: the Streamlit pages currently import `Pipeline` from `HiReS/source/pipeline.py`,
but that module only defines `SegmentationPipeline`, `ChunkingPipeline`, and
`PlottingPipeline`. The UI may need a small update to run as-is.

---

## Project Structure

```
HiReS/
├── source/
│   ├── anno/              # Annotation parsing, filtering, NMS
│   ├── ios/               # Chunking, plotting, writer, YOLO inference
│   ├── utils/             # Logging + CLI helpers
│   ├── config.py          # Settings dataclass
│   ├── pipeline.py        # Segmentation, chunking, plotting pipelines
│   └── cli.py             # Command-line interface
├── ui/                    # Streamlit UI
└── __init__.py            # Top-level exports
```

---

## Dependencies

Core dependencies (from `pyproject.toml`):
- **ultralytics** ≥ 8.0.0
- **shapely** ≥ 2.0.0
- **Pillow** ≥ 10.0.0
- **numpy** ≥ 1.25.0
- **matplotlib** ≥ 3.8.0
- **tqdm** ≥ 4.66.0
- **pandas** ≥ 2.2.2

Optional (for GPU):
- **torch** (install via the official PyTorch site)

---

## Author

**Stylianos Mavrianos**
University of Hamburg
stylianosmavrianos@gmail.com

---

## License

Licensed under the **MIT License**.
See the `LICENSE` file for details.
