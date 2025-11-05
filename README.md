# HiReS  
**High-Resolution Image Segmentation and Analysis Pipeline**

HiReS is a modular Python package and command-line tool for **automated image segmentation and analysis**.  
It was designed for high-resolution biological or microscopy datasets, combining **YOLO-based segmentation** with **geometry-aware postprocessing** (via Shapely).

HiReS makes it easy to:
- Split large `.tif` or `.png` images into manageable Chunks  
- Run instance segmentation on each Chunk  
- Merge predictions seamlessly into global coordinates  
- Filter and unify polygons  
- Generate high-quality annotation overlays  

---

## Key Features
- **Chunking:** Divide large images into smaller overlapping Chunks for model inference  
- **Segmentation:** Use YOLO-based instance segmentation on image chunks  
- **Filtering:** Automatically remove edge-touching or invalid polygons  
- **Merging:** Unify chunk-level detections into full-resolution space  
- **NMS:** Apply IoU-based Non-Maximum Suppression on polygons  
- **Visualization:** Create clear overlays showing detected regions  
- **Parallel Execution:** Run multiple images simultaneously  
- **Cross-platform:** Works on Linux, macOS, and Windows  

---

##  Installation

```bash
# Clone and install
git clone https://github.com/StevetheGreek97/HiReS.git
cd HiReS
pip install -e .
```

or install directly from PyPI:
```bash
pip install hires
```

> **Requirements:** Python ≥ 3.10  
> HiReS automatically installs Ultralytics, Shapely, Pillow, NumPy, and Matplotlib.  
> For GPU inference, install PyTorch separately from [pytorch.org](https://pytorch.org/get-started/locally/).

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
| `hires chunk` | Split a large image into smaller Chunks |
| `hires run` | Run the full segmentation pipeline (single image or directory) |
| `hires plot` | Visualize segmentation overlays |

---

### `hires chunk`
Split an image into evenly sized Chunks for segmentation.

```bash
hires chunk --image raw_image.tif --out chunks/ --chunk-size 1024 1024 --overlap 150
```

**Arguments:**
| Flag | Description | Default |
|------|--------------|----------|
| `--image` | Path to a single image file | — |
| `--out` | Output directory for Chunks | — |
| `--chunk-size` | Chunk size in pixels: width height | `1024 1024` |
| `--overlap` | Overlap in pixels between Chunks | `150` |

---

### `hires run`
Run the complete segmentation pipeline on one image or a folder of images.  
This automatically performs chunking, segmentation, filtering, merging, NMS, and visualization.

```bash
hires run --image data/ --model models/DaphnAI.pt --out results/ --workers 4
```

**Pipeline steps:**
1. Chunk input images  
2. Predict segmentations using YOLO  
3. Filter polygons touching image edges  
4. Merge chunks into full-image coordinates  
5. Apply IoU-based polygon NMS  
6. Save final annotation file and visualization overlay  

**Arguments:**
| Flag | Description | Default |
|------|--------------|----------|
| `--image` | Image file or directory of images | — |
| `--model` | Path to YOLO model (.pt) | — |
| `--out` | Output directory | — |
| `--conf` | Model confidence threshold | `0.5` |
| `--imgsz` | Inference image size | `1024` |
| `--device` | Compute device: `cpu`, `cuda:0`, or `mps` | `cpu` |
| `--chunk-size` | Chunk size (width height) | `1024 1024` |
| `--overlap` | Chunk overlap (pixels) | `150` |
| `--edge-thr` | Border-touch filtering threshold | `1e-2` |
| `--iou-thr` | IoU threshold for NMS | `0.7` |
| `--workers` | Number of parallel workers (for directories) | `1` |

**Outputs:**
- `results/<image>.txt` → YOLO-style segmentation annotations  
- `results/<image>_annotated.tif` → segmentation overlay image  

---

### `hires plot`
Overlay YOLO-format segmentation polygons on the original image.

```bash
hires plot --image raw_image.tif --ann results/raw_image.txt --out overlay.png
```

**Arguments:**
| Flag | Description |
|------|--------------|
| `--image` | Path to the input image |
| `--ann` | YOLO-format annotation file |
| `--out` | Path to save the rendered overlay |
| `--model` | (Optional) model name/path for color consistency |

---

## Python API Example

If you prefer working from Python, HiReS can be used programmatically:

```python
from HiReS.config import Settings
from HiReS.pipeline import Pipeline

cfg = Settings(
    conf=0.58,
    imgsz=1024,
    device="cpu",
    chunk_size=(1024, 1024),
    overlap=300,
    edge_threshold=0.01,
    iou_thresh=0.7
)

Pipeline(cfg).run(
    input_path="data/images/",
    model_path="models/DaphnAI.pt",
    output_dir="results/",
    workers=4
)
```

---

## Project Structure

```
HiReS/
├── anno/              # Annotation parsing, filtering, NMS
├── ios/               # Chunking, plotting, writer, and YOLO inference
├── config.py          # Dataclass for pipeline settings
├── pipeline.py        # Main pipeline entry point
├── cli.py             # Command-line interface
└── tests/             # Example notebooks
```

---

## Dependencies

HiReS depends on:
- **ultralytics** ≥ 8.0.0  
- **shapely** ≥ 2.0.0  
- **Pillow** ≥ 10.0.0  
- **numpy** ≥ 1.25.0  
- **matplotlib** ≥ 3.8.0  

Optional (for GPU):
- **torch** (install via [PyTorch.org](https://pytorch.org/get-started/locally/))

---

## Author

**Stylianos Mavrianos**  
University of Hamburg  
[stylianosmavrianos@gmail.com](mailto:stylianosmavrianos@gmail.com)

---

## License

Licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---
