# HiReS Documentation

**High-Resolution Image Segmentation and Analysis Pipeline**

HiReS is a modular Python package and CLI tool for automated segmentation of large microscopy or biological images. It combines YOLO-based instance segmentation with geometry-aware post-processing (Shapely) to handle images that exceed typical GPU memory limits.

---

## How it works

Large images are split into overlapping tiles, each tile is run through YOLO, and the resulting polygons are then merged back into full-image coordinates with edge filtering and IoU-based NMS applied.

```
Image → Chunking → YOLO Prediction → Edge Filter → Unify → NMS → Outputs
```

---

## Documentation

| Page | Description |
|------|-------------|
| [Installation](installation.md) | Setup and requirements |
| [Quickstart](quickstart.md) | Run your first segmentation in minutes |
| [CLI Reference](cli.md) | All CLI commands and flags |
| [Python API](api.md) | Using HiReS from Python |
| [Configuration](configuration.md) | All `Settings` parameters |
| [Pipeline Architecture](pipeline.md) | How the pipeline works internally |
| [Outputs](outputs.md) | Output files and formats |
| [Data Models](models.md) | Annotation, Collection, and Album classes with examples |
| [Evaluation](evaluation.md) | Pair, Bundle, IoUMatrix, CollectionMatchMaker, AlbumMatchMaker |

---

## Key features

- Overlapping tile chunking for arbitrarily large images
- YOLO instance segmentation per chunk
- Polygon edge filtering, coordinate unification, and IoU-based NMS
- Overlay rendering for quick QA
- Per-object crops and shape descriptor CSV (area, perimeter, circularity, solidity, OBB dimensions)
- Physical unit support (DPI-aware measurements in nm, μm, mm, etc.)
- CLI, Python API, and optional Streamlit UI
