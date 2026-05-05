# Pipeline Architecture

## Overview

HiReS provides three pipelines, all inheriting from `BasePipeline`:

```
BasePipeline
├── SegmentationPipeline   — full end-to-end workflow
├── ChunkingPipeline       — tiling only
└── PlottingPipeline       — annotation rendering only
```

---

## SegmentationPipeline

The main pipeline runs these steps for each input image:

### 1. Chunking

`ImageChunker` splits the image into overlapping tiles of size `chunk_size` with `overlap` pixels of padding between tiles. The overlap prevents missed detections at tile boundaries.

```
[full image] → [tile_0, tile_1, ..., tile_N]
```

### 2. YOLO Prediction

`YOLOSegPredictor` runs YOLO instance segmentation on each tile. Polygon coordinates are returned in normalized form (0–1 relative to the tile).

### 3. Edge Filtering

Polygons whose perimeter touches a tile edge beyond `edge_threshold` are discarded. This removes partial detections that were cut by the tile boundary and would otherwise appear as artifacts after merging.

### 4. Coordinate Unification

`unify_collections()` transforms each tile's normalized polygon coordinates into pixel coordinates in the full image's reference frame using affine offsets derived from the tile grid.

```
tile_polygon (normalized) → full_image_polygon (pixels)
```

### 5. Non-Maximum Suppression

IoU-based NMS is applied across all polygons in the merged full-image collection. Overlapping polygons (IoU > `iou_thresh`) are reduced to the one with the highest confidence score.

### 6. Output Generation

For each image the pipeline writes:

- YOLO-format `.txt` annotation file
- Annotated overlay `.tif`
- Per-object crop images (if `save_crops=True`)
- Shape descriptor CSV

See [Outputs](outputs.md) for details.

---

## ChunkingPipeline

Runs only step 1 (chunking) and saves the tiles. Useful for:
- Preprocessing large datasets before inference
- Inspecting tile coverage and overlap
- Generating training data patches

---

## PlottingPipeline

Loads an existing YOLO annotation file and renders polygons on the source image. Used for QA and visualization without re-running inference.

---

## Debug mode

When `debug=True`, the pipeline saves intermediate per-chunk annotation files to `<out>/<image>_debug/`. This is useful for diagnosing edge filtering or NMS behaviour.

---

## Module map

| Module | Class / Function | Role |
|--------|-----------------|------|
| `hires/pipeline/base.py` | `BasePipeline` | Image iteration, mode detection, logging |
| `hires/pipeline/seg_pipeline.py` | `SegmentationPipeline` | Orchestrates all steps |
| `hires/pipeline/chunk_pipeline.py` | `ChunkingPipeline`, `PlottingPipeline` | Standalone tile and plot pipelines |
| `hires/processing/chunker.py` | `ImageChunker` | Overlapping tile splitting |
| `hires/processing/predictor.py` | `YOLOSegPredictor` | YOLO inference wrapper |
| `hires/operations/ops.py` | `unify_collections()` | Coordinate transform from tile to full image |
| `hires/models/eval/iou.py` | `IoUMatrix` | Pairwise IoU computation |
| `hires/viz/plotting.py` | `SegmentationPlotter` | Polygon overlay rendering |
| `hires/analysis/analyis.py` | — | Shape descriptor extraction |
