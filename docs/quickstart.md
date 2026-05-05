# Quickstart

## 1. Install HiReS

```bash
pip install HiReSeg
```

## 2. Run the segmentation pipeline

```bash
hires run --source image.tif --model model.pt --out results/
```

This produces:
- `results/image.txt` — YOLO polygon annotations
- `results/image_annotated.tif` — overlay with drawn polygons
- `results/image_crops/` — per-object image crops
- `results/image_shapes.csv` — shape descriptor table

## 3. Inspect results

```bash
hires plot --image image.tif --ann results/image.txt --out results/ --model model.pt
```

Opens a rendered overlay saved to `results/image_annotated.tif`.

---

## Python API quickstart

```python
from hires import Settings, SegmentationPipeline

cfg = Settings(
    source="image.tif",
    model_path="model.pt",
    output_dir="results/",
    conf=0.5,
    device="cpu",
)

SegmentationPipeline(cfg).run()
```

---

## Tips

- Use `--device cuda:0` if you have a CUDA GPU — inference is significantly faster.
- Increase `--overlap` (default 150 px) if you see missed detections at tile boundaries.
- Use `--debug` to save intermediate chunk-level annotations for troubleshooting.
- Run `hires compare` after a manual annotation session to get TP/FP/FN metrics.
