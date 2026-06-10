# Quickstart

## 1. Install HiReS

```bash
pip install HiReSeg
```

## 2. Run the segmentation pipeline

```bash
hires run --source image.tif --model model.pt --output results/
```

This produces:
- `results/image.txt` — YOLO polygon annotations
- `results/image_annotated.tif` — overlay with drawn polygons
- `results/image_shapes.csv` — shape descriptor table
- `results/run_config.yaml` — the exact settings used for the run
- `results/image_crops/` — per-object image crops (only with `--save-crops`)

## 3. Inspect results

```bash
hires plot --source image.tif --ann results/image.txt --output results/ --model model.pt
```

Writes a rendered overlay to `results/image_annotated.png`.

---

## Python API quickstart

```python
from hires.models import Settings
from hires.pipeline.seg_pipeline import SegmentationPipeline

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

- Use `--device cuda:0` (or `--device 0`) if you have a CUDA GPU — inference is significantly faster.
- Increase `--overlap` (default 150 px) if you see missed detections at tile boundaries.
- Add `--save-crops` to export a masked crop image for every detection.
- Pass `--dpi` and `--unit` (e.g. `--dpi 1200 --unit um`) to report descriptors in physical units.
- Use `--debug` to save intermediate chunk-level annotations for troubleshooting.
- To evaluate predictions against manual annotations, use the `hires.models.eval` API — see [Evaluation](library/evaluation.md).
