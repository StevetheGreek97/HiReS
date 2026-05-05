from __future__ import annotations

from pathlib import Path
from tqdm import tqdm
from hires.viz import SegmentationPlotter
from hires.pipeline.base import BasePipeline


class PlottingPipeline(BasePipeline):
    def _resolve_ann(self, img: Path, ann_dir_map: dict[str, Path] | None) -> Path | None:
        """Return the annotation path for *img*, or None if not found."""
        ann_cfg = self.cfg.ann
        if ann_dir_map is not None:
            return ann_dir_map.get(img.stem)
        if ann_cfg:
            p = Path(ann_cfg)
            return p if p.exists() else None
        # fallback: look for <output_dir>/<stem>.txt
        fallback = Path(self.cfg.output_dir) / f"{img.stem}.txt"
        return fallback if fallback.exists() else None

    def _run_single(self, image: str | Path, ann_path: Path | None) -> str | None:
        img = Path(image)
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if ann_path is None or not ann_path.exists():
            self.logger.warning("[PLOT] missing annotation for %s", img.name)
            return None

        out_img = out_dir / f"{img.stem}_annotated.png"
        SegmentationPlotter(str(self.cfg.model_path)).plot_annotations(
            str(img),
            str(ann_path),
            seg=True,
            save=str(out_img),
        )
        return str(out_img)

    def run(self) -> None:
        src = Path(self.cfg.source)
        images = self._iter_images()

        if not images:
            self.logger.error(
                "No valid images found at: %s. If recursive processing is needed, enable '--recursive'.",
                src,
            )
            return

        # Build stem→path map when --ann points to a directory
        ann_dir_map: dict[str, Path] | None = None
        ann_cfg = self.cfg.ann
        if ann_cfg and Path(ann_cfg).is_dir():
            ann_dir = Path(ann_cfg)
            ann_dir_map = {p.stem: p for p in ann_dir.glob("*.txt")}
            self.logger.info("[PLOT] annotation dir=%s  (%d .txt files)", ann_dir, len(ann_dir_map))

        self.logger.info("[PLOT] images=%d output=%s", len(images), self.cfg.output_dir)

        for img in tqdm(images, desc="Plotting", unit="img"):
            try:
                ann_path = self._resolve_ann(Path(img), ann_dir_map)
                self._run_single(img, ann_path)
            except Exception:
                self.logger.exception("[PLOT] FAIL %s", img)
