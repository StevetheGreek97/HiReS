from __future__ import annotations

import dataclasses
import datetime
import os
import shutil
import tempfile
from pathlib import Path

import yaml

from hires.models.collection import Collection
from hires.operations.ops import unify_collections
from hires.pipeline.base import BasePipeline, IMG_EXTS
from hires.pipeline.logger import log_step, spinner
from hires.processing.chunker import ImageChunker
from hires.processing.predictor import YOLOSegPredictor
from hires.viz.plotting import SegmentationPlotter


class SegmentationPipeline(BasePipeline):

    def _save_run_config(self, n_images: int) -> None:
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg = dataclasses.asdict(self.cfg)
        cfg["chunk_size"] = list(cfg["chunk_size"])  # tuple → list for clean yaml

        record = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "n_images": n_images,
            "settings": cfg,
        }

        log_path = out_dir / "run_config.yaml"
        with open(log_path, "w") as fh:
            yaml.dump(record, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
        self.logger.info("Run config saved → %s", log_path)

    def run(self):
        if isinstance(self.cfg.source, (list, tuple)):
            saved = self.cfg.source
            try:
                for item in saved:
                    if not Path(item).exists():
                        self.logger.warning("Source not found, skipping: %s", item)
                        continue
                    self.cfg.source = item
                    self.run()
            finally:
                self.cfg.source = saved
            return

        src = Path(self.cfg.source)
        images = self._iter_images()

        if not images:
            self.logger.error(
                "No valid images found at: %s. If recursive processing is needed, enable '--recursive'.",
                src,
            )
            return

        mode = self._detect_mode(src)
        self._log_run_header(mode, len(images))
        self._save_run_config(len(images))

        for i, img in enumerate(images, start=1):
            out_dir = self._get_output_dir(img)
            out_dir.mkdir(parents=True, exist_ok=True)

            old_out = self.cfg.output_dir
            self.cfg.output_dir = str(out_dir)

            try:
                self._run_single(img, debug=getattr(self.cfg, "debug", False))
            except Exception:
                self.logger.exception("✖ FAILED [%d/%d] %s", i, len(images), img)
            finally:
                self.cfg.output_dir = old_out

    def _run_single(self, image_path: Path, debug: bool = False) -> str:
        self.logger.info("Image: %s | Model: %s", image_path, self.cfg.model_path)
        self.logger.info(
            "Config: conf=%.3f imgsz=%d device=%s chunk=%s overlap=%d edge_thr=%.4g iou_thr=%.3f",
            self.cfg.conf,
            self.cfg.imgsz,
            self.cfg.device,
            self.cfg.chunk_size,
            self.cfg.overlap,
            self.cfg.edge_threshold,
            self.cfg.iou_thresh,
        )

        image_stem = image_path.stem
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

        base_tmp = os.environ.get("TMPDIR")
        debug_dir = None
        debug_chunks_dir = None
        debug_pred_dir = None
        debug_filtered_dir = None
        debug_filtered_txt_dir = None
        debug_plotter = None
        if debug:
            debug_dir = Path(self.cfg.output_dir) / f"{image_stem}_debug"
            debug_chunks_dir = debug_dir / "chunks"
            debug_pred_dir = debug_dir / "pred"
            debug_filtered_dir = debug_dir / "filtered"
            debug_filtered_txt_dir = debug_dir / "filtered_txt"
            for path in (
                debug_dir,
                debug_chunks_dir,
                debug_pred_dir,
                debug_filtered_dir,
                debug_filtered_txt_dir,
            ):
                path.mkdir(parents=True, exist_ok=True)
            debug_plotter = SegmentationPlotter(str(self.cfg.model_path))
            self.logger.info("Debug artifacts -> %s", debug_dir)

        with tempfile.TemporaryDirectory(dir=base_tmp) as workdir:
            tmp = Path(workdir)
            tmp_chunks = tmp / "chunks"
            tmp_pred = tmp / "pred"
            tmp_chunks.mkdir(parents=True, exist_ok=True)
            tmp_pred.mkdir(parents=True, exist_ok=True)

            with log_step("Chunking", self.logger):
                ImageChunker(str(image_path)).slice(
                    save_folder=str(tmp_chunks),
                    chunk_size=self.cfg.chunk_size,
                    overlap=self.cfg.overlap,
                )
                chunk_imgs = sorted(
                    p for p in tmp_chunks.iterdir() if p.suffix.lower() in IMG_EXTS
                )
                chunk_img_by_stem = {p.stem: p for p in chunk_imgs}
                self.logger.info("Chunks: %d", len(chunk_imgs))

                if not chunk_imgs:
                    raise FileNotFoundError(
                        f"No chunk images were created in {tmp_chunks} for {image_path}."
                    )

                if debug and debug_chunks_dir is not None:
                    for chunk_img in chunk_imgs:
                        shutil.copy2(chunk_img, debug_chunks_dir / chunk_img.name)

            with log_step("Prediction", self.logger):
                with spinner(f"Predicting {len(chunk_imgs)} chunks…"):
                    YOLOSegPredictor(
                        str(self.cfg.model_path), output_dir=str(tmp_pred)
                    ).predict(
                        image_dir=str(tmp_chunks),
                        conf=self.cfg.conf,
                        imgsz=self.cfg.imgsz,
                        device=self.cfg.device,
                    )

            pred_txts = sorted(tmp_pred.glob("*.txt"))

            if debug and debug_pred_dir is not None and debug_plotter is not None:
                for txt in pred_txts:
                    chunk_img = chunk_img_by_stem.get(txt.stem)
                    if chunk_img is None:
                        continue
                    debug_plotter.plot_annotations(
                        str(chunk_img),
                        str(txt),
                        save=str(debug_pred_dir / f"{txt.stem}_pred.png"),
                    )

            edge_thr = self.cfg.edge_threshold

            with log_step("Filtering edge-touching polygons", self.logger):
                chunk_colls: dict[str, Collection] = {}
                for txt in pred_txts:
                    coll = Collection.read_txt(txt, collection_name=txt.stem)
                    filtered_coll = coll.filter(
                        predicate=lambda ann, t=edge_thr: ann.is_inside_unit_box(threshold=t)
                    )
                    chunk_colls[txt.name] = filtered_coll

                    if (
                        debug
                        and debug_filtered_dir is not None
                        and debug_filtered_txt_dir is not None
                        and debug_plotter is not None
                    ):
                        filtered_txt = debug_filtered_txt_dir / f"{txt.stem}_filtered.txt"
                        filtered_coll.to_txt(str(filtered_txt), include_conf=True)
                        chunk_img = chunk_img_by_stem.get(txt.stem)
                        if chunk_img is not None:
                            debug_plotter.plot_annotations(
                                str(chunk_img),
                                str(filtered_txt),
                                save=str(debug_filtered_dir / f"{txt.stem}_filtered.png"),
                            )

            with log_step("Unifying chunk annotations", self.logger):
                unified_coll = unify_collections(
                    chunk_collections=chunk_colls,
                    chunk_size=self.cfg.chunk_size,
                    full_img_path=str(image_path),
                )
                unified_coll = unified_coll.filter(
                    predicate=lambda ann, t=edge_thr: ann.is_inside_unit_box(threshold=t)
                )

                if debug and debug_dir is not None and debug_plotter is not None:
                    unified_txt = debug_dir / f"{image_stem}_unified.txt"
                    unified_coll.to_txt(str(unified_txt), include_conf=True)
                    debug_plotter.plot_annotations(
                        str(image_path),
                        str(unified_txt),
                        seg=True,
                        save=str(debug_dir / f"{image_stem}_unified.png"),
                    )

            with log_step("Applying polygon NMS", self.logger):
                kept_coll = unified_coll.nms(
                    iou_threshold=self.cfg.iou_thresh,
                    class_aware=False,
                )
                final_txt = Path(self.cfg.output_dir) / f"{image_stem}.txt"
                kept_coll.to_txt(str(final_txt), include_conf=True)
                self.logger.info(
                    "Detections: %d | class_counts: %s", len(kept_coll), kept_coll.class_counts
                )

            with log_step("Visualization", self.logger):
                out_img = Path(self.cfg.output_dir) / f"{image_stem}_annotated.tif"
                SegmentationPlotter(str(self.cfg.model_path)).plot_annotations(
                    str(image_path),
                    str(final_txt),
                    seg=True,
                    save=str(out_img),
                )

            with log_step("Shape descriptors & crops", self.logger):
                kept_coll.image_path = str(image_path)

                if self.cfg.dpi is not None or self.cfg.unit is not None:
                    kept_coll.set_scale(dpi=self.cfg.dpi, unit=self.cfg.unit)

                if getattr(self.cfg, "save_crops", True):
                    crops_dir = Path(self.cfg.output_dir) / f"{image_stem}_crops"
                    kept_coll.save_crops(
                        out_dir=crops_dir,
                        use_mask=True,
                        file_prefix=image_stem,
                        ext="png",
                    )
                    self.logger.info("Saved crops → %s", crops_dir)

                shapes_csv = Path(self.cfg.output_dir) / f"{image_stem}_shapes.csv"
                kept_coll.to_csv(shapes_csv)
                self.logger.info("Saved shape descriptors → %s", shapes_csv)

        self.logger.info("Done → %s", self.cfg.output_dir)
        return str(final_txt)
