from __future__ import annotations


from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from HiReS import ImageChunker
from abc import ABC, abstractmethod
from pathlib import Path

from HiReS.source.config import Settings
from HiReS.source.utils.logger import create_logger, image_context

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class BasePipeline(ABC):
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg
        self.logger = create_logger()

    def _iter_images(self) -> list[Path]:
        src = Path(self.cfg.source)

        if src.is_file():
            return [src]

        if src.is_dir():
            if getattr(self.cfg, "recursive", False):
                files = [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS]
            else:
                files = [p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS]
            return sorted(files)

        return []

    def _get_output_dir(self, img: Path) -> Path:
        base = Path(self.cfg.output_dir)
        src = Path(self.cfg.source)

        if src.is_dir() and getattr(self.cfg, "recursive", False):
            return base / img.parent.relative_to(src)

        return base

    def _detect_mode(self, src: Path) -> str:
        if src.is_file():
            return "single file"
        if getattr(self.cfg, "recursive", False):
            return "recursive directory"
        return "directory"
    
    def _log_run_header(self, mode: str, n_images: int) -> None:
        self.logger.info(
            "%s | mode=%s | images=%d | debug=%s | output=%s",
            #self.label,
            mode,
            n_images,
            getattr(self.cfg, "debug", False),
            self.cfg.output_dir,
        )
    def run(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _run_single(self, image_path: Path, debug: bool = False) -> str | None:
        """Implement one-image processing."""
        raise NotImplementedError


import os
import tempfile
from contextlib import nullcontext
from pathlib import Path

from HiReS import ImageChunker, YOLOSegPredictor, SegmentationPlotter, AnnotationParser
from HiReS.source.anno.datatypes import AnnotationCollection
from HiReS.source.anno.ops import unify_collections
from HiReS.source.utils.logger import log_step
from HiReS.source.utils.fun import spinner


class SegmentationPipeline(BasePipeline):
    def run(self):
        src = Path(self.cfg.source)
        images = self._iter_images()

        if not images:
            self.logger.error(
                "No valid images found at: %s. If recursive processing is needed, enable '--recursive'.",
                src,
            )
            return

        mode = self._detect_mode(src)
        #self._log_run_header(mode, len(images))

        for i, img in enumerate(images, start=1):
            out_dir = self._get_output_dir(img)
            out_dir.mkdir(parents=True, exist_ok=True)

            old_out = self.cfg.output_dir
            self.cfg.output_dir = str(out_dir)

            try:
                #with image_context(self.logger, img, i, len(images)):
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

        base = os.environ.get("TMPDIR")
        if debug:
            debug_workdir = Path(self.cfg.output_dir) / "_temp" / image_stem
            debug_workdir.mkdir(parents=True, exist_ok=True)
            ctx = nullcontext(debug_workdir)
        else:
            ctx = tempfile.TemporaryDirectory(dir=base)

        with ctx as workdir:
            tmp = Path(workdir)
            tmp_chunks = tmp / "chunks"
            tmp_pred = tmp / "pred"
            tmp_filtered = tmp / "filtered"
            tmp_chunks.mkdir(parents=True, exist_ok=True)
            tmp_pred.mkdir(parents=True, exist_ok=True)
            tmp_filtered.mkdir(parents=True, exist_ok=True)

            with log_step("Chunking", self.logger):
                ImageChunker(str(image_path)).slice(
                    save_folder=str(tmp_chunks),
                    chunk_size=self.cfg.chunk_size,
                    overlap=self.cfg.overlap,
                )
                chunk_imgs = sorted(
                    p for p in tmp_chunks.iterdir() if p.suffix.lower() in IMG_EXTS
                )
                self.logger.info("Chunks: %d", len(chunk_imgs))

                if not chunk_imgs:
                    raise FileNotFoundError(
                        f"No chunk images were created in {tmp_chunks} for {image_path}."
                    )

            with log_step("Prediction", self.logger):
                with spinner("Predicting"):
                    YOLOSegPredictor(str(self.cfg.model_path), output_dir=str(tmp_pred)).predict(
                        image_dir=str(tmp_chunks),
                        conf=self.cfg.conf,
                        imgsz=self.cfg.imgsz,
                        device=self.cfg.device,
                    )

            pred_txts = sorted(tmp_pred.glob("*.txt"))

            with log_step("Filtering edge-touching polygons", self.logger):
                chunk_colls: dict[str, AnnotationCollection] = {}
                for txt in pred_txts:
                    anns = list(AnnotationParser(str(txt)).parse())
                    coll = AnnotationCollection(anns, collection_name=txt.stem)

                    filtered_coll = coll.remove_edge_cases(threshold=self.cfg.edge_threshold)
                    chunk_colls[txt.name] = filtered_coll

            with log_step("Unifying chunk annotations", self.logger):
                unified_coll = unify_collections(
                    chunk_collections=chunk_colls,
                    chunk_size=self.cfg.chunk_size,
                    full_img_path=str(image_path),
                )
                unified_coll = unified_coll.remove_edge_cases(threshold=self.cfg.edge_threshold)

            with log_step("Applying polygon NMS", self.logger):
                kept_coll = unified_coll.nms(
                    iou_threshold=self.cfg.iou_thresh,
                    class_aware=False,
                    return_indices=False,
                )
                final_txt = Path(self.cfg.output_dir) / f"{image_stem}.txt"
                kept_coll.write_annotations_to_txt(str(final_txt), include_conf=True)

            with log_step("Visualization", self.logger):
                out_img = Path(self.cfg.output_dir) / f"{image_stem}_annotated.tif"
                SegmentationPlotter(str(self.cfg.model_path)).plot_annotations(
                    str(image_path),
                    str(final_txt),
                    seg=True,
                    save=str(out_img),
                )

            with log_step("Shape descriptors & crops", self.logger):
                crops_dir = Path(self.cfg.output_dir) / f"{image_stem}_crops"
                crops = kept_coll.save_crops(
                    image=str(image_path),
                    out_dir=crops_dir,
                    use_mask=True,
                    file_prefix=image_stem,
                    ext="png",
                    denormalize=True,
                )

                df = kept_coll.shape_descriptors(crops=crops, image=str(image_path))
                shapes_csv = Path(self.cfg.output_dir) / f"{image_stem}_shapes.csv"
                df.to_csv(shapes_csv, index=False)

                self.logger.info("Saved %d crops → %s", len(crops), crops_dir)
                self.logger.info("Saved shape descriptors → %s", shapes_csv)

            self.logger.info("Done → %s", self.cfg.output_dir)
            return str(final_txt)



# ----------------------------
# Chunking
# ----------------------------
class ChunkingPipeline(BasePipeline):
    def _run_single(self, image: str | Path) -> str:
        img = Path(image)

        out_dir = self._get_output_dir(img)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_chunks = out_dir / f"{img.stem}_chunks"
        out_chunks.mkdir(parents=True, exist_ok=True)

        ImageChunker(str(img)).slice(
            save_folder=str(out_chunks),
            chunk_size=self.cfg.chunk_size,
            overlap=self.cfg.overlap,
        )

        return str(out_chunks)

    def run(self) -> None:
        src = Path(self.cfg.source)
        images = self._iter_images()

        if not images:
            self.logger.error(
                "No valid images found at: %s. If recursive processing is needed, enable '--recursive'.",
                src,
            )
            return

        self.logger.info("[CHUNK] images=%d output=%s", len(images), self.cfg.output_dir)

        for img in tqdm(images, desc="Chunking", unit="img"):
            try:
                self._run_single(img)
            except Exception:
                self.logger.exception("[CHUNK] FAIL %s", img)


# ----------------------------
# Plotting
# ----------------------------
class PlottingPipeline(BasePipeline):
    def _run_single(self, image: str | Path, ann_path: str | Path = None) -> str | None:
        img = Path(image)

        out_dir = self._get_output_dir(img)
        out_dir.mkdir(parents=True, exist_ok=True)
        if ann_path:
            ann = Path(ann_path)
        else:
            # If annotation path not provided, assume same name as image but .txt in the output dir
            ann = out_dir / f"{img.stem}.txt"
        if not ann.exists():
            self.logger.warning("[PLOT] missing annotation for %s", img.name)
            return None

        out_img = out_dir / f"{img.stem}_annotated.tif"

        SegmentationPlotter(str(self.cfg.model_path)).plot_annotations(
            str(img),
            str(ann),
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

        self.logger.info("[PLOT] images=%d output=%s", len(images), self.cfg.output_dir)

        for img in tqdm(images, desc="Plotting", unit="img"):
            try:
                self._run_single(img,self.cfg.ann)

            except Exception:
                self.logger.exception("[PLOT] FAIL %s", img)
