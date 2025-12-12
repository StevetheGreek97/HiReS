# HiReS/pipeline.py
import time
import tempfile
import shutil
from contextlib import contextmanager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import logging
import sys
import queue
from logging.handlers import QueueHandler, QueueListener

from HiReS.source.config import Settings
from HiReS.source.ios.chunker import ImageChunker
from HiReS.source.ios.yolo_predictor import YOLOSegPredictor
from HiReS.source.anno.parser import AnnotationParser
from HiReS.source.anno.datatypes import AnnotationCollection
from HiReS.source.anno.ops import unify_collections  # adjust path/name if needed
from HiReS.source.ios.plotting import SegmentationPlotter


def setup_logging(level=logging.INFO, log_file=None):
    log_q = queue.Queue(-1)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    qh = QueueHandler(log_q)
    root.addHandler(qh)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    handlers = [console]
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        handlers.append(fh)

    listener = QueueListener(log_q, *handlers, respect_handler_level=True)
    listener.start()
    return listener


@contextmanager
def log_step(name: str, logger: logging.Logger):
    t0 = time.perf_counter()
    logger.info(">> %s: start", name)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        logger.info("[OK] %s: done in %.2f s", name, dt)


setup_logging()

# HiReS/pipeline.py
import time
import tempfile
import shutil
from contextlib import contextmanager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import logging
import sys
import queue
from logging.handlers import QueueHandler, QueueListener

from HiReS.source.config import Settings
from HiReS.source.ios.chunker import ImageChunker
from HiReS.source.ios.yolo_predictor import YOLOSegPredictor
from HiReS.source.anno.parser import AnnotationParser
from HiReS.source.anno.datatypes import AnnotationCollection
from HiReS.source.anno.ops import unify_collections  # adjust path/name if needed
from HiReS.source.ios.plotting import SegmentationPlotter


def setup_logging(level=logging.INFO, log_file=None):
    log_q = queue.Queue(-1)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    qh = QueueHandler(log_q)
    root.addHandler(qh)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    handlers = [console]
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        handlers.append(fh)

    listener = QueueListener(log_q, *handlers, respect_handler_level=True)
    listener.start()
    return listener


@contextmanager
def log_step(name: str, logger: logging.Logger):
    t0 = time.perf_counter()
    logger.info(">> %s: start", name)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        logger.info("[OK] %s: done in %.2f s", name, dt)


setup_logging()


class Pipeline:
    def __init__(self, cfg: Settings, logger: logging.Logger | None = None):
        self.cfg = cfg
        self.log = logger or logging.getLogger("HiReS.Pipeline")

    def run(
        self,
        *,
        workers: int = 1,
        patterns: tuple[str, ...] = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"),
        debug: bool = False,
        progress_cb=None,   # <-- NEW
    ) -> list[str] | str:
        """
        Smart entry:
          - If input_path is a FILE → process single image, return final .txt path (str).
          - If input_path is a DIR  → process all images inside, return list of final .txt paths.
          - progress_cb(frac, message) is called with frac in [0, 1].
        """
        ipath = Path(self.cfg.source)

        # -----------------------------
        # SINGLE IMAGE MODE
        # -----------------------------
        if ipath.is_file():
            self.log.info("Detected single image: %s", ipath)

            if progress_cb:
                progress_cb(0.0, f"Starting {ipath.name}")

            result_txt = self._run_single(
                ipath,
                Path(self.cfg.model_path),
                Path(self.cfg.output_dir),
                self.log,
                debug=debug,
                progress_cb=progress_cb
            )

            if progress_cb:
                progress_cb(1.0, f"Finished {ipath.name}")

            return result_txt

        # -----------------------------
        # DIRECTORY MODE
        # -----------------------------
        if not ipath.is_dir():
            raise FileNotFoundError(f"Input path not found: {ipath}")

        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        images: list[Path] = []
        for pat in patterns:
            images.extend(sorted(ipath.glob(pat)))

        self.log.info(
            "Detected directory: %s (%d images, patterns=%s)",
            ipath,
            len(images),
            patterns,
        )

        if not images:
            self.log.warning("No images found in %s", ipath)
            if progress_cb:
                progress_cb(1.0, "No images found")
            return []

        total = len(images)
        completed = 0
        results: list[str] = []
        workers = max(1, int(workers))
        self.log.info("Processing in parallel with workers=%d", workers)

        if progress_cb:
            progress_cb(0.0, f"Queued {total} images")

        def submit(img: Path):
            child = logging.getLogger(f"HiReS.Pipeline[{img.name}]")
            try:
                return self._run_single(
                    img,
                    Path(self.cfg.model_path),
                    out_dir,
                    child,
                    debug=debug,
                )
            except Exception as e:
                child.exception("Failed: %s", e)
                return None

        # ---- workers == 1 → simple for-loop ----
        if workers == 1:
            for img in images:
                r = submit(img)
                if r:
                    results.append(r)

                completed += 1
                if progress_cb:
                    frac = completed / total
                    progress_cb(frac, f"[{completed}/{total}]")

        # ---- workers > 1 → ThreadPoolExecutor + as_completed ----
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(submit, img): img for img in images}
                for fut in as_completed(futs):
                    img = futs[fut]
                    r = fut.result()
                    if r:
                        results.append(r)

                    completed += 1
                    if progress_cb:
                        frac = completed / total
                        progress_cb(frac, f"[{completed}/{total}]")

        self.log.info("Completed %d/%d images", len(results), len(images))

        if progress_cb:
            progress_cb(1.0, "All images processed")

        return results


    def _run_single(
        self,
        image_path: Path,
        model_path: Path,
        output_dir: Path,
        logger: logging.Logger,
        debug: bool = False,
        progress_cb=None,
    ) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        image_stem = image_path.stem
        total_steps = 6

        def update(step_idx: int, message: str):
            """
            step_idx: 0..total_steps
            """
            if progress_cb:
                frac = max(0.0, min(1.0, step_idx / total_steps))
                progress_cb(frac, message)

        logger.info("Image: %s | Model: %s", image_path, model_path)
        logger.info(
            "Config: conf=%.3f imgsz=%d device=%s chunk=%s overlap=%d edge_thr=%.4g iou_thr=%.3f",
            self.cfg.conf,
            self.cfg.imgsz,
            self.cfg.device,
            self.cfg.chunk_size,
            self.cfg.overlap,
            self.cfg.edge_threshold,
            self.cfg.iou_thresh,
        )

        # Optional debug dirs
        if debug:
            debug_dir = output_dir / f"{image_stem}_debug"
            debug_chunks_dir = debug_dir / "chunks"
            debug_pred_dir = debug_dir / "pred"
            debug_filtered_dir = debug_dir / "filtered"
            debug_filtered_txt_dir = debug_dir / "filtered_txt"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_chunks_dir.mkdir(parents=True, exist_ok=True)
            debug_pred_dir.mkdir(parents=True, exist_ok=True)
            debug_filtered_dir.mkdir(parents=True, exist_ok=True)
            debug_filtered_txt_dir.mkdir(parents=True, exist_ok=True)
        else:
            debug_dir = None
            debug_chunks_dir = None
            debug_pred_dir = None
            debug_filtered_dir = None
            debug_filtered_txt_dir = None

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            tmp_chunks = tmp / "chunks"
            tmp_pred = tmp / "pred"
            for p in (tmp_chunks, tmp_pred):
                p.mkdir(parents=True, exist_ok=True)

            # 1) Chunking
            with log_step("Chunking", logger):
                ImageChunker(str(image_path)).slice(
                    save_folder=str(tmp_chunks),
                    chunk_size=self.cfg.chunk_size,
                    overlap=self.cfg.overlap,
                )
                chunks = sorted(tmp_chunks.glob("*.png"))
                logger.info("Chunks: %d", len(chunks))

                if debug and debug_chunks_dir is not None:
                    # keep chunk images
                    for ch in chunks:
                        shutil.copy2(ch, debug_chunks_dir / ch.name)
            update(1, "Chunking done")      
            # 2) Prediction
            with log_step("Prediction", logger):
                YOLOSegPredictor(str(model_path), output_dir=str(tmp_pred)).predict(
                    image_dir=str(tmp_chunks),
                    conf=self.cfg.conf,
                    imgsz=self.cfg.imgsz,
                    device=self.cfg.device,
                )
                pred_txts = sorted(tmp_pred.glob("*.txt"))
                logger.info("Prediction txt: %d", len(pred_txts))

                if debug and debug_pred_dir is not None:
                    plotter = SegmentationPlotter(str(model_path))
                    for txt in pred_txts:
                        chunk_img = tmp_chunks / f"{txt.stem}.png"
                        if not chunk_img.exists():
                            continue
                        out_img = debug_pred_dir / f"{txt.stem}_pred.png"
                        plotter.plot_annotations(
                            str(chunk_img),
                            str(txt),
                            save=str(out_img),
                        )
            update(2, "Prediction done")
            # 3) Filtering edge-touching polygons (in-memory)
            with log_step("Filtering edge-touching polygons", logger):
                chunk_colls: dict[str, AnnotationCollection] = {}
                total = 0
                plotter = SegmentationPlotter(str(model_path)) if debug else None

                for txt in pred_txts:
                    anns = list(AnnotationParser(str(txt)).parse())
                    coll = AnnotationCollection(anns, collection_name=txt.stem)

                    filtered_coll = coll.remove_edge_cases(
                        threshold=self.cfg.edge_threshold
                    )

                    chunk_colls[txt.name] = filtered_coll
                    total += len(filtered_coll)

                    if debug and all(
                        d is not None
                        for d in (debug_filtered_dir, debug_filtered_txt_dir)
                    ):
                        # write filtered txt + plot per chunk
                        filtered_txt = (
                            debug_filtered_txt_dir / f"{txt.stem}_filtered.txt"
                        )
                        filtered_coll.write_annotations_to_txt(
                            str(filtered_txt), include_conf=True
                        )
                        chunk_img = tmp_chunks / f"{txt.stem}.png"
                        if chunk_img.exists():
                            out_img = (
                                debug_filtered_dir / f"{txt.stem}_filtered.png"
                            )
                            plotter.plot_annotations(
                                str(chunk_img),
                                str(filtered_txt),
                                save=str(out_img),
                            )

                logger.info("Kept polygons after edge filter: %d", total)
            update(3, "Filtering done")
            # 4) Unify chunk annotations back into full-image coordinates (in-memory)
            with log_step("Unifying chunk annotations", logger):
                unified_coll = unify_collections(
                    chunk_collections=chunk_colls,
                    chunk_size=self.cfg.chunk_size,
                    full_img_path=str(image_path),
                )
                logger.info("Unified polygons: %d", len(unified_coll))
                
                unified_coll = unified_coll.remove_edge_cases(
                        threshold=self.cfg.edge_threshold
                    )

                if debug and debug_dir is not None:
                    unified_txt = debug_dir / f"{image_stem}_unified.txt"
                    unified_coll.write_annotations_to_txt(
                        str(unified_txt), include_conf=True
                    )
                    unified_img = debug_dir / f"{image_stem}_unified.png"
                    SegmentationPlotter(str(model_path)).plot_annotations(
                        str(image_path),
                        str(unified_txt),
                        save=str(unified_img),
                    )

            # end of tempfile context (unified_coll still in memory)
            update(4, "Unifying done")  
        # 5) Polygon NMS (in-memory)
        with log_step("Applying polygon NMS", logger):
            before_n = len(unified_coll)

            kept_coll = unified_coll.nms(
                iou_threshold=self.cfg.iou_thresh,
                class_aware=False,
                return_indices=False,
            )

            after_n = len(kept_coll)

            logger.info("Polygons before NMS: %d", before_n)
            logger.info("Polygons after  NMS: %d", after_n)
            logger.info("NMS removed: %d polygons", before_n - after_n)

            final_txt = output_dir / f"{image_stem}.txt"
            kept_coll.write_annotations_to_txt(str(final_txt), include_conf=True)

            logger.info("NMS kept: %d → %s", after_n, final_txt)
            update(5, "NMS done")
            # 6) Final visualization (NMS result)
            with log_step("Visualization", logger):
                out_img = output_dir / f"{image_stem}_annotated.tif"
                SegmentationPlotter(str(model_path)).plot_annotations(
                    str(image_path),
                    str(final_txt),
                    seg=True,
                    save=str(out_img),
                )
                logger.info("Overlay saved: %s", out_img)

        # 7) Shape descriptors + crops (unchanged)
        with log_step("Shape descriptors & crops", logger):
            crops_dir = output_dir / f"{image_stem}_crops"
            crops = kept_coll.save_crops(
                image=str(image_path),
                out_dir=crops_dir,
                use_mask=True,
                file_prefix=image_stem,
                ext="png",
                denormalize=True,
            )

            df = kept_coll.shape_descriptors(crops=crops)
            shapes_csv = output_dir / f"{image_stem}_shapes.csv"
            df.to_csv(shapes_csv, index=False)

            logger.info("Saved %d crops → %s", len(crops), crops_dir)
            logger.info("Saved shape descriptors → %s", shapes_csv)
        update(6, "Shape descriptors done")
        logger.info("Done → %s", output_dir)
        return str(final_txt)


if __name__ == "__main__":
    input_path = "/media/steve/UHH_EXT/Pictures/transfer_2961247_files_8d6ee684/2024112_VeraTest_043.tif"
    model_path = "/home/steve/Desktop/NonofYaBuisness/zenodo/DaphnAI.pt"
    setup_logging()

    cfg = Settings(
        source= input_path,
        model_path= model_path,
        output_dir="/home/steve/Desktop/tester/",
        conf=0.58,
        imgsz=1024,
        device="cpu",
        chunk_size=(1024, 1024),
        overlap=300,
        edge_threshold=0.01,
        iou_thresh=0.7,
    )

    Pipeline(cfg).run(
        debug=True,  # set False if you don't want intermediate plots
    )
