# HiReS/pipeline.py
import time
import tempfile
from contextlib import contextmanager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from HiReS.config import Settings
from HiReS.ios.chunker import ImageChunker
from HiReS.ios.yolo_predictor import YOLOSegPredictor
from HiReS.anno.parser import AnnotationParser
from HiReS.anno.ops import filter_touching_edges, unify, write_annotations_to_txt
from HiReS.anno.nms import run_nms
from HiReS.ios.plotting import SegmentationPlotter


import logging, sys, queue
from logging.handlers import QueueHandler, QueueListener

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
    # return so caller can stop it on exit if they want
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
        input_path: str,
        model_path: str,
        output_dir: str,
        *,
        workers: int = 1,
        patterns: tuple[str, ...] = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"),
    ) -> list[str] | str:
        """
        Smart entry:
          - If input_path is a FILE → process single image, return final .txt path (str).
          - If input_path is a DIR  → process all images inside, return list of final .txt paths.
        """
        ipath = Path(input_path)
        if ipath.is_file():
            self.log.info("Detected single image: %s", ipath)
            return self._run_single(ipath, Path(model_path), Path(output_dir), self.log)

        if not ipath.is_dir():
            raise FileNotFoundError(f"Input path not found: {ipath}")

        # Directory mode
        out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        images: list[Path] = []
        for pat in patterns:
            images.extend(sorted(ipath.glob(pat)))
        self.log.info("Detected directory: %s (%d images, patterns=%s)", ipath, len(images), patterns)

        if not images:
            self.log.warning("No images found in %s", ipath)
            return []

        results: list[str] = []
        workers = max(1, int(workers))
        self.log.info("Processing in parallel with workers=%d", workers)

        def submit(img: Path):
            child = logging.getLogger(f"HiReS.Pipeline[{img.name}]")
            try:
                return self._run_single(img, Path(model_path), out_dir, child)
            except Exception as e:
                child.exception("Failed: %s", e)
                return None

        if workers == 1:
            for img in images:
                r = submit(img)
                if r: results.append(r)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(submit, img): img for img in images}
                for fut in as_completed(futs):
                    r = fut.result()
                    if r: results.append(r)

        self.log.info("Completed %d/%d images", len(results), len(images))
        return results

    # ---------- internal: single image ----------
    def _run_single(self, image_path: Path, model_path: Path, output_dir: Path, logger: logging.Logger) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        image_stem = image_path.stem

        logger.info("Image: %s | Model: %s", image_path, model_path)
        logger.info("Config: conf=%.3f imgsz=%d device=%s chunk=%s overlap=%d edge_thr=%.4g iou_thr=%.3f",
                    self.cfg.conf, self.cfg.imgsz, self.cfg.device, self.cfg.chunk_size,
                    self.cfg.overlap, self.cfg.edge_threshold, self.cfg.iou_thresh)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            tmp_chunks, tmp_pred, tmp_filt = tmp / "chunks", tmp / "pred", tmp / "filtered"
            for p in (tmp_chunks, tmp_pred, tmp_filt): p.mkdir(parents=True, exist_ok=True)

            with log_step("Chunking", logger):
                ImageChunker(str(image_path)).slice(save_folder=str(tmp_chunks),
                                                    chunk_size=self.cfg.chunk_size,
                                                    overlap=self.cfg.overlap)
                logger.info("Chunks: %d", len(list(tmp_chunks.glob("*.png"))))

            with log_step("Prediction", logger):
                YOLOSegPredictor(str(model_path), output_dir=str(tmp_pred)).predict(
                    image_dir=str(tmp_chunks), conf=self.cfg.conf, imgsz=self.cfg.imgsz, device=self.cfg.device)
                logger.info("Prediction txt: %d", len(list(tmp_pred.glob("*.txt"))))

            with log_step("Filtering edge-touching polygons", logger):
                total = 0
                for txt in tmp_pred.glob("*.txt"):
                    anns = list(AnnotationParser(str(txt)).parse())
                    filtered = filter_touching_edges(anns, threshold=self.cfg.edge_threshold)
                    write_annotations_to_txt(filtered, str(tmp_filt / txt.name), include_conf=True)
                    total += len(filtered)
                logger.info("Kept polygons: %d", total)

            unified_txt = tmp / "unified.txt"
            with log_step("Unifying chunk annotations", logger):
                unify(str(tmp_filt), str(unified_txt), self.cfg.chunk_size, str(image_path))
                logger.info("Unified: %s", unified_txt)

            with log_step("Applying polygon NMS", logger):
                kept = run_nms(AnnotationParser(str(unified_txt)), iou_thresh=self.cfg.iou_thresh)
                final_txt = output_dir / f"{image_stem}.txt"
                write_annotations_to_txt(kept, str(final_txt), include_conf=True)
                logger.info("NMS kept: %d → %s", len(kept), final_txt)

            with log_step("Visualization", logger):
                out_img = output_dir / f"{image_stem}_annotated.tif"
                SegmentationPlotter(str(model_path)).plot_annotations(str(image_path), str(final_txt), save=str(out_img))
                logger.info("Overlay saved: %s", out_img)

        logger.info("Done → %s", output_dir)
        return str(final_txt)
if __name__ == "__main__":
    input_path = '/media/steve/UHH_EXT/Pictures/transfer_2961247_files_8d6ee684/2024112_VeraTest_043.tif'
    model_path = '/home/steve/Desktop/NonofYaBuisness/zenodo/DaphnAI.pt'
    setup_logging()

    cfg = Settings(
        conf=0.58, imgsz=1024, device="cpu",
        chunk_size=(1024, 1024), overlap=300,
        edge_threshold=0.01, iou_thresh=0.7
    )

    Pipeline(cfg).run(
        image_path=input_path,
        model_path=model_path, 
        output_dir="/home/steve/Desktop/tester/"
    )