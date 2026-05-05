from __future__ import annotations

from pathlib import Path
from typing import List

from hires.models.config import Settings
from hires.pipeline.logger import create_logger

IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


class BasePipeline:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.logger = create_logger()

    def _iter_images(self) -> List[Path]:
        src = Path(self.cfg.source)
        if src.is_file():
            if src.suffix.lower() in IMG_EXTS:
                return [src]
            return []
        if src.is_dir():
            pattern = "**/*" if getattr(self.cfg, "recursive", False) else "*"
            return sorted(
                p for p in src.glob(pattern)
                if p.is_file() and p.suffix.lower() in IMG_EXTS
            )
        return []

    def _detect_mode(self, src: Path) -> str:
        return "single" if src.is_file() else "batch"

    def _log_run_header(self, mode: str, count: int) -> None:
        self.logger.info(
            "Mode: %s | Images: %d | Output: %s", mode, count, self.cfg.output_dir
        )

    def _get_output_dir(self, img: Path) -> Path:
        base = Path(self.cfg.output_dir)
        src = Path(self.cfg.source)
        if src.is_dir():
            return base / img.relative_to(src).parent
        return base
