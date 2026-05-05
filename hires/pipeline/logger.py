from __future__ import annotations

import itertools
import logging
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path


def create_logger(name: str = "HiReS", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


@contextmanager
def log_step(name: str, logger: logging.Logger):
    t0 = time.perf_counter()
    logger.info(f"step={name}")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        logger.info(f"Done. Duration_s={dt:.3f}")


@contextmanager
def image_context(
    logger: logging.Logger,
    image_path: Path,
    idx: int,
    total: int,
):
    prefix = f"[{idx}/{total}] {image_path.name}"
    start = time.perf_counter()
    logger.info("START %s", prefix)
    try:
        yield prefix
    finally:
        dur = time.perf_counter() - start
        logger.info("DONE  %s (%.1fs)", prefix, dur)


@contextmanager
def spinner(message: str = "Working…", interval: float = 0.1):
    """Context manager that shows a terminal spinner on stderr while the block runs."""
    stop = threading.Event()

    def _spin():
        frames = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
        while not stop.is_set():
            sys.stderr.write(f"\r{next(frames)}  {message}")
            sys.stderr.flush()
            stop.wait(interval)
        sys.stderr.write("\r" + " " * (len(message) + 4) + "\r")
        sys.stderr.flush()

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()


__all__ = ["create_logger", "image_context", "log_step", "spinner"]
