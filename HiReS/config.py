# HiReS/config.py
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Settings:
    conf: float = 0.5
    imgsz: int = 1024
    device: str = "cpu"
    chunk_size: Tuple[int, int] = (1024, 1024)
    overlap: int = 150
    edge_threshold: float = 1e-2
    iou_thresh: float = 0.7
