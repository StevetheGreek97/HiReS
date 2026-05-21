from __future__ import annotations
import os
from ultralytics import YOLO

class YOLOSegPredictor:
    def __init__(self, model_path: str, output_dir: str = "output"):
        self.model = YOLO(model_path)
        self.output_dir = output_dir

    def predict(
        self,
        image_dir: str,
        conf: float = 0.5,
        imgsz: int = 1024,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        kwargs.setdefault('verbose', False)
        kwargs.setdefault('stream', True)
        kwargs.setdefault('visualize', False)

        for result in self.model(
            image_dir,
            conf=conf,
            imgsz=imgsz,
            device=device,
            **kwargs,
        ):
            image_name = os.path.splitext(os.path.basename(result.path))[0]
            result.save_txt(f"{self.output_dir}/{image_name}.txt", save_conf=True)