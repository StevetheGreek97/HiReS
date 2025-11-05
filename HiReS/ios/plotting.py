import os
from typing import Dict
import cv2
import numpy as np
from ..anno.parser import AnnotationParser
from ultralytics import YOLO
from ultralytics.utils.plotting import colors as yolo_colors


class SegmentationPlotter:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.classes = self._load_classes()
        self.class_colors = self._generate_class_colors()

    def _load_classes(self) -> Dict[int, str]:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model weights file {self.model_path} not found.")
        model = YOLO(self.model_path)
        return model.names

    def _generate_class_colors(self) -> Dict[int, tuple]:
        return {i: yolo_colors(i) for i in self.classes}

    def plot_annotations(
        self,
        image_path: str,
        txt_path: str,
        save: str,
        bbox: bool = True,
        seg: bool = True,
        include_name: bool = True,
        include_conf: bool = True
    ) -> None:
        if not os.path.exists(txt_path):
            print(f"Skipping {image_path}: No annotation file found.")
            return

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image from {image_path}")
            return

        parser = AnnotationParser(txt_path)
        annotations = list(parser.parse())

        h, w = image.shape[:2]
        overlay = image.copy()

        for ann in annotations:
            class_id = ann.class_id
            class_name = self.classes.get(class_id, str(class_id))
            color = self.class_colors.get(class_id, (0, 255, 0))
            confidence = ann.confidence
            polygon_np = np.array(ann.polygon.exterior.coords[:-1], dtype=np.float32)
            polygon_np *= [w, h]
            polygon_np = polygon_np.astype(np.int32)

            if seg:
                # First draw a thicker white line
                cv2.polylines(overlay, [polygon_np], isClosed=True, color=(255, 255, 255), thickness=4, lineType=cv2.LINE_AA)
                # Then draw a thinner black line on top
                cv2.polylines(overlay, [polygon_np], isClosed=True, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                cv2.fillPoly(overlay, [polygon_np], color)
                

            if bbox and ann.bounding_box:
                xmin = int(ann.bounding_box["xmin"] * w)
                ymin = int(ann.bounding_box["ymin"] * h)
                xmax = int(ann.bounding_box["xmax"] * w)
                ymax = int(ann.bounding_box["ymax"] * h)
                cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, 2)

            label = ""
            if include_name:
                label += f"{class_name}"
            if include_conf and confidence is not None:
                label += f" {confidence:.2f}"

            if label:
                center_x, center_y = polygon_np.mean(axis=0).astype(int)
                label_y = center_y - 10 if center_y - 10 > 10 else center_y + 20

                # Text with outline
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(overlay, label, (center_x + 1, label_y + 1), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(overlay, label, (center_x, label_y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Blend once after drawing everything
        result = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

        if save:
            cv2.imwrite(save, result)
            print(f"Annotated image saved to: {save}")
