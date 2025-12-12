import os
from typing import Dict
import cv2
import numpy as np
from ..anno.parser import AnnotationParser
from ultralytics import YOLO
from ultralytics.utils.plotting import colors as yolo_colors
import os
from typing import Dict
import cv2
import numpy as np
import yaml  # <-- new
from ..anno.parser import AnnotationParser
from ultralytics import YOLO
from ultralytics.utils.plotting import colors as yolo_colors


class SegmentationPlotter:
    """
    Plot YOLO-style segmentation annotations on an image.

    Can be initialized with:
      - a model weights file (.pt, .onnx, etc.)  → classes loaded from YOLO model
      - a data.yaml (.yaml/.yml)                → classes loaded from yaml['names']
    """

    def __init__(self, source_path: str):
        """
        Parameters
        ----------
        source_path : str
            Path to either:
              - YOLO model weights file (e.g. 'model.pt')
              - data.yaml file with a 'names' field
        """
        self.source_path = source_path
        self.classes = self._load_classes()
        self.class_colors = self._generate_class_colors()

    # -----------------------------
    # Class loading
    # -----------------------------
    def _normalize_names(self, names) -> Dict[int, str]:
        """
        Normalize various 'names' formats (list, dict) to Dict[int, str].
        """
        if isinstance(names, dict):
            # keys might be str or int
            return {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, (list, tuple)):
            return {i: str(n) for i, n in enumerate(names)}
        else:
            raise ValueError(f"Unsupported 'names' format in {self.source_path}: {type(names)}")

    def _load_from_model(self) -> Dict[int, str]:
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Model weights file {self.source_path} not found.")
        model = YOLO(self.source_path)
        return self._normalize_names(model.names)

    def _load_from_yaml(self) -> Dict[int, str]:
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"data.yaml file {self.source_path} not found.")
        with open(self.source_path, "r") as f:
            data = yaml.safe_load(f)

        if "names" not in data:
            raise KeyError(f"'names' field not found in yaml file: {self.source_path}")

        return self._normalize_names(data["names"])

    def _load_classes(self) -> Dict[int, str]:
        ext = os.path.splitext(self.source_path)[1].lower()

        # very simple heuristic: yaml vs model
        if ext in {".yaml", ".yml"}:
            return self._load_from_yaml()
        else:
            # assume it's a model (pt, onnx, engine, etc.)
            return self._load_from_model()

    def _generate_class_colors(self) -> Dict[int, tuple]:
        return {i: yolo_colors(i) for i in self.classes}

    # -----------------------------
    # Plotting
    # -----------------------------
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


        colection = AnnotationParser(txt_path).parse()

        h, w = image.shape[:2]
        overlay = image.copy()


        # Enumerate to get line number (0-based index)
        for idx, ann in enumerate(colection.annotations, start=0):

            class_id = ann.class_id
            class_name = self.classes.get(class_id, str(class_id))
            color = self.class_colors.get(class_id, (0, 255, 0))
            confidence = ann.confidence

            polygon_np = np.array(ann.polygon.exterior.coords[:-1], dtype=np.float32)
            polygon_np *= [w, h]
            polygon_np = polygon_np.astype(np.int32)

            if seg:
                # First draw a thicker white line
                cv2.polylines(
                    overlay, [polygon_np],
                    isClosed=True,
                    color=(255, 255, 255),
                    thickness=4,
                    lineType=cv2.LINE_AA
                )
                # Then draw a thinner black line on top
                cv2.polylines(
                    overlay, [polygon_np],
                    isClosed=True,
                    color=(0, 0, 0),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
                cv2.fillPoly(overlay, [polygon_np], color)

            if bbox and ann.bounding_box:
                xmin = int(ann.bounding_box.minx * w)
                ymin = int(ann.bounding_box.miny * h)
                xmax = int(ann.bounding_box.maxx * w)
                ymax = int(ann.bounding_box.maxy * h)
                cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, 2)

            # Build label: "<line_number> <class_name> [conf]"
            label = ""
            if include_name:
                label += f"{idx} {class_name}"
            if include_conf and confidence is not None:
                label += f" {confidence:.2f}"

            if label:
                center_x, center_y = polygon_np.mean(axis=0).astype(int)
                label_y = center_y - 10 if center_y - 10 > 10 else center_y + 20

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5

                # Text outline
                cv2.putText(
                    overlay,
                    label,
                    (center_x + 1, label_y + 1),
                    font,
                    font_scale,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    label,
                    (center_x, label_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Blend once after drawing everything
        result = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

        if save:
            cv2.imwrite(save, result)
            print(f"Annotated image saved to: {save}")
