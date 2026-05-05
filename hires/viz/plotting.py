from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict

import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.utils.plotting import colors as yolo_colors

from hires.models.parser import AnnotationParser

if TYPE_CHECKING:
    from models.collection import Collection as AnnotationCollection
    from models.eval.match_maker import MatchMaker as AnnotationMatchResult


class SegmentationPlotter:
    """Plot YOLO-style segmentation annotations on an image."""

    def __init__(self, source_path: str):
        self.source_path = source_path
        self.classes = self._load_classes()
        self.class_colors = self._generate_class_colors()

    def _normalize_names(self, names) -> Dict[int, str]:
        if isinstance(names, dict):
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
        if ext in {".yaml", ".yml"}:
            return self._load_from_yaml()
        return self._load_from_model()

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
        include_conf: bool = True,
    ) -> None:
        if not os.path.exists(txt_path):
            print(f"Skipping {image_path}: No annotation file found.")
            return

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image from {image_path}")
            return

        collection = AnnotationParser(txt_path).parse()
        h, w = image.shape[:2]
        overlay = image.copy()

        for idx, ann in enumerate(collection.annotations):
            class_id = ann.class_id
            class_name = self.classes.get(class_id, str(class_id))
            color = self.class_colors.get(class_id, (0, 255, 0))
            confidence = ann.confidence

            polygon_np = np.array(ann.polygon.exterior.coords[:-1], dtype=np.float32)
            polygon_np *= [w, h]
            polygon_np = polygon_np.astype(np.int32)

            if seg:
                cv2.polylines(overlay, [polygon_np], isClosed=True,
                              color=(255, 255, 255), thickness=4, lineType=cv2.LINE_AA)
                cv2.polylines(overlay, [polygon_np], isClosed=True,
                              color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                cv2.fillPoly(overlay, [polygon_np], color)

            if bbox and ann.bounding_box:
                xmin = int(ann.bounding_box.minx * w)
                ymin = int(ann.bounding_box.miny * h)
                xmax = int(ann.bounding_box.maxx * w)
                ymax = int(ann.bounding_box.maxy * h)
                cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, 2)

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
                cv2.putText(overlay, label, (center_x + 1, label_y + 1),
                            font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(overlay, label, (center_x, label_y),
                            font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        result = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
        if save:
            cv2.imwrite(save, result)

    def plot_comparison(
        self,
        image_path: str,
        pred_collection: AnnotationCollection,
        gt_collection: AnnotationCollection,
        match_result: AnnotationMatchResult,
        out_dir: str,
        stem: str,
    ) -> dict[str, str]:
        """Draw TP/FP/FN overlays and save a combined comparison image."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image from {image_path}")

        h, w = image.shape[:2]
        overlay = image.copy()

        tp_pred_set = {m.pred_index for m in match_result.matches if m.class_match}
        mismatch_pred_set = {m.pred_index for m in match_result.matches if not m.class_match}
        fp_pred_set = set(match_result.unmatched_pred_indices)
        fn_gt_set = set(match_result.unmatched_gt_indices)

        COLOR_TP = (0, 200, 0)
        COLOR_FP = (0, 0, 220)
        COLOR_FN = (220, 100, 0)
        COLOR_MISMATCH = (0, 165, 255)

        def _draw(poly, color):
            pts = np.array(list(poly.exterior.coords[:-1]), dtype=np.float32)
            pts[:, 0] *= w
            pts[:, 1] *= h
            pts = pts.astype(np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(overlay, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)

        for idx, ann in enumerate(pred_collection.annotations):
            if ann.polygon.is_empty:
                continue
            if idx in tp_pred_set:
                _draw(ann.polygon, COLOR_TP)
            elif idx in mismatch_pred_set:
                _draw(ann.polygon, COLOR_MISMATCH)
            elif idx in fp_pred_set:
                _draw(ann.polygon, COLOR_FP)

        for idx, ann in enumerate(gt_collection.annotations):
            if ann.polygon.is_empty:
                continue
            if idx in fn_gt_set:
                _draw(ann.polygon, COLOR_FN)

        result = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{stem}_comparison.tif")
        cv2.imwrite(out_path, result)
        return {"comparison": out_path}


__all__ = ["SegmentationPlotter"]
