from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd

from ..album import Album
from ..collection import Collection
from .match_maker import CollectionMatchMaker


@dataclass
class PairInspector:
    gt_album: Album
    pred_album: Album

    def pairs_df(self, iou_threshold: float = 0.5) -> pd.DataFrame:
        """Return a DataFrame where each row is a Pair outcome across all matched collections.

        Columns include:
        - collection_name, image_path
        - left_index, right_index, iou, status
        - left_class, left_conf, left_area, left_perimeter, left_solidity, left_convexity, left_circularity, left_bbox_w, left_bbox_h, left_obb_w, left_obb_l
        - right_class, right_conf, right_area, right_perimeter, right_solidity, right_convexity, right_circularity, right_bbox_w, right_bbox_h, right_obb_w, right_obb_l

        Uses `CollectionMatchMaker(gt, pred, iou_threshold)` for each matched collection name in the albums.
        """
        rows: List[Dict[str, Any]] = []

        left_map: Dict[str, Collection] = self.gt_album.name_map
        right_map: Dict[str, Collection] = self.pred_album.name_map

        matched = sorted(set(left_map.keys()) & set(right_map.keys()))

        for name in matched:
            left_col = left_map[name]
            right_col = right_map[name]
            cm = CollectionMatchMaker(left_col, right_col, iou_threshold=float(iou_threshold))

            for p in cm.pairs_list():
                left_ann = p.left_ann
                right_ann = p.right_ann

                left_dict = left_ann.to_dict() if left_ann is not None else {
                    "class_id": None,
                    "confidence": None,
                    "area": None,
                    "perimeter": None,
                    "solidity": None,
                    "convexity": None,
                    "circularity": None,
                    "bbox_width": None,
                    "bbox_height": None,
                    "obb_width": None,
                    "obb_length": None,
                }

                right_dict = right_ann.to_dict() if right_ann is not None else {
                    "class_id": None,
                    "confidence": None,
                    "area": None,
                    "perimeter": None,
                    "solidity": None,
                    "convexity": None,
                    "circularity": None,
                    "bbox_width": None,
                    "bbox_height": None,
                    "obb_width": None,
                    "obb_length": None,
                }

                row: Dict[str, Any] = {
                    "collection_name": name,
                    "image_path": getattr(left_col, "image_path", None) or getattr(right_col, "image_path", None),
                    "iou": float(p.iou),
                    "status": p.status,
                    # left descriptors prefixed
                    "left_class": left_dict.get("class_id"),
                    "left_conf": left_dict.get("confidence"),
                    "left_area": left_dict.get("area"),
                    "left_perimeter": left_dict.get("perimeter"),
                    "left_solidity": left_dict.get("solidity"),
                    "left_convexity": left_dict.get("convexity"),
                    "left_circularity": left_dict.get("circularity"),
                    "left_bbox_w": left_dict.get("bbox_width"),
                    "left_bbox_h": left_dict.get("bbox_height"),
                    "left_obb_w": left_dict.get("obb_width"),
                    "left_obb_l": left_dict.get("obb_length"),
                    # right descriptors prefixed
                    "right_class": right_dict.get("class_id"),
                    "right_conf": right_dict.get("confidence"),
                    "right_area": right_dict.get("area"),
                    "right_perimeter": right_dict.get("perimeter"),
                    "right_solidity": right_dict.get("solidity"),
                    "right_convexity": right_dict.get("convexity"),
                    "right_circularity": right_dict.get("circularity"),
                    "right_bbox_w": right_dict.get("bbox_width"),
                    "right_bbox_h": right_dict.get("bbox_height"),
                    "right_obb_w": right_dict.get("obb_width"),
                    "right_obb_l": right_dict.get("obb_length"),
                }

                rows.append(row)

        df = pd.DataFrame(rows)

        # CollectionMatchMaker labels unmatched-left as "fp" and unmatched-right
        # as "fn", assuming pred-on-left convention.  PairInspector puts GT on
        # the left, so those labels are semantically backwards here.
        _remap = {"fp": "missed_gt", "fn": "fp"}
        df["status"] = df["status"].map(lambda s: _remap.get(s, s))

        return df
