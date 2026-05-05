from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree


@dataclass
class IoUMatrix:
    left: Any
    right: Any
    return_dense: bool = True

    values: dict[tuple[int, int], float] = field(default_factory=dict, init=False)
    dense: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.values, self.dense = self._compute_iou()

    @property
    def left_annotations(self) -> list[Any]:
        return list(self.left.annotations)

    @property
    def right_annotations(self) -> list[Any]:
        return list(self.right.annotations)

    @property
    def left_polygons(self) -> list[Polygon]:
        return [ann.polygon for ann in self.left_annotations]

    @property
    def right_polygons(self) -> list[Polygon]:
        return [ann.polygon for ann in self.right_annotations]

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.left_annotations), len(self.right_annotations)

    def _compute_iou(self) -> tuple[dict[tuple[int, int], float], np.ndarray | None]:
        left_polygons = self.left_polygons
        right_polygons = self.right_polygons

        n_left = len(left_polygons)
        n_right = len(right_polygons)

        dense = np.zeros((n_left, n_right), dtype=float) if self.return_dense else None
        values: dict[tuple[int, int], float] = {}

        if n_left == 0 or n_right == 0:
            return values, dense

        tree = STRtree(right_polygons)

        for left_index, left_polygon in enumerate(left_polygons):
            for candidate in tree.query(left_polygon):
                right_index = self._resolve_candidate_index(candidate, right_polygons)
                iou = self.compute_iou(left_polygon, right_polygons[right_index])

                values[(left_index, right_index)] = iou
                if dense is not None:
                    dense[left_index, right_index] = iou

        return values, dense

    @staticmethod
    def compute_iou(a: Polygon, b: Polygon) -> float:
        if a.is_empty or b.is_empty:
            return 0.0

        union = a.union(b).area
        if union == 0:
            return 0.0

        return a.intersection(b).area / union

    @staticmethod
    def _resolve_candidate_index(candidate: Any, polygons: Sequence[Polygon]) -> int:
        if isinstance(candidate, (int, np.integer)):
            return int(candidate)

        for index, polygon in enumerate(polygons):
            if candidate.equals(polygon):
                return index

        raise ValueError("Could not resolve STRtree candidate index.")

    def require_dense(self) -> np.ndarray:
        if self.dense is None:
            raise ValueError("Dense IoU matrix is required.")
        return self.dense

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": self.values,
            "dense": self.dense,
            "shape": self.shape,
        }