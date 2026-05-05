from __future__ import annotations

import os
from os import PathLike
from pathlib import Path
from typing import Optional, Sequence

from shapely.geometry import MultiPolygon, Polygon
from shapely.errors import TopologicalError
from shapely.validation import explain_validity

from .annotation import Annotation
from .collection import Collection


class AnnotationParser:
    """Parse YOLO-style polygon annotation text files into an annotation collection."""

    def __init__(self, txt_path: str | PathLike[str]):
        self.txt_path = str(txt_path)
        self._check_existence()
        self._collection: Collection | None = None

    def parse(self) -> Collection:
        """Parse the file once and cache the resulting collection."""
        if self._collection is not None:
            return self._collection

        annotations: list[Annotation] = []
        for line in self._read_valid_lines():
            values = line.strip().split()
            annotation = self._extract_data(values)
            if annotation is not None:
                annotations.append(annotation)

        self._collection = Collection(
            annotations=annotations,
        collection_name=Path(self.txt_path).name,
        )
        return self._collection

    def __len__(self) -> int:
        return len(self.parse().annotations)

    def __getitem__(self, index: int | slice) -> Annotation | list[Annotation]:
        return self.parse().annotations[index]

    def validate(self) -> list[str]:
        """Validate parsed polygons and return any geometry errors."""
        errors: list[str] = []
        for index, annotation in enumerate(self.parse().annotations):
            if not annotation.polygon.is_valid:
                errors.append(
                    f"Annotation {index} invalid: {explain_validity(annotation.polygon)}"
                )
        return errors

    def _check_existence(self) -> None:
        if not os.path.exists(self.txt_path):
            raise FileNotFoundError(f"Annotation file {self.txt_path} not found.")

    def _read_valid_lines(self) -> list[str]:
        """
        Read lines with enough tokens to plausibly describe an annotation.

        A stricter geometric validation still happens later during parsing.
        """
        with open(self.txt_path, "r", encoding="utf-8") as handle:
            return [
                line
                for line in handle
                if len(line.strip().split()) >= 7
            ]

    def _extract_confidence(
        self,
        coordinates: list[float],
    ) -> tuple[list[float], float | None]:
        """Treat a trailing odd coordinate value as confidence."""
        if len(coordinates) % 2 == 1:
            return coordinates[:-1], coordinates[-1]
        return coordinates, 1

    @staticmethod
    def _polygon_from_flat_coordinates(coordinates: Sequence[float]) -> Polygon | None:
        """Build a robust polygon from a flat coordinate list."""
        if len(coordinates) < 6 or len(coordinates) % 2 != 0:
            return None

        coords = list(zip(coordinates[::2], coordinates[1::2]))
        if len(coords) < 3:
            return None

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        try:
            polygon = Polygon(coords)
        except (TopologicalError, Exception):
            return None

        if polygon.area == 0:
            return None

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if isinstance(polygon, MultiPolygon):
            if not polygon.geoms:
                return None
            polygon = max(polygon.geoms, key=lambda g: g.area)

        if polygon.is_empty or not polygon.is_valid or polygon.area == 0:
            return None

        return polygon

    def _extract_data(self, values: list[str]) -> Optional[Annotation]:
        """Convert one tokenized line into an Annotation."""
        try:
            class_id = int(values[0])
            coordinates = list(map(float, values[1:]))
            coordinates, confidence = self._extract_confidence(coordinates)

            polygon = self._polygon_from_flat_coordinates(coordinates)
            if polygon is None:
                return None

            return Annotation(
                class_id=class_id,
                polygon=polygon,
                confidence=confidence,
            )
        except Exception as exc:
            print(f"Failed to extract annotation from {self.txt_path}: {exc}")
            return None