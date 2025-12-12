import os
#import logging
from typing import List, Union, Generator, Optional, Tuple
from os import PathLike
from .datatypes import Annotation, OrientedBoundingBox, BoundingBox

from shapely.errors import TopologicalError
from shapely.validation import explain_validity
from shapely.geometry import Polygon, MultiPolygon

#logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)

import os
from os import PathLike
from typing import Union, Optional, List, Tuple

from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import explain_validity
from shapely.errors import TopologicalError

from HiReS.source.anno.datatypes import Annotation, BoundingBox, OrientedBoundingBox, AnnotationCollection  # your class


class AnnotationParser:
    def __init__(self, txt_path: Union[str, PathLike], expect_confidence: bool = True):
        self.txt_path = str(txt_path)
        self.expect_confidence = expect_confidence
        self._check_existence()
        self._collection: Optional[AnnotationCollection] = None

    # -----------------------------
    # Core public API
    # -----------------------------
    def parse(self) -> AnnotationCollection:
        if self._collection is not None:
            return self._collection

        lines = self._read_valid_lines()
        annotations = []

        for line in lines:
            values = line.strip().split()
            ann = self._extract_data(values)
            if ann is not None:
                annotations.append(ann)

        filename = os.path.basename(self.txt_path)

        self._collection = AnnotationCollection(
            annotations=annotations,
            collection_name=filename
        )
        return self._collection


    # convenience: let the parser behave like a collection
    def __len__(self) -> int:
        return len(self.parse().annotations)

    def __getitem__(self, index: int) -> Annotation:
        return self.parse().annotations[index]

    def validate(self) -> List[str]:
        """
        Validate all annotations and return a list of error messages, if any.
        """
        errors = []
        collection = self.parse()
        for i, ann in enumerate(collection.annotations):
            if not ann.polygon.is_valid:
                errors.append(f"Annotation {i} invalid: {explain_validity(ann.polygon)}")
        return errors

    # -----------------------------
    # Internals
    # -----------------------------
    def _check_existence(self):
        if not os.path.exists(self.txt_path):
            raise FileNotFoundError(f"Annotation file {self.txt_path} not found.")

    def _read_valid_lines(self) -> List[str]:
        """
        Read all lines that look like valid annotations
        (at least 1 class_id + 1 coordinate pair).
        """
        with open(self.txt_path, "r") as f:
            return [
                line
                for line in f
                if len(line.strip().split()) >= 3
            ]

    def _extract_confidence(self, coords: List[float]) -> Tuple[List[float], Optional[float]]:
        """
        If expect_confidence=True and we have an odd number of floats, treat the last one as confidence.
        """
        if self.expect_confidence and len(coords) % 2 == 1:
            return coords[:-1], coords[-1]
        return coords, None

    def _extract_data(self, values: List[str]) -> Optional[Annotation]:
        """
        Convert one line (already split into tokens) into an Annotation.
        Returns None if the polygon is invalid or degenerate.
        """
        try:
            class_id = int(values[0])
            coords = list(map(float, values[1:]))
            coords, confidence = self._extract_confidence(coords)

            poly = self._to_shapely(coords)
            if poly is None:
                return None

            minx, miny, maxx, maxy = poly.bounds
            bbox = BoundingBox(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

            obb_poly = poly.oriented_envelope
            obb_coords = tuple(list(obb_poly.exterior.coords)[:4])
            obb = OrientedBoundingBox(coords=obb_coords)

            return Annotation(
                class_id=class_id,
                polygon=poly,
                confidence=confidence,
                bounding_box=bbox,
                oriented_bounding_box=obb,
            )
        except Exception as e:
            print(f"Failed to extract annotation: {e}")
            return None

    def _to_shapely(
        self,
        coordinates: List[float],
        source_id: Optional[str] = None,
    ) -> Optional[Polygon]:
        """
        Build a robust Shapely Polygon from a flat list of coordinates.
        Handles:
        - empty / too short
        - auto-closing the ring
        - buffer(0) fix
        - MultiPolygon → keep largest
        """
        if not coordinates or len(coordinates) < 6:
            return None

        coords = list(zip(coordinates[::2], coordinates[1::2]))

        # make sure closed
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        try:
            poly = Polygon(coords)

            if poly.area == 0:
                return None

            if not poly.is_valid:
                poly = poly.buffer(0)

            if isinstance(poly, MultiPolygon):
                if len(poly.geoms) == 0:
                    return None
                poly = max(poly.geoms, key=lambda p: p.area)

            if poly.is_empty or not poly.is_valid:
                return None

            return poly

        except TopologicalError:
            return None
        except Exception:
            return None



def _visualize_polygon(coords: List[Tuple[float, float]], title: str = "Invalid Polygon") -> None:

    import matplotlib.pyplot as plt
    x, y = zip(*coords)
    plt.figure(figsize=(5, 5))
    plt.plot(x + (x[0],), y + (y[0],), 'r--')
    plt.scatter(x, y, c='blue')
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.show()

