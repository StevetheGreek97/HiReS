import os
#import logging
from typing import List, Union, Generator, Optional, Tuple
from os import PathLike
from .datatypes import ParsedAnnotationData

from shapely.errors import TopologicalError
from shapely.validation import explain_validity
from shapely.geometry import Polygon, MultiPolygon

#logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)

class AnnotationParser:
    def __init__(self, txt_path: Union[str, PathLike], expect_confidence: bool = True):
        self.txt_path = str(txt_path)
        self.expect_confidence = expect_confidence
        self._check_existence()
        self._num_annotations = None
        self._cached_annotations: Optional[List[ParsedAnnotationData]] = None

    def _check_existence(self):
        if not os.path.exists(self.txt_path):
            raise FileNotFoundError(f"Annotation file {self.txt_path} not found.")

    def parse(self) -> Generator[ParsedAnnotationData, None, None]:
        if self._cached_annotations is not None:
            yield from self._cached_annotations
        else:
            parsed = [self._extract_data(line.strip().split()) for line in self._read_valid_lines()]
            self._cached_annotations = [ann for ann in parsed if ann is not None]
            yield from self._cached_annotations

    def get_num_annotations(self) -> int:
        return len(self._read_valid_lines())

    def _read_valid_lines(self) -> List[str]:
        with open(self.txt_path, 'r') as file:
            return [line for line in file if len(line.strip().split()) >= 3]

    def _to_shapely(
        self,
        coordinates: List[float],
        source_id: Optional[str] = None,
    ) -> Optional[Polygon]:
        coords = list(zip(coordinates[::2], coordinates[1::2]))
        context = f"[{source_id}]" if source_id else ""

        #logger.debug(f"{context} Creating polygon with coordinates: {coords}")

        if not coordinates:
            #logger.error(f"{context} Empty coordinates list provided to _to_shapely.")
            return None

        if len(coordinates) < 6:
            #logger.warning(f"{context} Insufficient coordinates to form a polygon: {coordinates}")
            return None

        if coords[0] != coords[-1]:
            coords.append(coords[0])
            #logger.debug(f"{context} Polygon coordinates were not closed. Automatically closing the polygon.")

        try:
            poly = Polygon(coords)

            if poly.area == 0:
               # logger.warning(f"{context} Polygon has zero area. Possibly degenerate: {coords}")
                return None

            if not poly.is_valid:
                reason = explain_validity(poly)
                #logger.warning(f"{context} Initial polygon is invalid: {reason}")
                #logger.info(f"{context} Attempting to fix polygon with buffer(0).")
                poly = poly.buffer(0)

            if isinstance(poly, MultiPolygon):
                #logger.warning(f"{context} Converted to MultiPolygon after buffer fix.")
                if len(poly.geoms) == 0:
                    #logger.error(f"{context} MultiPolygon is empty after fixing.")
                    return None
                #logger.info(f"{context} MultiPolygon contains {len(poly.geoms)} polygons. Selecting largest by area.")
                #_visualize_polygon(coords, title="Invalid Polygon After Fix") debug
                poly = max(poly.geoms, key=lambda p: p.area)

            if poly.is_empty:
                #logger.error(f"{context} Polygon is empty after fix attempt.")
                return None

            if not poly.is_valid:
                reason = explain_validity(poly)
                #logger.error(f"{context} Polygon is still invalid after fix: {reason}")
                return None

            #logger.debug(f"{context} Polygon successfully created with bounds: {poly.bounds}")
            return poly

        except TopologicalError as e:
            #logger.error(f"{context} TopologicalError while creating polygon: {coords} -> {e}")
            return None
        except Exception as e:
           # logger.exception(f"{context} Unexpected error while creating polygon: {coords} -> {e}")
            return None


        except TopologicalError as e:
            #logger.error(f"Topology error for polygon {coords}: {e}")
            return None
        except Exception as e:
            #logger.error(f"Unexpected error while creating polygon: {e}")
            return None

    def _extract_confidence(self, coords: List[float]) -> Tuple[List[float], Optional[float]]:
        if self.expect_confidence and len(coords) % 2 == 1:
            return coords[:-1], coords[-1]
        return coords, None

    def _extract_data(self, values: List[str]) -> Optional[ParsedAnnotationData]:
        try:
            class_id = int(values[0])
            coords = list(map(float, values[1:]))
            coords, confidence = self._extract_confidence(coords)

            poly = self._to_shapely(coords)
            if poly is None:
                return None

            minx, miny, maxx, maxy = poly.bounds
            bbox = {
                "xmin": minx,
                "ymin": miny,
                "xmax": maxx,
                "ymax": maxy,
                "width": maxx - minx,
                "height": maxy - miny
            }

            return ParsedAnnotationData(
                class_id=class_id,
                polygon=poly,
                confidence=confidence,
                bounding_box=bbox,
                oriented_bounding_box=None
            )
        except Exception as e:
            print(f"Failed to extract annotation: {e}")
            return None

    def __len__(self) -> int:
        if self._num_annotations is None:
            self._num_annotations = self.get_num_annotations()
        return self._num_annotations

    def __getitem__(self, index: int) -> ParsedAnnotationData:
        annotations = list(self.parse())
        if index < 0:
            index += len(annotations)
        if index < 0 or index >= len(annotations):
            raise IndexError("Annotation index out of range.")
        return annotations[index]

    def validate(self) -> List[str]:
        errors = []
        for i, ann in enumerate(self.parse()):
            if not ann.polygon.is_valid:
                errors.append(f"Annotation {i} invalid: {explain_validity(ann.polygon)}")
        return errors


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

