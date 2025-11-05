from shapely.geometry import Polygon
from typing import  Optional
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ParsedAnnotationData:
    class_id: int
    polygon: Polygon
    confidence: Optional[float]
    bounding_box: dict
    oriented_bounding_box: Optional[dict] = None


