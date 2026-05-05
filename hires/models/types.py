from __future__ import annotations

import math
from dataclasses import dataclass
from shapely.geometry import Polygon, box


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounds for an annotation polygon."""

    minx: float
    miny: float
    maxx: float
    maxy: float

    @classmethod
    def from_polygon(cls, polygon: Polygon) -> BoundingBox:
        minx, miny, maxx, maxy = polygon.bounds
        return cls(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    @property
    def width(self) -> float:
        return self.maxx - self.minx

    @property
    def height(self) -> float:
        return self.maxy - self.miny

    @property
    def center(self) -> tuple[float, float]:
        return ((self.minx + self.maxx) / 2, (self.miny + self.maxy) / 2)

    @property
    def geometry(self) -> Polygon:
        return box(self.minx, self.miny, self.maxx, self.maxy)

    def contains(self, other: BoundingBox) -> bool:
        return self.geometry.contains(other.geometry)

    def covers(self, other: BoundingBox) -> bool:
        return self.geometry.covers(other.geometry)

@dataclass(frozen=True)
class OrientedBoundingBox:
    """Minimum rotated rectangle represented by its four outer corners."""

    coords: tuple[tuple[float, float], ...]

    @classmethod
    def from_polygon(cls, polygon: Polygon) -> OrientedBoundingBox:
        obb_polygon = polygon.oriented_envelope
        coords = tuple(list(obb_polygon.exterior.coords)[:4])
        return cls(coords=coords)

    @property
    def width_length(self) -> tuple[float, float]:
        edges: list[float] = []
        for index in range(4):
            x1, y1 = self.coords[index]
            x2, y2 = self.coords[(index + 1) % 4]
            edges.append(math.hypot(x2 - x1, y2 - y1))
        unique_edges = sorted(set(round(edge, 8) for edge in edges))
        if len(unique_edges) == 1:
            return unique_edges[0], unique_edges[0]
        return unique_edges[0], unique_edges[1]

