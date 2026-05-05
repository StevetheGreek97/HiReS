from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

from shapely.geometry import Polygon, box
from .types import BoundingBox, OrientedBoundingBox
from .utils import _square_plot_span


@lru_cache(maxsize=32)
def _safe_unit_box(threshold: float) -> Polygon:
    return box(0.0, 0.0, 1.0, 1.0).buffer(-threshold)

UNIT_FACTORS = {
    "nm": 25_400_000.0,
    "um": 25400.0,
    "mm": 25.4,
    "cm": 2.54,
    "m": 0.0254,
    "inch": 1.0,
}

@dataclass
class Annotation:
    """Single annotated polygon with class metadata and derived geometry."""

    class_id: int
    polygon: Polygon
    confidence: float | None = None
    dpi: float | None = None
    unit: str | None = None

    def set_scale(self, dpi: float | None = None, unit: str | None = None) -> None:
        self.dpi = dpi
        self.unit = unit

    @property
    def scale(self) -> float:
        if self.dpi is None or self.unit is None:
            return 1.0

        if self.unit not in UNIT_FACTORS:
            raise ValueError(f"Unsupported unit '{self.unit}'")

        return UNIT_FACTORS[self.unit] / self.dpi

    @property
    def bounding_box(self) -> BoundingBox:
        return BoundingBox.from_polygon(self.polygon)

    @property
    def oriented_bounding_box(self) -> OrientedBoundingBox | None:
        if self.polygon.is_empty or not self.polygon.is_valid:
            return None

        obb = self.polygon.minimum_rotated_rectangle
        if obb.is_empty or not obb.is_valid:
            return None

        coords = list(obb.exterior.coords)[:-1]
        if len(coords) != 4:
            return None

        return OrientedBoundingBox(coords=tuple(coords))
    # Geometry properties
    @property
    def area(self) -> float:
        return self.polygon.area * (self.scale ** 2)

    @property
    def perimeter(self) -> float:
        return self.polygon.length * self.scale

    @property
    def convex_hull(self) -> Polygon:
        # geometry itself is unchanged (still pixel coords)
        return self.polygon.convex_hull

    @property
    def convex_hull_area(self) -> float:
        return self.convex_hull.area * (self.scale ** 2)

    @property
    def convex_hull_perimeter(self) -> float:
        return self.convex_hull.length * self.scale

    # Shape descriptors properties
    @property
    def solidity(self) -> float:
        hull_area = self.convex_hull.area
        return self.polygon.area / hull_area if hull_area > 0 else 0.0

    @property
    def convexity(self) -> float:
        hull_perimeter = self.convex_hull.length
        return hull_perimeter / self.polygon.length if self.polygon.length > 0 else 0.0

    @property
    def circularity(self) -> float:
        perimeter = self.polygon.length
        area = self.polygon.area
        return (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    def is_inside_unit_box(self, threshold: float = 1e-4) -> bool:
        safe_box = _safe_unit_box(threshold)
        return (
            self.polygon.is_valid
            and not self.polygon.is_empty
            and safe_box.contains(self.polygon)
        )
 
    def iou(self, other: "Annotation") -> float:
        if self.polygon.is_empty or other.polygon.is_empty:
            return 0.0

        inter = self.polygon.intersection(other.polygon).area
        union = self.polygon.union(other.polygon).area
        return inter / union if union > 0 else 0.0  

    def plot(
        self,
        obb: bool = False,
        box: bool = False,
        dims: bool = False,
        padding: float = 0.5,
        ax: Optional[Any] = None,
        show: bool = False,
        tight: bool = True,
        clear_ax: bool = False,
    ) -> Any:
        import matplotlib.pyplot as plt

        if dims and obb and box:
            raise ValueError("Only one of obb or box can be True when dims=True")
        if ax is None:
            ax = plt.gca()
        if clear_ax:
            ax.cla()

        figure = ax.figure
        cx, cy, span = _square_plot_span(self.polygon, padding=padding)
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_aspect("equal", adjustable="box")

        xx, yy = self.polygon.exterior.xy
        ax.fill(xx, yy, alpha=0.35, color="#4c72b0", label="Polygon")
        ax.plot(xx, yy, color="#1f3b5d", lw=2)

        if obb and self.oriented_bounding_box is not None:
            coords = list(self.oriented_bounding_box.coords)
            ring = coords + [coords[0]]
            obx, oby = zip(*ring)
            ax.plot(obx, oby, "-.", color="#e76f51", lw=2)
            if dims:
                width, length = self.oriented_bounding_box.width_length
                pcx, pcy = self.polygon.centroid.x, self.polygon.centroid.y
                ex = coords[1][0] - coords[0][0]
                ey = coords[1][1] - coords[0][1]
                edge_length = math.hypot(ex, ey) or 1e-9
                ux, uy = ex / edge_length, ey / edge_length
                vx, vy = -uy, ux
                ax.annotate("", xy=(pcx + (length / 2) * ux, pcy + (length / 2) * uy),
                            xytext=(pcx - (length / 2) * ux, pcy - (length / 2) * uy),
                            arrowprops=dict(arrowstyle="<->", color="#2ca02c", lw=1.6))
                ax.annotate("", xy=(pcx + (width / 2) * vx, pcy + (width / 2) * vy),
                            xytext=(pcx - (width / 2) * vx, pcy - (width / 2) * vy),
                            arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.6))
                ax.text(pcx, pcy, f"L={length:.4f}\nW={width:.4f}", ha="center", va="center",
                        fontsize=9, bbox=dict(fc="white", ec="black", alpha=0.9))

        if box and self.bounding_box is not None:
            bounds = self.bounding_box
            ax.plot(
                [bounds.minx, bounds.maxx, bounds.maxx, bounds.minx, bounds.minx],
                [bounds.miny, bounds.miny, bounds.maxy, bounds.maxy, bounds.miny],
                "--", color="#000000", lw=2,
            )
            if dims:
                box_cx, box_cy = bounds.center
                ax.annotate("", xy=(box_cx + bounds.width / 2, box_cy),
                            xytext=(box_cx - bounds.width / 2, box_cy),
                            arrowprops=dict(arrowstyle="<->", color="#2ca02c", lw=1.6))
                ax.annotate("", xy=(box_cx, box_cy + bounds.height / 2),
                            xytext=(box_cx, box_cy - bounds.height / 2),
                            arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.6))
                ax.text(box_cx, box_cy, f"W={bounds.width:.4f}\nH={bounds.height:.4f}",
                        ha="center", va="center", fontsize=9,
                        bbox=dict(fc="white", ec="black", alpha=0.9))

        title = f"Annotation (class_id={self.class_id}"
        title += f", conf={self.confidence:.3f})" if self.confidence is not None else ")"
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.6)
        if tight:
            figure.tight_layout()
        if show:
            plt.show()
        return ax

    def __repr__(self) -> str:
        return (
            f"Annotation("
            f"class_id={self.class_id}, "
            f"confidence={self.confidence}, "
            f"area={self.area:.4f}, "
            f"perimeter={self.perimeter:.4f}, "
            f"scale={self.scale:.6f}"
            f")"
        )

    def __str__(self) -> str:
        lines = [
            f"Annotation",
            f"  class_id   : {self.class_id}",
            f"  confidence : {self.confidence if self.confidence is not None else 'N/A'}",
            f"  area       : {self.area:.4f}",
            f"  perimeter  : {self.perimeter:.4f}",
            f"  solidity   : {self.solidity:.4f}",
            f"  convexity  : {self.convexity:.4f}",
            f"  circularity: {self.circularity:.4f}",
            f"  scale      : {self.scale:.6f}",
        ]
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        s = self.scale
        obb = self.oriented_bounding_box
        return {
            "class_id": self.class_id,
            "confidence": self.confidence,
            "area": self.area,
            "perimeter": self.perimeter,
            "solidity": self.solidity,
            "convexity": self.convexity,
            "circularity": self.circularity,
            "bbox_width": self.bounding_box.width * s,
            "bbox_height": self.bounding_box.height * s,
            "obb_width": obb.width_length[0] * s if obb is not None else None,
            "obb_length": obb.width_length[1] * s if obb is not None else None,
        }
