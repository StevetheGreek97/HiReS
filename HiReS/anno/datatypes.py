from dataclasses import dataclass, field
from typing import Optional, Tuple,List, Iterable, Dict, Union, Any
from collections import defaultdict
from pathlib import Path
import copy
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, box
from shapely.strtree import STRtree
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

from dataclasses import dataclass

from shapely.ops import unary_union

@dataclass
class BoundingBox:
    minx: float
    miny: float
    maxx: float
    maxy: float

    @property
    def width(self) -> float:
        return self.maxx - self.minx

    @property
    def height(self) -> float:
        return self.maxy - self.miny

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.minx + self.maxx) / 2, (self.miny + self.maxy) / 2)

    @property
    def geometry(self):
        """Return a Shapely box (Polygon) for this bounding box."""
        return box(self.minx, self.miny, self.maxx, self.maxy)

    def contains(self, other: "BoundingBox") -> bool:
        """Strict containment: other must be fully inside, no touching edges."""
        return self.geometry.contains(other.geometry)

    def covers(self, other: "BoundingBox") -> bool:
        """Allow touching edges."""
        return self.geometry.covers(other.geometry)

@dataclass
class OrientedBoundingBox:
    coords: Tuple[Tuple[float, float], ...]  # 4 corners

    @property
    def width_length(self) -> Tuple[float, float]:
        edges = []
        for i in range(4):
            x1, y1 = self.coords[i]
            x2, y2 = self.coords[(i+1) % 4]
            edges.append(math.hypot(x2 - x1, y2 - y1))
        unique_edges = sorted(set(round(e, 8) for e in edges))
        if len(unique_edges) == 1:  # square
            return unique_edges[0], unique_edges[0]
        return unique_edges[0], unique_edges[1]

@dataclass
class Annotation:
    class_id: int
    polygon: Polygon
    confidence: Optional[float] = None
    bounding_box: BoundingBox = None
    oriented_bounding_box: Optional[OrientedBoundingBox] = None

    def plot(self, obb: bool = False, box: bool = False, dims: bool = False, padding: float = 0.05):
        """
        Plot polygon with optional bounding boxes and dimensions,
        auto-zooming tightly around the object.
        """

        if dims and obb and box:
            raise ValueError("Only one of obb or box can be True when dims=True")

        fig, ax = plt.subplots(figsize=(7, 7))

        # -------------------------------------------------
        # Auto-zoom around polygon
        # -------------------------------------------------
        minx, miny, maxx, maxy = self.polygon.bounds
        dx = maxx - minx
        dy = maxy - miny
        pad_x = dx * padding
        pad_y = dy * padding

        # Handle degenerate cases (very tiny polygons)
        if dx == 0:
            pad_x = padding
        if dy == 0:
            pad_y = padding

        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)

        # -------------------------------------------------
        # Base polygon (filled)
        # -------------------------------------------------
        xx, yy = self.polygon.exterior.xy
        ax.fill(xx, yy, alpha=0.35, color="#4c72b0", label="Polygon")
        ax.plot(xx, yy, color="#1f3b5d", lw=2)

        # -------------------------------------------------
        # Axis-aligned bounding box
        # -------------------------------------------------
        if box and self.bounding_box:
            bb = self.bounding_box
            ax.plot(
                [bb.minx, bb.maxx, bb.maxx, bb.minx, bb.minx],
                [bb.miny, bb.miny, bb.maxy, bb.maxy, bb.miny],
                "--",
                color="#222",
                lw=2,
                label="Bounding Box",
            )

            if dims:
                cx = (bb.minx + bb.maxx) / 2
                cy = (bb.miny + bb.maxy) / 2
                ax.text(
                    cx,
                    cy,
                    f"W={bb.width:.4f}\nH={bb.height:.4f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    bbox=dict(fc="white", ec="black", alpha=0.9),
                    color="black",
                )

        # -------------------------------------------------
        # Oriented bounding box + dims (all in one block)
        # -------------------------------------------------
        if obb and self.oriented_bounding_box:
            coords = list(self.oriented_bounding_box.coords)  # expected 4 points
            if len(coords) < 4:
                # fall back gracefully if something weird comes in
                coords = coords + coords[: 4 - len(coords)]

            # close the ring for plotting
            ring = coords + [coords[0]]
            obx, oby = zip(*ring)
            ax.plot(obx, oby, "-.", color="#e76f51", lw=2, label="Oriented BBox")

            if dims:
                # width & length from your OBB object
                width, length = self.oriented_bounding_box.width_length

                obb_poly = Polygon(coords)
                cx, cy = obb_poly.centroid.x, obb_poly.centroid.y

                # use first edge as the "length" direction
                dx = coords[1][0] - coords[0][0]
                dy = coords[1][1] - coords[0][1]
                L = math.hypot(dx, dy)
                if L == 0:
                    L = 1e-9  # avoid division by zero

                # unit vectors along length (u) and width (v)
                ux, uy = dx / L, dy / L          # length direction
                vx, vy = -uy, ux                 # width direction

                # ---- length arrow ----
                ax.annotate(
                    "",
                    xy=(cx + (length / 2) * ux, cy + (length / 2) * uy),
                    xytext=(cx - (length / 2) * ux, cy - (length / 2) * uy),
                    arrowprops=dict(arrowstyle="<->", color="#2ca02c", lw=1.6),
                )

                # ---- width arrow ----
                ax.annotate(
                    "",
                    xy=(cx + (width / 2) * vx, cy + (width / 2) * vy),
                    xytext=(cx - (width / 2) * vx, cy - (width / 2) * vy),
                    arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.6),
                )

                # combined label near the centroid
                ax.text(
                    cx,
                    cy,
                    f"L={length:.4f}\nW={width:.4f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    bbox=dict(fc="white", ec="black", alpha=0.9),
                    color="black",
                )

        # -------------------------------------------------
        # Cosmetics
        # -------------------------------------------------
        if self.confidence is not None:
            title = f"Annotation (class_id={self.class_id}, conf={self.confidence:.3f})"
        else:
            title = f"Annotation (class_id={self.class_id})"

        ax.set_title(title)
        ax.set_aspect("equal", "box")
        ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.show()


@dataclass
class AnnotationCollection:
    annotations: List[Annotation] = field(default_factory=list)
    collection_name: Optional[str] = None

    # -----------------------------
    # Basic sequence API
    # -----------------------------
    def __len__(self):
        return len(self.annotations)

    def __iter__(self):
        return iter(self.annotations)

    def __getitem__(self, idx):
        return self.annotations[idx]

    def add(self, annotation: Annotation):
        self.annotations.append(annotation)

    def extend(self, anns: Iterable[Annotation]):
        self.annotations.extend(anns)

    # -----------------------------
    # IoU BETWEEN COLLECTIONS (via STRtree)
    # -----------------------------
    def iou_with(
        self,
        other: "AnnotationCollection",
        return_dense: bool = True,
    ) -> Tuple[Dict[Tuple[int, int], float], np.ndarray | None]:

        polys_a = [ann.polygon for ann in self.annotations]
        polys_b = [ann.polygon for ann in other.annotations]

        Na, Nb = len(polys_a), len(polys_b)
        tree_b = STRtree(polys_b)

        # mapping for the "geometries returned" case
        geom_to_idx_b = {g.wkb: j for j, g in enumerate(polys_b)}

        iou_dict: Dict[Tuple[int, int], float] = {}
        dense = np.zeros((Na, Nb), dtype=float) if return_dense else None

        for i, poly_a in enumerate(polys_a):
            candidates = tree_b.query(poly_a)

            for cand in candidates:
                # shapely/pygeos can return either indices or geometries
                if isinstance(cand, (int, np.integer)):
                    j = int(cand)
                    poly_b = polys_b[j]
                else:
                    poly_b = cand
                    j = geom_to_idx_b[poly_b.wkb]

                inter = poly_a.intersection(poly_b).area
                if inter == 0.0:
                    val = 0.0
                else:
                    val = inter / (poly_a.area + poly_b.area - inter)

                iou_dict[(i, j)] = val
                if dense is not None:
                    dense[i, j] = val

        return iou_dict, dense

    # -----------------------------
    # IoU WITHIN THE SAME COLLECTION (via STRtree)
    # -----------------------------
    def iou_self(
        self,
        return_dense: bool = True,
    ) -> Tuple[Dict[Tuple[int, int], float], np.ndarray | None]:

        polys = [ann.polygon for ann in self.annotations]
        N = len(polys)

        tree = STRtree(polys)
        geom_to_idx = {g.wkb: i for i, g in enumerate(polys)}

        iou_dict: Dict[Tuple[int, int], float] = {}
        dense = np.zeros((N, N), dtype=float) if return_dense else None

        for i, poly_i in enumerate(polys):
            candidates = tree.query(poly_i)

            for cand in candidates:
                # again: indices vs geometries
                if isinstance(cand, (int, np.integer)):
                    j = int(cand)
                    poly_j = polys[j]
                else:
                    poly_j = cand
                    j = geom_to_idx[poly_j.wkb]

                if j <= i:
                    # skip self and already computed pairs
                    continue

                inter = poly_i.intersection(poly_j).area
                if inter == 0.0:
                    val = 0.0
                else:
                    val = inter / (poly_i.area + poly_j.area - inter)

                iou_dict[(i, j)] = val
                iou_dict[(j, i)] = val

                if dense is not None:
                    dense[i, j] = val
                    dense[j, i] = val

        return iou_dict, dense
    def get_conf(self, idx: int) -> float:
        conf = self.annotations[idx].confidence
        return 1.0 if conf is None else conf
    
    def nms(
        self,
        iou_threshold: float = 0.7,
        class_aware: bool = False,
        return_indices: bool = False,
    ) -> "AnnotationCollection | List[int]":
        """
        Perform Non-Maximum Suppression (NMS) on this collection using IoU.

        Uses `self.iou_self(return_dense=True)` to get a pairwise IoU matrix and then
        suppresses polygons that highly overlap (IoU > iou_threshold) with a
        higher-confidence polygon.

        Parameters
        ----------
        iou_threshold : float
            Suppress polygons with IoU > this value.
        class_aware : bool
            If True, NMS is done separately per class_id.
            If False, all annotations compete with each other.
        return_indices : bool
            If True, return a list of kept indices instead of a new collection.

        Returns
        -------
        AnnotationCollection or List[int]
            - If return_indices=False: a new AnnotationCollection with kept annotations.
            - If return_indices=True: a sorted list of indices kept from the original collection.
        """
        N = len(self.annotations)
        if N <= 1:
            # Trivial cases
            if return_indices:
                return list(range(N))
            return AnnotationCollection(self.annotations.copy())

        # 1) Get dense IoU matrix via existing method
        _, dense = self.iou_self(return_dense=True)
        # dense[i, j] = IoU between annotation i and j

        kept_indices: List[int] = []

        if class_aware:
            # 2a) Group indices by class_id
            class_to_indices: Dict[int, List[int]] = defaultdict(list)
            for idx, ann in enumerate(self.annotations):
                class_to_indices[ann.class_id].append(idx)

            # 3a) Run NMS independently on each class group
            for _, idxs in class_to_indices.items():
                # Sort by confidence descending
                idxs_sorted = sorted(idxs, key=self.get_conf, reverse=True)

                while idxs_sorted:
                    current = idxs_sorted.pop(0)
                    kept_indices.append(current)

                    # Suppress all remaining with IoU > threshold vs current
                    idxs_sorted = [
                        j for j in idxs_sorted
                        if dense[current, j] <= iou_threshold
                    ]
        else:
            # 2b) Global NMS (ignoring class_id)
            idxs_sorted = sorted(range(N), key=self.get_conf, reverse=True)

            while idxs_sorted:
                current = idxs_sorted.pop(0)
                kept_indices.append(current)

                idxs_sorted = [
                    j for j in idxs_sorted
                    if dense[current, j] <= iou_threshold
                ]

        # 4) De-duplicate & sort kept indices (just in case)
        kept_indices = sorted(set(kept_indices))

        if return_indices:
            return kept_indices

        # 5) Build new AnnotationCollection with kept annotations
        kept_annotations = [self.annotations[i] for i in kept_indices]
        return AnnotationCollection(kept_annotations, self.collection_name)
        
    def remove_edge_cases(
        self,
        threshold: float = 1e-4
    ) -> "AnnotationCollection":
        """
        Removes polygons that touch or cross the image edges (normalized [0, 1]),
        using the actual polygon geometry instead of its bounding box.

        Parameters
        ----------
        threshold : float, default 1e-4
            Small inward offset to avoid floating-point errors.

        Returns
        -------
        AnnotationCollection
            New collection with polygons fully contained inside the image.
        """
        image_box = box(0.0, 0.0, 1.0, 1.0)
        safe_box = image_box.buffer(-threshold)

        filtered: list[Annotation] = []
        for ann in self.annotations:
            poly = ann.polygon
            if not poly.is_valid or poly.is_empty:
                continue
            if safe_box.contains(poly):
                filtered.append(ann)

        return AnnotationCollection(filtered, collection_name=self.collection_name)


    def shape_descriptors(self, crops: list[str] | None = None) -> pd.DataFrame:
        """
        Compute geometric shape descriptors for every Annotation in this collection,
        include collection_name, and return a pandas DataFrame.

        If `crops` is provided (list of file paths from `save_crops`), a
        'crop_path' column is added and linked to the corresponding annotation
        by index (parsed from 'idx{idx}' in the filenames).

        Parameters
        ----------
        crops : list[str] | None
            Optional list of crop file paths, typically as returned by `save_crops`.

        Returns
        -------
        pd.DataFrame
            Each row = one annotation with full set of shape descriptors,
            optionally including 'crop_path'.
        """
        # Map idx -> crop_path by parsing filenames like "..._idx3_class1_conf0.95.png"
        idx_to_crop: dict[int, str] = {}
        if crops is not None:
            for p in crops:
                stem = Path(p).stem  # e.g. "crop_idx3_class1_conf0.95"
                parts = stem.split("_")
                for part in parts:
                    if part.startswith("idx"):
                        try:
                            idx = int(part[3:])
                            idx_to_crop[idx] = p
                        except ValueError:
                            pass
                        break

        descriptors: List[Dict[str, Any]] = []
        collection_name = self.collection_name if self.collection_name else None

        for idx, ann in enumerate(self.annotations):
            poly = ann.polygon

            base_row: Dict[str, Any] = {
                "collection_name": collection_name,
                "index": idx,
                "class_id": ann.class_id,
                "confidence": ann.confidence,
            }

            if poly.is_empty:
                row = {
                    **base_row,
                    "area": 0.0,
                    "perimeter": 0.0,
                    #"centroid_x": np.nan,
                    #"centroid_y": np.nan,
                    #"bbox_width": np.nan,
                    #"bbox_height": np.nan,
                    #"bbox_area": np.nan,
                    #"bbox_aspect_ratio": np.nan,
                    #"extent": np.nan,
                    "convex_area": np.nan,
                    "solidity": np.nan,
                    #"equivalent_diameter": np.nan,
                    "circularity": np.nan,
                    "obb_width": np.nan,
                    "obb_height": np.nan,
                    #"obb_aspect_ratio": np.nan,
                }
            else:
                # ---- Basic metrics ----
                area = poly.area
                perimeter = poly.length
                cx, cy = poly.centroid.x, poly.centroid.y

                # ---- Axis-aligned bounding box ----
                minx, miny, maxx, maxy = poly.bounds
                bbox_width = maxx - minx
                bbox_height = maxy - miny
                bbox_area = bbox_width * bbox_height
                bbox_aspect_ratio = (
                    bbox_width / bbox_height if bbox_height != 0 else np.nan
                )
                extent = area / bbox_area if bbox_area > 0 else np.nan

                # ---- Convex hull ----
                hull = poly.convex_hull
                convex_area = hull.area
                solidity = area / convex_area if convex_area > 0 else np.nan

                # ---- Equivalent diameter ----
                equivalent_diameter = (
                    math.sqrt(4 * area / math.pi) if area > 0 else np.nan
                )

                # ---- Circularity ----
                circularity = (
                    4 * math.pi * area / (perimeter * perimeter)
                    if perimeter > 0 else np.nan
                )

                # ---- Oriented bounding box (min rotated rect) ----
                mrr = poly.minimum_rotated_rectangle
                mrr_x, mrr_y = mrr.exterior.coords.xy
                coords = list(zip(mrr_x, mrr_y))[:-1]  # drop closing point

                if len(coords) >= 4:
                    edges = [
                        math.hypot(
                            coords[(i + 1) % 4][0] - coords[i][0],
                            coords[(i + 1) % 4][1] - coords[i][1],
                        )
                        for i in range(4)
                    ]
                    unique_edges = sorted(set(round(e, 8) for e in edges))
                    if len(unique_edges) == 1:
                        obb_width = obb_height = unique_edges[0]
                    else:
                        obb_width, obb_height = unique_edges[0], unique_edges[1]
                else:
                    obb_width = obb_height = np.nan

                if np.isfinite(obb_height) and obb_height != 0:
                    obb_aspect_ratio = obb_width / obb_height
                else:
                    obb_aspect_ratio = np.nan

                row = {
                    **base_row,
                    "area": area,
                    "perimeter": perimeter,
                    #"centroid_x": cx,
                    #"centroid_y": cy,
                    #"bbox_width": bbox_width,
                    #"bbox_height": bbox_height,
                    #"bbox_area": bbox_area,
                    #"bbox_aspect_ratio": bbox_aspect_ratio,
                    #"extent": extent,
                    "convex_area": convex_area,
                    "solidity": solidity,
                    #"equivalent_diameter": equivalent_diameter,
                    "circularity": circularity,
                    "obb_width": obb_width,
                    "obb_height": obb_height,
                    #"obb_aspect_ratio": obb_aspect_ratio,
                }

            # Add crop_path if we have a mapping
            if crops is not None:
                row["crop_path"] = idx_to_crop.get(idx, None)

            descriptors.append(row)
        return pd.DataFrame(descriptors)

    def save_crops(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        out_dir: Union[str, Path],
        use_mask: bool = True,
        file_prefix: str | None = None,
        ext: str = "png",
        denormalize: bool = True,
    ) -> list[str]:
        """
        Crop each annotation from the given image and save to a folder.

        Parameters
        ----------
        image : str | Path | np.ndarray | PIL.Image.Image
            Source image corresponding to this collection's annotations.
        out_dir : str or Path
            Directory where crops will be saved (created if it doesn't exist).
        use_mask : bool, default True
            If True, apply the polygon as a mask inside the bounding box
            (background becomes transparent). If False, saves a plain bbox crop.
        file_prefix : str or None, default None
            Optional prefix for filenames. If None, collection_name is used if set.
        ext : str, default "png"
            File extension (png recommended to preserve transparency).
        denormalize : bool, default True
            If True, assumes polygons are in normalized [0,1] coords and
            rescales them to pixel coordinates using image width/height.

        Returns
        -------
        List[str]
            List of file paths to the saved crops.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # normalize input image to PIL
        pil_img = Image.open(image).convert("RGBA")

        img_width, img_height = pil_img.size

        # prefix for filenames
        if file_prefix is None:
            file_prefix = self.collection_name or "crop"

        saved_paths: list[str] = []

        for idx, ann in enumerate(self.annotations):
            poly = ann.polygon

            if poly.is_empty:
                continue

            # denormalize polygon if requested
            if denormalize:
                poly_px = denormalize_polygon(poly, img_width, img_height)
            else:
                poly_px = poly

            if poly_px.is_empty:
                continue

            # Use bounding box of the (possibly denormalized) polygon
            minx, miny, maxx, maxy = poly_px.bounds

            # Ensure integer pixel coordinates
            left = max(int(math.floor(minx)), 0)
            upper = max(int(math.floor(miny)), 0)
            right = int(math.ceil(maxx))
            lower = int(math.ceil(maxy))

            if right <= left or lower <= upper:
                continue  # degenerate bbox

            # Crop the region from the image
            crop = pil_img.crop((left, upper, right, lower))

            if use_mask:
                # Create a mask with the polygon shifted into crop coordinates
                mask = Image.new("L", crop.size, 0)
                draw = ImageDraw.Draw(mask)

                # Shift polygon coordinates by bbox top-left
                poly_coords = []
                for x, y in poly_px.exterior.coords:
                    sx = x - left
                    sy = y - upper
                    poly_coords.append((sx, sy))

                # Draw the polygon as white on black mask
                draw.polygon(poly_coords, fill=255)

                # Apply alpha mask
                crop = crop.convert("RGBA")
                crop.putalpha(mask)

            class_id = getattr(ann, "class_id", None)
            conf = getattr(ann, "confidence", None)

            # Build filename
            parts = [file_prefix, f"idx{idx}"]
            if class_id is not None:
                parts.append(f"class{class_id}")
            if conf is not None:
                parts.append(f"conf{conf:.2f}")
            filename = "_".join(parts) + f".{ext}"

            save_path = out_dir / filename
            crop.save(save_path)
            saved_paths.append(str(save_path))

        return saved_paths
    def merge_containers(self, containers, contained):
        """
        Return a NEW AnnotationCollection where:
        - container annotations (class_id in `containers`)
          that contain at least one contained annotation (class_id in `contained`)
          have their polygon replaced with the union of:
                container polygon + contained polygons inside it
        - container holes removed via Polygon(exterior)
        - bounding boxes updated
        """

        # Deep-copy entire collection so nothing in the original is modified
        new_collection = copy.deepcopy(self)

        # Work on the copy
        new_annos = new_collection.annotations

        # 1. Collect container + contained lists
        container_annos = [a for a in new_annos if a.class_id in containers]
        contained_annos = [a for a in new_annos if a.class_id in contained]

        # 2. For each container, find which contained are inside
        for c in container_annos:
            inside_list = [
                x for x in contained_annos
                if c.bounding_box.contains(x.bounding_box)
            ]

            if not inside_list:
                continue  # Container has nothing inside

            # 3. Union container polygon + all inside polygons
            polys = [c.polygon] + [x.polygon for x in inside_list]
            union_geom = unary_union(polys)

            # 4. Remove holes if it's a single Polygon
            if union_geom.geom_type == "Polygon":
                new_poly = Polygon(union_geom.exterior)
            else:
                # MultiPolygon or mixed geometry → keep union as is
                new_poly = union_geom

            # 5. Update container polygon
            c.polygon = new_poly

            # 6. Update container bounding box
            minx, miny, maxx, maxy = new_poly.bounds
            c.bounding_box = BoundingBox(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

            # update obb
            obb_poly = new_poly.oriented_envelope
            obb_coords = tuple(list(obb_poly.exterior.coords)[:4])
            obb = OrientedBoundingBox(coords=obb_coords)
            c.oriented_bounding_box = obb

        return new_collection
    def write_annotations_to_txt(
        self, 
        output_path: str,
        include_conf: bool = True
    ) -> None:
        """
        Saves a list of ParsedAnnotationData to a YOLO-format .txt file.

        Args:
            annotations (List[ParsedAnnotationData]): List of parsed and filtered annotations.
            output_path (str): Path to save the .txt file.
            include_conf (bool): Whether to include confidence values (if available).
        """
        lines = []
        for ann in self.annotations:
            coords = ann.polygon.exterior.coords[:-1]  # drop last point (duplicate of first)
            flat = [f"{x:.6f} {y:.6f}" for x, y in coords]
            parts = [str(ann.class_id)] + flat
            if include_conf and ann.confidence is not None:
                parts.append(f"{ann.confidence:.4f}")
            lines.append(" ".join(parts))

        with open(output_path, 'w') as f:
            f.write("\n".join(lines))


from shapely import affinity


def denormalize_polygon(poly: Polygon, img_width: int, img_height: int) -> Polygon:
    """
    Scale a polygon from normalized [0,1] coords to pixel coords.

    Assumes:
        - x in [0,1] -> x * img_width
        - y in [0,1] -> y * img_height
        - origin at (0,0) in the top-left corner.
    """
    if poly.is_empty:
        return poly
    # scale relative to origin (0, 0)
    return affinity.scale(poly, xfact=img_width, yfact=img_height, origin=(0, 0))
