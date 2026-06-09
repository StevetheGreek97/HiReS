from __future__ import annotations

import os

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Iterable, Iterator, List, Callable
from pathlib import Path

from shapely.strtree import STRtree

#from .eval.iou import IoUMatrix

#from .eval.match_maker import MatchMaker

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .annotation import Annotation


def _read_image_size(path: "str | os.PathLike[str]") -> "tuple[int | None, int | None]":
    """Return (width, height) from an image file by reading only its header."""
    try:
        from PIL import Image as _Image
        with _Image.open(path) as img:
            return img.size
    except Exception:
        return None, None


@dataclass
class Collection:
    """Ordered container for annotations plus query and processing helpers."""

    annotations: List[Annotation] = field(default_factory=list)
    collection_name: str | None = None
    image_path: str | os.PathLike[str] | None = None
    dpi: float | None = None
    unit: str | None = None
    image_width: int | None = None
    image_height: int | None = None

    def __len__(self) -> int:
        return len(self.annotations)

    def __iter__(self) -> Iterator[Annotation]:
        return iter(self.annotations)

    def __getitem__(self, idx: int | slice) -> Annotation | list[Annotation]:
        return self.annotations[idx]

    def add(self, annotation: Annotation) -> None:
        self.annotations.append(annotation)

    def extend(self, anns: Iterable[Annotation]) -> None:
        self.annotations.extend(anns)

    @property
    def class_counts(self) -> dict[int, int]:
        counts = Counter(annotation.class_id for annotation in self.annotations)
        return dict(sorted(counts.items()))
    
    @classmethod
    def read_txt(
        cls,
        txt_path: str | os.PathLike[str],
        collection_name: str | None = None,
        image_path: str | os.PathLike[str] | None = None,
    ) -> "Collection":
        from .parser import AnnotationParser

        parser = AnnotationParser(txt_path)
        parsed = parser.parse()

        name = collection_name or Path(txt_path).stem

        image_width, image_height = _read_image_size(image_path) if image_path is not None else (None, None)

        return cls(
            annotations=parsed.annotations,
            collection_name=name,
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
        )

    def set_scale(self, dpi: float | None = None, unit: str | None = None) -> None:
        """Apply the same scale settings to all annotations in the collection."""
        if self.image_width is None and self.image_height is None and self.image_path is not None:
            self.image_width, self.image_height = _read_image_size(self.image_path)

        if self.image_width is None or self.image_height is None:
            import warnings
            warnings.warn(
                "image dimensions not available; scale not applied. "
                "Pass image_dir when loading the Album/Collection.",
                UserWarning,
                stacklevel=2,
            )
            return

        self.dpi = dpi
        self.unit = unit
        for annotation in self.annotations:
            annotation.set_scale(
                dpi=dpi,
                unit=unit,
                image_width=self.image_width,
                image_height=self.image_height,
            )

    def filter(
        self,
        by: str | None = None,
        value: Any = None,
        condition: Callable[[Any], bool] | None = None,
        predicate: Callable[[Annotation], bool] | None = None,
    ) -> "Collection":
        """
        Filter annotations.

        Use ONE of:
        - (by + value)
        - (by + condition)
        - predicate (function on Annotation)
        """
        if predicate is not None:
            filtered = [ann for ann in self.annotations if predicate(ann)]
            return Collection(
                annotations=filtered,
                collection_name=f"{self.collection_name}_filtered" if self.collection_name else None,
                dpi=self.dpi,
                unit=self.unit,
                image_width=self.image_width,
                image_height=self.image_height,
            )

        if by is None:
            raise ValueError("Provide 'by' or 'predicate'.")

        if value is None and condition is None:
            raise ValueError("Provide either 'value' or 'condition'.")

        if value is not None and condition is not None:
            raise ValueError("Provide only one of 'value' or 'condition'.")

        filtered: list[Annotation] = []
        for ann in self.annotations:
            if not hasattr(ann, by):
                raise AttributeError(f"Annotation has no attribute '{by}'.")

            attr_value = getattr(ann, by)

            if value is not None:
                if attr_value == value:
                    filtered.append(ann)
            else:
                if condition(attr_value):
                    filtered.append(ann)

        return Collection(
            annotations=filtered,
            collection_name=f"{self.collection_name}_filtered_by_{by}" if self.collection_name else None,
            dpi=self.dpi,
            unit=self.unit,
            image_path=self.image_path,
            image_width=self.image_width,
            image_height=self.image_height,
        )

    def to_records(self) -> list[dict]:
        """
        Return a row-wise representation (one dict per annotation).
        """
        records = []

        for ann in self.annotations:
            row = {
                "collection_name": self.collection_name,
                "image_path": str(self.image_path) if self.image_path else None,
                "image_width": self.image_width,
                "image_height": self.image_height,
                "dpi": self.dpi,
                "unit": self.unit,
                **ann.to_dict(),
            }
            records.append(row)

        return records

    def to_df(self):
        """
        Return annotations as a pandas DataFrame (one row per annotation).
        """
        import pandas as pd
        return pd.DataFrame(self.to_records())

    def to_csv(
        self,
        path: str | os.PathLike[str],
        *,
        index: bool = False,
        **kwargs,
    ) -> None:
        """
        Save annotations as a CSV file (one row per annotation).
        """
        df = self.to_df()
        df.to_csv(path, index=index, **kwargs)
    
    def to_txt(
        self,
        output_path: str | os.PathLike[str],
        *,
        include_conf: bool = True,
    ) -> None:
        """
        Write annotations to a YOLO segmentation-style txt file.

        Format per line:
            class_id x1 y1 x2 y2 ... xn yn [conf]

        Notes
        -----
        - Assumes polygons are stored in normalized coordinates.
        - The closing coordinate is omitted if the polygon exterior is closed
        (Shapely repeats the first point at the end).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []

        for ann in self.annotations:
            if ann.polygon.is_empty:
                continue

            coords = list(ann.polygon.exterior.coords)

            # remove duplicated closing point
            if len(coords) >= 2 and coords[0] == coords[-1]:
                coords = coords[:-1]

            if len(coords) < 3:
                continue

            flat_coords = [f"{x:.6f} {y:.6f}" for x, y in coords]
            line = f"{ann.class_id} " + " ".join(flat_coords)

            if include_conf and ann.confidence is not None:
                line += f" {ann.confidence:.6f}"

            lines.append(line)

        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    
    def remap_classes(
        self,
        mapping: "ClassMapping | dict[int, int]",
        resolve: dict[str, str] | None = None,
    ) -> "Collection":
        """Return a new Collection with class_ids remapped according to mapping.

        Parameters
        ----------
        mapping : ClassMapping (from build_class_mapping()) or a plain {old_id: new_id} dict.
        resolve : {old_label: chosen_new_label} — required for any ambiguous (list) entry.
            e.g. {'Daphnia': 'S_vetulus'}
            Only used when mapping is a ClassMapping.

        Keys absent from a plain dict mapping are left unchanged.
        """
        from .utils import ClassMapping
        if isinstance(mapping, ClassMapping):
            flat: dict[int, int] = mapping.flatten(resolve or {})
        else:
            flat = mapping  # type: ignore[assignment]

        new_anns = [
            replace(ann, class_id=flat.get(ann.class_id, ann.class_id))
            for ann in self.annotations
        ]
        return Collection(
            annotations=new_anns,
            collection_name=self.collection_name,
            image_path=self.image_path,
            dpi=self.dpi,
            unit=self.unit,
            image_width=self.image_width,
            image_height=self.image_height,
        )

    def nms(
        self,
        iou_threshold: float = 0.5,
        *,
        class_aware: bool = False,
    ) -> "Collection":
        """
        Apply non-maximum suppression to the collection.

        Parameters
        ----------
        iou_threshold : float
            Suppress annotations with IoU > threshold.
        class_aware : bool
            If True, suppress only annotations of the same class_id.

        Returns
        -------
        Collection
            A new collection containing only the kept annotations.
        """
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")

        if not self.annotations:
            return Collection(
                annotations=[],
                collection_name=self.collection_name,
                dpi=self.dpi,
                unit=self.unit,
                image_path=self.image_path,
                image_width=self.image_width,
                image_height=self.image_height,
            )

        for ann in self.annotations:
            if ann.confidence is None:
                raise ValueError("All annotations must have confidence for NMS.")

        annotations = sorted(
            self.annotations,
            # if confidence is the same we keep the one with larger area 
            key=lambda ann: (ann.confidence, ann.area),
            reverse=True,
        )

        polygons = [ann.polygon for ann in annotations]
        tree = STRtree(polygons)

        suppressed = set()
        kept: list[Annotation] = []

        for i, ann in enumerate(annotations):
            if i in suppressed:
                continue

            kept.append(ann)

            candidate_idxs = tree.query(ann.polygon)

            for idx in candidate_idxs:
                j = int(idx)

                if j == i or j in suppressed:
                    continue

                other = annotations[j]

                if class_aware and ann.class_id != other.class_id:
                    continue

                if ann.iou(other) > iou_threshold:
                    suppressed.add(j)

        return Collection(
            annotations=kept,
            collection_name=f"{self.collection_name}_nms" if self.collection_name else None,
            dpi=self.dpi,
            unit=self.unit,
            image_path=self.image_path,
            image_width=self.image_width,
            image_height=self.image_height,
        )

    def save_crops(
        self,
        out_dir: str | os.PathLike[str],
        *,
        use_mask: bool = True,
        padding: int = 0,
        file_prefix: str | None = None,
        image_path: str | os.PathLike[str] | None = None,
        ext: str = "png",
    ) -> list[str]:
        """
        Crop all annotations from the image and save them.

        Returns list of saved file paths.
        """
        import math
        import cv2
        import numpy as np
        from pathlib import Path
        from shapely import affinity

        self.image_path = image_path or self.image_path 

        if self.image_path is None and image_path is None:
            raise ValueError("image_path is not set on the collection")

        image = cv2.imread(str(self.image_path or image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {self.image_path or image_path}")

        # convert BGR → RGB for consistency (optional)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        h, w = image.shape[:2]
        prefix = file_prefix or self.collection_name or "crop"
        ext = ext.lstrip(".")

        saved_paths: list[str] = []

        for i, ann in enumerate(self.annotations):

            polygon = ann.polygon

            # Denormalize
            polygon = affinity.scale(polygon, xfact=w, yfact=h, origin=(0, 0))

            if polygon.is_empty or not polygon.is_valid:
                continue

            minx, miny, maxx, maxy = polygon.bounds

            minx = max(0, int(math.floor(minx)) - padding)
            miny = max(0, int(math.floor(miny)) - padding)
            maxx = min(w, int(math.ceil(maxx)) + padding)
            maxy = min(h, int(math.ceil(maxy)) + padding)

            if minx >= maxx or miny >= maxy:
                continue

            crop = image[miny:maxy, minx:maxx].copy()

            if use_mask:
                coords = np.asarray(polygon.exterior.coords, dtype=np.float32)
                coords[:, 0] -= minx
                coords[:, 1] -= miny
                coords = np.round(coords).astype(np.int32)

                mask = np.zeros(crop.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [coords], 255)

                if crop.ndim == 2:
                    crop = np.where(mask > 0, crop, 0)
                else:
                    crop = np.where(mask[..., None] > 0, crop, 0)

            filename = f"{prefix}_{i:04d}_class_{ann.class_id}.{ext}"
            save_path = out_dir / filename

            # convert back to BGR for saving
            crop_to_save = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

            ok = cv2.imwrite(str(save_path), crop_to_save)
            if not ok:
                raise IOError(f"Failed to save {save_path}")

            saved_paths.append(str(save_path))

        return saved_paths
