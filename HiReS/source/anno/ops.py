from shapely.geometry import box
from typing import List, Tuple
from .datatypes import Annotation
import os
from PIL import Image
from .parser import AnnotationParser
from ..ios.writer import write_annotations_to_txt
import os
from pathlib import Path
from typing import Dict, Tuple
from PIL import Image
from shapely.geometry import Polygon

# assumes these are already defined somewhere:
from HiReS.source.anno.datatypes import Annotation, AnnotationCollection


def _parse_chunk_offsets(filename: str) -> Tuple[int, int]:
    """
    Parse chunk offsets from a filename like: image_0_1024.txt or image_256_1024.txt

    Returns
    -------
    (chunk_x, chunk_y)
    """
    stem = Path(filename).stem
    # e.g. "image_0_1024" -> ["image", "0", "1024"]
    _, x_str, y_str = stem.rsplit("_", 2)
    return int(x_str), int(y_str)


def unify_collections(
    chunk_collections: Dict[str, AnnotationCollection],
    chunk_size: Tuple[int, int],
    full_img_path: str,
) -> AnnotationCollection:
    """
    Combine multiple chunk-level AnnotationCollections into one full-image collection.

    Parameters
    ----------
    chunk_collections : dict[str, AnnotationCollection]
        Mapping from chunk filename (e.g. "image_0_1024.txt") to its filtered annotations
        (with polygon coords still normalized in chunk space [0,1]).
    chunk_size : (int, int)
        (width, height) of each chunk in pixels.
    full_img_path : str
        Path to the full image (to get its size).

    Returns
    -------
    AnnotationCollection
        Unified annotations, polygons normalized in full-image coordinates [0,1].
    """
    with Image.open(full_img_path) as img:
        full_w, full_h = img.size

    combined: list[Annotation] = []

    for filename, coll in chunk_collections.items():
        try:
            chunk_x, chunk_y = _parse_chunk_offsets(filename)
        except ValueError:
            # skip filenames that don't match the pattern
            continue

        for ann in coll.annotations:
            poly = ann.polygon
            if poly.is_empty:
                continue

            # 1) chunk-normalized -> absolute pixel coords in full image
            abs_coords = [
                (
                    x * chunk_size[0] + chunk_x,
                    y * chunk_size[1] + chunk_y,
                )
                for x, y in poly.exterior.coords[:-1]  # drop closing point
            ]

            # 2) pixel coords -> full-image normalized [0,1]
            rel_coords = [
                (x / full_w, y / full_h)
                for x, y in abs_coords
            ]

            new_poly = Polygon(rel_coords)

            combined.append(
                Annotation(
                    class_id=ann.class_id,
                    polygon=new_poly,
                    confidence=ann.confidence,
                    # bboxes can be recomputed later if needed
                    bounding_box=None,
                    oriented_bounding_box=None,
                )
            )

    return AnnotationCollection(
        combined,
        collection_name=Path(full_img_path).stem,
    )
