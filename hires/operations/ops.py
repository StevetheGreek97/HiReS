from __future__ import annotations

from pathlib import Path

from PIL import Image
from shapely import affinity
from shapely.geometry import Polygon

from hires.models import Annotation, Collection


def _parse_chunk_offsets(filename: str) -> tuple[int, int]:
    """Parse chunk offsets from a filename like ``image_0_1024.txt``."""
    stem = Path(filename).stem
    _, x_str, y_str = stem.rsplit("_", 2)
    return int(x_str), int(y_str)


def unify_collections(
    chunk_collections: dict[str, Collection],
    chunk_size: tuple[int, int],
    full_img_path: str,
) -> Collection:
    """
    Combine chunk-level collections into one full-image collection.

    Input polygons are assumed to be normalized in chunk space [0, 1] and the
    output polygons are normalized in the full-image coordinate system.
    """
    with Image.open(full_img_path) as image:
        full_width, full_height = image.size

    combined: list[Annotation] = []

    for filename, collection in chunk_collections.items():
        try:
            chunk_x, chunk_y = _parse_chunk_offsets(filename)
        except ValueError:
            continue

        # Compose the two linear steps (chunk-normalize → absolute → full-normalize)
        # into a single affine transform: x' = x * sx + tx, y' = y * sy + ty
        sx = chunk_size[0] / full_width
        sy = chunk_size[1] / full_height
        tx = chunk_x / full_width
        ty = chunk_y / full_height

        for annotation in collection.annotations:
            if annotation.polygon.is_empty:
                continue

            polygon = affinity.affine_transform(
                annotation.polygon,
                [sx, 0, 0, sy, tx, ty],
            )

            combined.append(
                Annotation(
                    class_id=annotation.class_id,
                    polygon=polygon,
                    confidence=annotation.confidence,
                )
            )

    return Collection(combined, collection_name=Path(full_img_path).stem)
