from ..anno.datatypes import Annotation
from typing import List


def write_annotations_to_txt(
    annotations: List[Annotation],
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
    for ann in annotations:
        coords = ann.polygon.exterior.coords[:-1]  # drop last point (duplicate of first)
        flat = [f"{x:.6f} {y:.6f}" for x, y in coords]
        parts = [str(ann.class_id)] + flat
        if include_conf and ann.confidence is not None:
            parts.append(f"{ann.confidence:.4f}")
        lines.append(" ".join(parts))

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
