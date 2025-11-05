from shapely.geometry import box
from typing import List, Tuple
from .datatypes import ParsedAnnotationData
import os
from PIL import Image
from .parser import AnnotationParser
from ..ios.writer import write_annotations_to_txt

def filter_touching_edges(
    annotations: List[ParsedAnnotationData],
    threshold: float = 1e-4
) -> List[ParsedAnnotationData]:
    """
    Removes polygons that touch or cross the image edges (normalized [0, 1]),
    using the actual polygon geometry instead of its bounding box.

    Args:
        annotations (List[ParsedAnnotationData]): Parsed annotations.
        threshold (float): Small inward offset to avoid floating-point errors.

    Returns:
        List[ParsedAnnotationData]: Polygons fully contained inside the image.
    """
    image_box = box(0.0, 0.0, 1.0, 1.0)
    safe_box = image_box.buffer(-threshold)  # slightly inset boundary

    filtered = []
    for ann in annotations:
        poly = ann.polygon
        if not poly.is_valid or poly.is_empty:
            continue
        # Keep polygon only if it's fully inside (not touching/crossing edges)
        if safe_box.contains(poly):
            filtered.append(ann)

    return filtered


def unify(
    annotation_dir: str,
    output_txt_path: str,
    chunk_size: Tuple[int, int],
    full_img_path: str
) -> None:
    """
    Combine all YOLO segmentation .txt files from chunks into one file using AnnotationParser.
    Based on the chunk's name (e.g., image_0_1024.jpg), the polygon coords are transformed!

    Args:
        annotation_dir (str): Directory with YOLO .txt chunk annotations.
        output_txt_path (str): Path to save the combined annotation file.
        chunk_size (Tuple[int, int]): (width, height) of each chunk in pixels.
        full_img_size (Tuple[int, int]): (width, height) of the full image in pixels.
    """
    # Get image dimensions
    with Image.open(full_img_path) as img:
        full_img_size = img.size  # (width, height)
    combined_annotations: List[ParsedAnnotationData] = []

    for filename in os.listdir(annotation_dir):
        if not filename.endswith(".txt"):
            continue

        txt_path = os.path.join(annotation_dir, filename)
        if os.stat(txt_path).st_size == 0:
            #print(f"Skipping empty file: {filename}")
            continue

        try:
            _, x_str, y_str = os.path.splitext(filename)[0].rsplit("_", 2)
            chunk_x = int(x_str)
            chunk_y = int(y_str)
        except ValueError:
            print(f"Skipping invalid filename format: {filename}")
            continue

        try:
            parser = AnnotationParser(txt_path)
        except FileNotFoundError:
            continue

        for ann in parser.parse():
            poly = ann.polygon
            abs_coords = [(x * chunk_size[0] + chunk_x, y * chunk_size[1] + chunk_y) for x, y in poly.exterior.coords[:-1]]
            rel_coords = [(x / full_img_size[0], y / full_img_size[1]) for x, y in abs_coords]
            ann.polygon = type(poly)(rel_coords)  # replace polygon with transformed one
            combined_annotations.append(ann)

    write_annotations_to_txt(combined_annotations, output_txt_path)
