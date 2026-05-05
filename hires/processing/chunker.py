from __future__ import annotations

import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial


class ImageChunker:
    """Split large images into smaller overlapping chunks and save to disk."""

    def __init__(self, input_path):
        self.input_path = input_path
        self.extensions = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

    def _get_chunk_positions(self, width: int, height: int, chunk_size: tuple, overlap: int) -> tuple[list[int], list[int]]:
        chunk_w, chunk_h = chunk_size
        stride_w = chunk_w - overlap
        stride_h = chunk_h - overlap

        if stride_w <= 0 or stride_h <= 0:
            raise ValueError("Overlap must be smaller than chunk dimensions")

        x_positions = []
        y_positions = []

        x = 0
        while x < width:
            x_positions.append(x)
            x += stride_w
            if x + chunk_w > width:
                x_positions.append(x)
                break

        y = 0
        while y + chunk_h < height:
            y_positions.append(y)
            y += stride_h
            if y + chunk_h > height:
                y_positions.append(y)
                break

        return x_positions, y_positions

    def _chunk_and_save(self, image_array: np.ndarray, output_dir: str, base_filename: str,
                        chunk_size: tuple, overlap: int):
        os.makedirs(output_dir, exist_ok=True)
        height, width = image_array.shape[:2]
        chunk_w, chunk_h = chunk_size
        x_positions, y_positions = self._get_chunk_positions(width, height, chunk_size, overlap)

        for x in x_positions:
            for y in y_positions:
                chunk = image_array[y:y+chunk_h, x:x+chunk_w]

                if chunk.ndim == 2:
                    padded_chunk = np.zeros((chunk_h, chunk_w), dtype=chunk.dtype)
                    padded_chunk[:chunk.shape[0], :chunk.shape[1]] = chunk
                elif chunk.ndim == 3:
                    padded_chunk = np.zeros((chunk_h, chunk_w, chunk.shape[2]), dtype=chunk.dtype)
                    padded_chunk[:chunk.shape[0], :chunk.shape[1], :chunk.shape[2]] = chunk

                save_path = os.path.join(output_dir, f"{base_filename}_{x}_{y}.png")
                cv2.imwrite(save_path, padded_chunk)

    def _process_single_image(self, image_path: str, save_folder: str, chunk_size: tuple, overlap: int):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Skipping corrupt or unreadable image: {image_path}")
            return

        self._chunk_and_save(image, save_folder, base_name, chunk_size, overlap)

    def slice(self, save_folder: str, chunk_size: tuple = (1024, 1024), overlap: int = 150):
        if os.path.isfile(self.input_path):
            if self.input_path.lower().endswith(self.extensions):
                self._process_single_image(self.input_path, save_folder, chunk_size, overlap)
            else:
                print(f"Unsupported image format: {self.input_path}")

        elif os.path.isdir(self.input_path):
            image_files = [
                os.path.join(self.input_path, f)
                for f in os.listdir(self.input_path)
                if f.lower().endswith(self.extensions)
            ]
            if not image_files:
                print("No supported image files found.")
                return

            with Pool(cpu_count()) as pool:
                func = partial(self._process_single_image, save_folder=save_folder, chunk_size=chunk_size, overlap=overlap)
                list(tqdm(pool.imap(func, image_files), total=len(image_files), desc="Processing images"))

        else:
            print(f"Invalid input path: {self.input_path}")


class AnnotationChunker:
    """Split full-image YOLO segmentation txt files into per-chunk txt files.

    Mirrors :class:`ImageChunker` exactly.  The only extra parameter is
    ``image_size=(width, height)`` on :meth:`slice`, which is needed to
    convert normalised polygon coordinates to pixel space and back.

    Chunk file names follow the same ``{stem}_{x}_{y}.txt`` convention as
    :class:`ImageChunker` so the two can be paired directly.
    """

    def __init__(self, input_path):
        self.input_path = input_path

    def _get_chunk_positions(self, width: int, height: int, chunk_size: tuple, overlap: int) -> tuple[list[int], list[int]]:
        chunk_w, chunk_h = chunk_size
        stride_w = chunk_w - overlap
        stride_h = chunk_h - overlap

        if stride_w <= 0 or stride_h <= 0:
            raise ValueError("Overlap must be smaller than chunk dimensions")

        x_positions = []
        x = 0
        while x < width:
            x_positions.append(x)
            x += stride_w
            if x + chunk_w > width:
                x_positions.append(x)
                break

        y_positions = []
        y = 0
        while y + chunk_h < height:
            y_positions.append(y)
            y += stride_h
            if y + chunk_h > height:
                y_positions.append(y)
                break

        return x_positions, y_positions

    @staticmethod
    def _parse_line(line: str) -> tuple[int, list[float], float | None] | None:
        """Parse one YOLO annotation line.

        Format: ``class_id x1 y1 x2 y2 ... xn yn [conf]``
        Confidence is a trailing token that makes the value count odd.
        """
        parts = line.strip().split()
        if len(parts) < 7:  # class_id + at least 3 xy pairs
            return None
        class_id = int(parts[0])
        values = [float(v) for v in parts[1:]]
        if len(values) % 2 == 1:   # trailing confidence
            return class_id, values[:-1], values[-1]
        return class_id, values, None

    @staticmethod
    def _in_chunk(coords: list[float], x_start: int, y_start: int,
                  chunk_w: int, chunk_h: int, img_w: int, img_h: int) -> bool:
        """True if the polygon bounding box overlaps with the chunk region."""
        px = [c * img_w for c in coords[0::2]]
        py = [c * img_h for c in coords[1::2]]
        return (
            max(px) > x_start and min(px) < x_start + chunk_w
            and max(py) > y_start and min(py) < y_start + chunk_h
        )

    @staticmethod
    def _transform_coords(coords: list[float], x_start: int, y_start: int,
                          chunk_w: int, chunk_h: int, img_w: int, img_h: int) -> list[float]:
        """Shift polygon from full-image normalised space to chunk normalised space."""
        xs = [(c * img_w - x_start) / chunk_w for c in coords[0::2]]
        ys = [(c * img_h - y_start) / chunk_h for c in coords[1::2]]
        out: list[float] = []
        for x, y in zip(xs, ys):
            out.extend([x, y])
        return out

    def _chunk_and_save(self, txt_path: str, output_dir: str, base_filename: str,
                        chunk_size: tuple, overlap: int, image_size: tuple[int, int]):
        img_w, img_h = image_size
        chunk_w, chunk_h = chunk_size
        os.makedirs(output_dir, exist_ok=True)
        x_positions, y_positions = self._get_chunk_positions(img_w, img_h, chunk_size, overlap)

        annotations: list[tuple[int, list[float], float | None]] = []
        with open(txt_path) as fh:
            for line in fh:
                parsed = self._parse_line(line)
                if parsed is not None:
                    annotations.append(parsed)

        for x in x_positions:
            for y in y_positions:
                lines_out: list[str] = []
                for class_id, coords, conf in annotations:
                    if not self._in_chunk(coords, x, y, chunk_w, chunk_h, img_w, img_h):
                        continue
                    transformed = self._transform_coords(coords, x, y, chunk_w, chunk_h, img_w, img_h)
                    coord_str = " ".join(f"{v:.6f}" for v in transformed)
                    if conf is not None:
                        lines_out.append(f"{class_id} {coord_str} {conf:.4f}")
                    else:
                        lines_out.append(f"{class_id} {coord_str}")

                save_path = os.path.join(output_dir, f"{base_filename}_{x}_{y}.txt")
                with open(save_path, "w") as fh:
                    if lines_out:
                        fh.write("\n".join(lines_out) + "\n")

    def _process_single_txt(self, txt_path: str, save_folder: str,
                             chunk_size: tuple, overlap: int, image_size: tuple[int, int]):
        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        self._chunk_and_save(txt_path, save_folder, base_name, chunk_size, overlap, image_size)

    def slice(self, save_folder: str, chunk_size: tuple = (1024, 1024),
              overlap: int = 150, image_size: tuple[int, int] | None = None):
        """Chunk annotation files.

        Parameters
        ----------
        save_folder:
            Directory where per-chunk txt files are written.
        chunk_size:
            (width, height) of each chunk in pixels — must match the value
            used by :class:`ImageChunker`.
        overlap:
            Overlap in pixels — must match :class:`ImageChunker`.
        image_size:
            **(width, height) of the full image in pixels.**  Required because
            txt files do not embed image dimensions.
        """
        if image_size is None:
            raise ValueError("image_size=(width, height) is required to chunk annotation coordinates.")

        if os.path.isfile(self.input_path):
            if self.input_path.lower().endswith(".txt"):
                self._process_single_txt(self.input_path, save_folder, chunk_size, overlap, image_size)
            else:
                print(f"Unsupported format: {self.input_path}")

        elif os.path.isdir(self.input_path):
            txt_files = [
                os.path.join(self.input_path, f)
                for f in os.listdir(self.input_path)
                if f.lower().endswith(".txt")
            ]
            if not txt_files:
                print("No txt files found.")
                return

            with Pool(cpu_count()) as pool:
                func = partial(self._process_single_txt, save_folder=save_folder,
                               chunk_size=chunk_size, overlap=overlap, image_size=image_size)
                list(tqdm(pool.imap(func, txt_files), total=len(txt_files), desc="Processing annotations"))

        else:
            print(f"Invalid input path: {self.input_path}")


def main():
    parser = argparse.ArgumentParser(description="Image Chunker CLI")
    parser.add_argument('-i', '--input_path', type=str, required=True)
    parser.add_argument('-s', '--save_folder', type=str, required=True)
    parser.add_argument('-d', '--chunk_size', type=int, nargs=2, default=(1024, 1024))
    parser.add_argument('-o', '--overlap', type=int, default=150)
    args = parser.parse_args()
    ImageChunker(args.input_path).slice(
        save_folder=args.save_folder,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()

