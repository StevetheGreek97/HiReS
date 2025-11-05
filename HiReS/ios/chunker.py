"""
image_chunker.py
================

Chunk large images into overlapping tiles
-----------------------------------------
Utilities to split very large images into smaller, overlapping chunks and save
them to disk. Supports single-file and directory inputs, automatic edge padding,
and parallel processing via multiprocessing.

Typical use cases:
- Preparing tiled inputs for segmentation/detection models.
- Speeding up batch inference by pre-chunking large TIFFs.
- Creating training tiles with controllable overlap.

Quick start (Python API):
-------------------------
>>> from chucker import ImageChunker
>>> chunker = ImageChunker("/path/to/image_or_folder")
>>> chunker.slice(save_folder="/path/to/output_chunks",
...               chunk_size=(1024, 1024),
...               overlap=150)

Command-line usage:
-------------------
# Single image
$ python image_chunker.py -i /path/to/image.tif -s /out/chunks -d 1024 1024 -o 150

# Folder of images
$ python image_chunker.py -i /path/to/folder -s /out/chunks -d 1024 1024 -o 150

Main components:
----------------
- ImageChunker: High-level class to compute chunk coordinates and write tiles.

Functions & methods of interest:
--------------------------------
- ImageChunker.slice(save_folder, chunk_size=(W,H), overlap=px)
- ImageChunker._get_chunk_positions(width, height, chunk_size, overlap)
- ImageChunker._process_single_image(image_path, save_folder, chunk_size, overlap)
- ImageChunker._chunk_and_save(image_array, output_dir, base_filename, chunk_size, overlap)

Attributes:
-----------
- Supported extensions: .tif, .tiff, .png, .jpg

Notes:
------
- Overlap must be strictly smaller than both chunk width and height.
- Grayscale and multi-channel images are supported; edges are zero-padded.

Dependencies:
-------------
- opencv-python (cv2)
- numpy
- tqdm

Version & metadata:
-------------------
Version        : 1.0.0
Author         : Stylianos Mavrianos
Email          : stylianosmavrianos@gmail.com
Created        : 2025-10-13
Updated        : 2025-10-13
License        : MIT
Python Version : 3.10+

References:
-----------
- DaphnAI: doi.org/10.1101/2025.07.30.667622 


Changelog:
----------
v1.0.0 - Initial public version with CLI and multiprocessing.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

class ImageChunker:
    """
    A utility class for splitting large images into smaller overlapping chunks
    and saving them to disk. It supports processing single image files or directories
    of images, and includes automatic padding of image borders where necessary.

    The class uses OpenCV for image manipulation and multiprocessing to parallelize
    chunking over multiple images.

    Supported image formats: .tif, .tiff, .png

    Attributes:
        input_path (str): Path to a single image file or a directory of images.
        extensions (tuple): Valid image file extensions.

    Examples:
        >>> from chucker import ImageChunker

        >>> input_path = "/path/to/image_or_folder"
        >>> save_folder = "/path/to/output_chunks"

        >>> chunker = ImageChunker(input_path)
        >>> chunker.slice(save_folder=save_folder, chunk_size=(1024, 1024), overlap=150)
    """

    def __init__(self, input_path):
        """
        Initialize the ImageChunker.

        Args:
            input_path (str): Path to a single image or a folder containing images.
        """
        self.input_path = input_path
        self.extensions = (".tif", ".tiff", ".png", ".jpg")

    def _get_chunk_positions(self, width: int, height: int, chunk_size: tuple, overlap: int) -> tuple[list[int], list[int]]:
        """
        Calculate the top-left (x, y) coordinates of all chunks to extract from the image.

        Args:
            width (int): Width of the image.
            height (int): Height of the image.
            chunk_size (tuple): Size of each chunk (width, height).
            overlap (int): Overlap (in pixels) between adjacent chunks.

        Returns:
            tuple: Two lists containing x and y positions respectively.
        
        Raises:
            ValueError: If overlap is larger than chunk size.
        """
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
        """
        Slice an image array into chunks and save them, with padding where needed.

        Args:
            image_array (np.ndarray): The original image array.
            output_dir (str): Path to save the chunks.
            base_filename (str): Base filename for each saved chunk.
            chunk_size (tuple): Size (width, height) of each chunk.
            overlap (int): Overlap in pixels between chunks.
        """
        os.makedirs(output_dir, exist_ok=True)
        height, width = image_array.shape[:2]
        chunk_w, chunk_h = chunk_size
        x_positions, y_positions = self._get_chunk_positions(width, height, chunk_size, overlap)

        for x in x_positions:
            for y in y_positions:
                chunk = image_array[y:y+chunk_h, x:x+chunk_w]

                if chunk.ndim == 2:
                    # Grayscale (8-bit or 16-bit)
                    padded_chunk = np.zeros((chunk_h, chunk_w), dtype=chunk.dtype)
                    padded_chunk[:chunk.shape[0], :chunk.shape[1]] = chunk
                    ext = "tif" if chunk.dtype == np.uint16 else "png"

                elif chunk.ndim == 3:
                    # Color or multispectral image
                    padded_chunk = np.zeros((chunk_h, chunk_w, chunk.shape[2]), dtype=chunk.dtype)
                    padded_chunk[:chunk.shape[0], :chunk.shape[1], :chunk.shape[2]] = chunk
                    ext = "png"

                save_path = os.path.join(output_dir, f"{base_filename}_{x}_{y}.png")
                cv2.imwrite(save_path, padded_chunk)

    def _process_single_image(self, image_path: str, save_folder: str, chunk_size: tuple, overlap: int):
        """
        Process a single image: load, chunk, and save to disk.
        Skips image if it's unreadable (e.g. corrupt or unsupported format).
        """
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"❌ Skipping corrupt or unreadable image: {image_path}")
            return

        self._chunk_and_save(image, save_folder, base_name, chunk_size, overlap)


    def slice(self, save_folder: str, chunk_size: tuple = (1024, 1024), overlap: int = 150):
        """
        Process one or more images to create and save overlapping chunks.

        Supports multiprocessing for batch processing when input_path is a directory.

        Args:
            save_folder (str): Destination directory for image chunks.
            chunk_size (tuple, optional): Size of each chunk. Defaults to (1024, 1024).
            overlap (int, optional): Overlap in pixels. Defaults to 150.

        """

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

def main():
    parser = argparse.ArgumentParser(description="Image Chunker CLI")
    parser.add_argument('-i', '--input_path', type=str, required=True, help="Path to image or folder of images")
    parser.add_argument('-s', '--save_folder', type=str, required=True, help="Directory to save image chunks")
    parser.add_argument('-d', '--chunk_size', type=int,nargs=2, default=(1024, 1024), help="Chunk width")
    parser.add_argument('-o', '--overlap', type=int, default=150, help="Overlap in pixels between chunks")

    args = parser.parse_args()

    chunker = ImageChunker(args.input_path)
    chunker.slice(save_folder=args.save_folder, chunk_size=args.chunk_size, overlap=args.overlap)

if __name__ == "__main__":
    main()