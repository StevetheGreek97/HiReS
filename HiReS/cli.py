"""
HiReS Command-Line Interface (CLI)
==================================

This module provides the command-line interface for **HiReS**, a
high-resolution image segmentation and analysis toolkit built on YOLO-based
instance segmentation. It allows researchers and developers to execute key
parts of the HiReS pipeline directly from the terminal.

Usage:
    hires <subcommand> [options]

Available subcommands:
    hires chunk   Split a large image into smaller tiles
    hires run     Execute the full segmentation pipeline (image or directory)
    hires plot    Render segmentation overlays for visualization

Examples:
    hires chunk --image raw.tif --out chunks/ --chunk-size 1024 1024 --overlap 150
    hires run --image data/ --model models/DaphnAI.pt --out results/ --workers 4
    hires plot --image raw.tif --ann results/raw.txt --out overlay.png

For detailed documentation, visit:
https://github.com/<your-username>/HiReS
"""

import argparse
import sys
from HiReS.config import Settings
from HiReS.pipeline import Pipeline, setup_logging
from HiReS.ios.chunker import ImageChunker
from HiReS.ios.plotting import SegmentationPlotter


# ---------------------------------------------------------------------
# Command Implementations
# ---------------------------------------------------------------------
def cmd_chunk(args):
    """Slice a large image into tiles for processing."""
    chunk_w, chunk_h = args.chunk_size
    ImageChunker(args.image).slice(
        save_folder=args.out,
        chunk_size=(chunk_w, chunk_h),
        overlap=args.overlap,
    )


def cmd_plot(args):
    """Render and save segmentation overlay."""
    plotter = SegmentationPlotter(args.model) if args.model else SegmentationPlotter(None)
    plotter.plot_annotations(
        image_path=args.image,
        txt_path=args.ann,
        save=args.out,
    )


def cmd_run(args):
    """Execute the full segmentation pipeline."""
    cfg = Settings(
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        chunk_size=tuple(args.chunk_size),
        overlap=args.overlap,
        edge_threshold=args.edge_thr,
        iou_thresh=args.iou_thr,
    )

    setup_logging()
    Pipeline(cfg).run(
        input_path=args.image,
        model_path=args.model,
        output_dir=args.out,
        workers=args.workers,
    )


# Argument Parser Construction
def build_parser():
    """Build and return the top-level CLI parser."""
    ap = argparse.ArgumentParser(
        prog="hires",
        description=(
            "HiReS — High-Resolution Image Segmentation Pipeline\n\n"
            "Run advanced image segmentation workflows directly from the command line.\n"
            "Use YOLO-based segmentation, automatic chunking, polygon filtering, NMS, and visualization.\n\n"
            "Available commands:\n"
            "  hires chunk    Split a large image into chunks for segmentation\n"
            "  hires run      Execute the full segmentation pipeline on an image or directory\n"
            "  hires plot     Render and save segmentation overlays for visualization\n\n"
            "Examples:\n"
            "  hires chunk --image raw.tif --out chunks/ --chunk-size 1024 1024 --overlap 150\n"
            "  hires run --image data/ --model models/DaphnAI.pt --out results/ --workers 4\n"
            "  hires plot --image raw.tif --ann results/raw.txt --out overlay.png\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = ap.add_subparsers(dest="command")


    # hires chunk
    p_chunk = sub.add_parser("chunk", help="Split an image into tiles")
    p_chunk.add_argument("--image", required=True, help="Path to a single image")
    p_chunk.add_argument("--out", required=True, help="Directory to save chunks")
    p_chunk.add_argument(
        "--chunk-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(1024, 1024),
        help="Chunk size as two integers (default: 1024 1024)",
    )
    p_chunk.add_argument("--overlap", type=int, default=150, help="Overlap in pixels")
    p_chunk.set_defaults(func=cmd_chunk)


    # hires plot
    p_plot = sub.add_parser("plot", help="Overlay polygons on an image")
    p_plot.add_argument("--image", required=True, help="Path to the image")
    p_plot.add_argument("--ann", required=True, help="YOLO .txt annotations")
    p_plot.add_argument("--out", required=True, help="Output image path with extention")
    p_plot.add_argument("--model", default=None, help="Optional model for color mapping")
    p_plot.set_defaults(func=cmd_plot)

    # hires run
    p_run = sub.add_parser("run", help="Run full segmentation pipeline")
    p_run.add_argument("--image", required=True, help="Image file or directory")
    p_run.add_argument("--model", required=True, help="Path to YOLO model (.pt)")
    p_run.add_argument("--out", required=True, help="Output directory")
    p_run.add_argument("--conf", type=float, default=0.5, help="Model confidence threshold")
    p_run.add_argument("--imgsz", type=int, default=1024, help="Inference image size")
    p_run.add_argument("--device", default="cpu", help="Device: 'cpu' or 'cuda:0'")
    p_run.add_argument(
        "--chunk-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(1024, 1024),
        help="Chunk size as two integers (default: 1024 1024)",
    )
    p_run.add_argument("--overlap", type=int, default=150, help="Overlap in pixels")
    p_run.add_argument("--edge-thr", type=float, default=1e-2, help="Edge threshold for filtering polygons")
    p_run.add_argument("--iou-thr", type=float, default=0.7, help="IoU threshold for polygon NMS")
    p_run.add_argument("--workers", type=int, default=1, help="Parallel workers for directory input")
    p_run.set_defaults(func=cmd_run)

    return ap


def main():
    ap = build_parser()

    # Show help if no arguments were provided
    if len(sys.argv) == 1:
        ap.print_help(sys.stderr)
        sys.exit(0)

    args = ap.parse_args()

    # Run the corresponding subcommand
    if hasattr(args, "func"):
        args.func(args)
    else:
        ap.print_help(sys.stderr)


if __name__ == "__main__":
    main()
