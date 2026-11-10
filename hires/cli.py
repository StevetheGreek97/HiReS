"""Command-line interface: ``hires run``, ``chunk``, ``plot`` and ``evaluate``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .annotation import UNITS
from .errors import HiresError

logger = logging.getLogger("hires")

DEFAULT_MODEL = "models/DaphnAI.pt"


class _Formatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


def _add_tiling_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("chunking")
    group.add_argument("--chunk-size", type=int, nargs=2, default=[1024, 1024],
                       metavar=("W", "H"), help="Chunk width and height in pixels")
    group.add_argument("--overlap", type=int, default=150, metavar="INT",
                       help="Overlap between adjacent chunks in pixels")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hires",
        formatter_class=_Formatter,
        description="HiReS - high-resolution segmentation toolkit",
        epilog="Run 'hires <command> --help' for per-command options.",
    )
    parser.add_argument("--version", action="version", version=f"hires {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    # ---- run ---------------------------------------------------------- #
    run = commands.add_parser(
        "run",
        formatter_class=_Formatter,
        help="Run the full segmentation pipeline on an image or directory",
        description=(
            "Run the full segmentation pipeline on a single image or a directory.\n"
            "Steps: chunk -> predict -> filter edges -> merge -> NMS -> overlay -> CSV."
        ),
        epilog=(
            "examples:\n"
            "  hires run -s image.tif -m model.pt\n"
            "  hires run -s images/ -o results/ --conf 0.4 --device 0\n"
            "  hires run -s images/ --chunk-size 2048 2048 --overlap 256 --iou-thresh 0.6\n"
            "  hires run -s images/ --dpi 1200 --unit um\n"
            "  hires run -s images/ -r --debug\n"
        ),
    )
    io = run.add_argument_group("input / output")
    io.add_argument("-s", "--source", required=True, metavar="PATH",
                    help="Image file or directory to process")
    io.add_argument("-m", "--model", default=DEFAULT_MODEL, metavar="PATH",
                    help="Path to YOLO model weights")
    io.add_argument("-o", "--output", default="results", metavar="DIR",
                    help="Root output directory")
    io.add_argument("-r", "--recursive", action="store_true",
                    help="Search source directory recursively")
    inference = run.add_argument_group("inference")
    inference.add_argument("--conf", type=float, default=0.5, metavar="FLOAT",
                           help="Detection confidence threshold")
    inference.add_argument("--imgsz", type=int, default=1024, metavar="INT",
                           help="Inference image size (pixels)")
    inference.add_argument("--device", default="cpu", metavar="STR",
                           help="Inference device: cpu, 0, cuda:0, ...")
    inference.add_argument("--batch-size", type=int, default=8, metavar="INT",
                           help="Number of chunks per model call")
    _add_tiling_options(run)
    post = run.add_argument_group("post-processing")
    post.add_argument("--edge-threshold", type=float, default=1e-2, metavar="FLOAT",
                      help="Normalised inset to remove edge-touching polygons")
    post.add_argument("--iou-thresh", type=float, default=0.7, metavar="FLOAT",
                      help="IoU threshold for polygon NMS")
    out = run.add_argument_group("outputs")
    out.add_argument("--save-crops", action="store_true",
                     help="Save a masked crop image for every detection")
    out.add_argument("--dpi", type=float, default=None, metavar="FLOAT",
                     help="Scan resolution in DPI; enables physical measurements")
    out.add_argument("--unit", default=None, choices=sorted(UNITS),
                     help="Physical unit for shape descriptors")
    out.add_argument("--debug", action="store_true",
                     help="Write intermediate files (chunks and raw predictions)")

    # ---- chunk -------------------------------------------------------- #
    chunk = commands.add_parser(
        "chunk",
        formatter_class=_Formatter,
        help="Slice images into overlapping chunks",
        description="Slice one image or every image in a directory into overlapping chunks.",
        epilog=(
            "examples:\n"
            "  hires chunk -s image.tif -o chunks/\n"
            "  hires chunk -s images/ --chunk-size 2048 2048 --overlap 256\n"
        ),
    )
    io = chunk.add_argument_group("input / output")
    io.add_argument("-s", "--source", required=True, metavar="PATH",
                    help="Image file or directory to chunk")
    io.add_argument("-o", "--output", default="chunks", metavar="DIR",
                    help="Output directory for chunk images")
    io.add_argument("-r", "--recursive", action="store_true",
                    help="Search source directory recursively")
    _add_tiling_options(chunk)

    # ---- plot --------------------------------------------------------- #
    plot = commands.add_parser(
        "plot",
        formatter_class=_Formatter,
        help="Overlay existing annotations on images",
        description=(
            "Overlay existing YOLO-format annotations on images.\n"
            "If no annotation path is given, looks for <output>/<image_stem>.txt."
        ),
        epilog=(
            "examples:\n"
            "  hires plot -s image.tif --ann results/image.txt -m data.yaml\n"
            "  hires plot -s images/ --ann labels/ -o plots/ -m model.pt\n"
        ),
    )
    io = plot.add_argument_group("input / output")
    io.add_argument("-s", "--source", required=True, metavar="PATH",
                    help="Image file or directory to annotate")
    io.add_argument("-m", "--model", default=DEFAULT_MODEL, metavar="PATH",
                    help="Model weights (.pt) or data YAML file; used for class names only")
    io.add_argument("-o", "--output", default="results", metavar="DIR",
                    help="Output directory for annotated images")
    io.add_argument("--ann", default=None, metavar="PATH",
                    help="Annotation .txt file, or a directory of .txt files matched "
                         "to images by file stem")
    io.add_argument("-r", "--recursive", action="store_true",
                    help="Search source directory recursively")

    # ---- evaluate ----------------------------------------------------- #
    evaluate = commands.add_parser(
        "evaluate",
        formatter_class=_Formatter,
        help="Compare predicted annotations with ground truth",
        description=(
            "Compare a directory of predicted YOLO-format annotations with ground truth.\n"
            "Files are paired by name. Writes the detection report, the confusion matrix,\n"
            "the precision-recall curve, a table of all matches and comparison images."
        ),
        epilog=(
            "examples:\n"
            "  hires evaluate --gt labels/ --pred results/\n"
            "  hires evaluate --gt labels/ --pred results/ --images images/ -n data.yaml\n"
            "  hires evaluate --gt labels/ --pred results/ -o evaluation/ --iou 0.7\n"
        ),
    )
    io = evaluate.add_argument_group("input / output")
    io.add_argument("--gt", required=True, metavar="DIR",
                    help="Directory of ground-truth .txt files")
    io.add_argument("--pred", required=True, metavar="DIR",
                    help="Directory of predicted .txt files (e.g. the output of 'hires run')")
    io.add_argument("--images", default=None, metavar="DIR",
                    help="Directory of the images; enables pixel measurements and comparison images")
    io.add_argument("-n", "--names", default=None, metavar="PATH",
                    help="Model weights (.pt) or data YAML file; used for class names only")
    io.add_argument("-o", "--output", default="evaluation", metavar="DIR",
                    help="Output directory for the report")
    options = evaluate.add_argument_group("evaluation")
    options.add_argument("--iou", type=float, default=0.5, metavar="FLOAT",
                         help="Minimum IoU for a prediction to match a ground-truth object")
    options.add_argument("--no-images", action="store_true",
                         help="Do not write comparison images")
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    from .segmentation import run_pipeline

    if (args.dpi is None) != (args.unit is None):
        raise HiresError("--dpi and --unit must be given together.")
    album = run_pipeline(
        args.source,
        args.model,
        args.output,
        recursive=args.recursive,
        dpi=args.dpi,
        unit=args.unit,
        save_crops=args.save_crops,
        debug=args.debug,
        on_error="skip",
        tile_size=tuple(args.chunk_size),
        overlap=args.overlap,
        confidence=args.conf,
        inference_size=args.imgsz,
        device=args.device,
        edge_margin=args.edge_threshold,
        nms_iou=args.iou_thresh,
        batch_size=args.batch_size,
    )
    from .io import find_images

    n_images = len(find_images(args.source, recursive=args.recursive))
    return 0 if len(album) == n_images else 1


def _cmd_chunk(args: argparse.Namespace) -> int:
    from .io import find_images
    from .tiling import tile_image

    images = find_images(args.source, recursive=args.recursive)
    if not images:
        raise HiresError(f"No images found at {args.source}.")
    for image in images:
        tiles = tile_image(
            image, args.output, tile_size=tuple(args.chunk_size), overlap=args.overlap
        )
        logger.info("%s -> %d chunks", image.name, len(tiles))
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    from . import plot
    from .io import find_images, read_class_names, read_txt

    images = find_images(args.source, recursive=args.recursive)
    if not images:
        raise HiresError(f"No images found at {args.source}.")
    class_names = read_class_names(args.model)
    output = Path(args.output)
    annotations = Path(args.ann) if args.ann else output

    missing = 0
    for image in images:
        labels = annotations / f"{image.stem}.txt" if annotations.is_dir() else annotations
        if not labels.is_file():
            logger.warning("No annotation file for %s (looked for %s)", image.name, labels)
            missing += 1
            continue
        target = output / f"{image.stem}_annotated.png"
        plot.overlay(image, read_txt(labels), class_names=class_names, save=target)
        logger.info("%s -> %s", image.name, target)
    return 1 if missing else 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .io import read_album, read_class_names
    from .metrics import evaluate

    class_names = read_class_names(args.names) if args.names else None
    gt = read_album(args.gt, image_dir=args.images)
    pred = read_album(args.pred, image_dir=args.images)
    for album, directory in ((gt, args.gt), (pred, args.pred)):
        if not len(album):
            raise HiresError(f"No .txt annotation files found in {directory}.")
    report = evaluate(
        gt,
        pred,
        args.output,
        iou_threshold=args.iou,
        class_names=class_names,
        save_images=not args.no_images,
    )
    print(report.to_string(float_format=lambda value: f"{value:.3f}"))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``hires`` command. Returns the process exit code."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    command = {
        "run": _cmd_run,
        "chunk": _cmd_chunk,
        "plot": _cmd_plot,
        "evaluate": _cmd_evaluate,
    }[args.command]
    try:
        return command(args)
    except (HiresError, FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
