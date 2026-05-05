#!/usr/bin/env python3
from __future__ import annotations

import argparse

from hires.models.config import Settings


class _Fmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


def _build_run_parser(sub: argparse.ArgumentParser) -> None:
    sub.formatter_class = _Fmt
    sub.description = (
        "Run the full segmentation pipeline on a single image or a directory.\n"
        "Steps: chunk → predict → filter edges → unify → NMS → visualise → crops + CSV."
    )
    sub.epilog = (
        "examples:\n"
        "  hires run -s image.tif\n"
        "  hires run -s images/ -o results/ --conf 0.4 --device 0\n"
        "  hires run -s images/ --chunk-size 2048 2048 --overlap 256 --iou-thresh 0.6\n"
        "  hires run -s images/ --dpi 1200 --unit um\n"
        "  hires run -s images/ -r --debug\n"
    )

    io = sub.add_argument_group("input / output")
    io.add_argument("-s", "--source", required=True, metavar="PATH",
                    help="Image file or directory to process")
    io.add_argument("-m", "--model", default="models/DaphnAI.pt", metavar="PATH",
                    help="Path to YOLO model weights")
    io.add_argument("-o", "--output", default="results", metavar="DIR",
                    help="Root output directory")
    io.add_argument("-r", "--recursive", action="store_true",
                    help="Search source directory recursively")

    inf = sub.add_argument_group("inference")
    inf.add_argument("--conf", type=float, default=0.5, metavar="FLOAT",
                     help="Detection confidence threshold")
    inf.add_argument("--imgsz", type=int, default=1024, metavar="INT",
                     help="Inference image size (pixels)")
    inf.add_argument("--device", default="cpu", metavar="STR",
                     help="Inference device — cpu, 0, cuda:0, …")

    chunk = sub.add_argument_group("chunking")
    chunk.add_argument("--chunk-size", type=int, nargs=2, default=[1024, 1024],
                       metavar=("W", "H"),
                       help="Chunk width and height in pixels")
    chunk.add_argument("--overlap", type=int, default=150, metavar="INT",
                       help="Overlap between adjacent chunks in pixels")

    post = sub.add_argument_group("post-processing")
    post.add_argument("--edge-threshold", type=float, default=1e-2, metavar="FLOAT",
                      help="Normalised inset to remove edge-touching polygons")
    post.add_argument("--iou-thresh", type=float, default=0.7, metavar="FLOAT",
                      help="IoU threshold for polygon NMS")

    out = sub.add_argument_group("outputs")
    out.add_argument("--save-crops", action="store_true", default=False,
                     help="Save a masked crop image for every detection")
    out.add_argument("--dpi", type=float, default=None, metavar="FLOAT",
                     help="Scan resolution in DPI — enables physical measurements")
    out.add_argument("--unit", default=None, metavar="UNIT",
                     choices=["nm", "um", "mm", "cm", "m", "inch"],
                     help="Physical unit for shape descriptors {nm,um,mm,cm,m,inch}")

    misc = sub.add_argument_group("misc")
    misc.add_argument("--debug", action="store_true",
                      help="Write intermediate artefacts (chunks, raw preds, filtered preds)")


def _build_chunk_parser(sub: argparse.ArgumentParser) -> None:
    sub.formatter_class = _Fmt
    sub.description = "Slice one image or every image in a directory into overlapping chunks."
    sub.epilog = (
        "examples:\n"
        "  hires chunk -s image.tif -o chunks/\n"
        "  hires chunk -s images/ --chunk-size 2048 2048 --overlap 256\n"
    )

    io = sub.add_argument_group("input / output")
    io.add_argument("-s", "--source", required=True, metavar="PATH",
                    help="Image file or directory to chunk")
    io.add_argument("-o", "--output", default="chunks", metavar="DIR",
                    help="Output directory for chunk images")

    chunk = sub.add_argument_group("chunking")
    chunk.add_argument("--chunk-size", type=int, nargs=2, default=[1024, 1024],
                       metavar=("W", "H"),
                       help="Chunk width and height in pixels")
    chunk.add_argument("--overlap", type=int, default=150, metavar="INT",
                       help="Overlap between adjacent chunks in pixels")


def _build_plot_parser(sub: argparse.ArgumentParser) -> None:
    sub.formatter_class = _Fmt
    sub.description = (
        "Overlay existing YOLO-format annotations on images and save visualisations.\n"
        "If no annotation file is given, looks for <output>/<image_stem>.txt."
    )
    sub.epilog = (
        "examples:\n"
        "  hires plot -s image.tif --ann results/image.txt\n"
        "  hires plot -s images/ --ann labels/ -o plots/ -m data.yaml\n"
        "  hires plot -s images/ -o plots/ -r\n"
    )

    io = sub.add_argument_group("input / output")
    io.add_argument("-s", "--source", required=True, metavar="PATH",
                    help="Image file or directory to annotate")
    io.add_argument("-m", "--model", default="models/DaphnAI.pt", metavar="PATH",
                    help="Model weights (.pt) or data YAML file — used for class names only")
    io.add_argument("-o", "--output", default="results", metavar="DIR",
                    help="Output directory for annotated images")
    io.add_argument("--ann", default="", metavar="PATH",
                    help="Annotation .txt file or directory of .txt files (YOLO polygon format). "
                         "When a directory is given, files are matched to images by stem name.")
    io.add_argument("-r", "--recursive", action="store_true",
                    help="Search source directory recursively")


def _settings_from_args(args: argparse.Namespace) -> Settings:
    return Settings(
        source=args.source,
        model_path=getattr(args, "model", "models/DaphnAI.pt"),
        output_dir=args.output,
        conf=getattr(args, "conf", 0.5),
        imgsz=getattr(args, "imgsz", 1024),
        device=getattr(args, "device", "cpu"),
        chunk_size=tuple(getattr(args, "chunk_size", [1024, 1024])),
        overlap=getattr(args, "overlap", 150),
        edge_threshold=getattr(args, "edge_threshold", 1e-2),
        iou_thresh=getattr(args, "iou_thresh", 0.7),
        save_crops=getattr(args, "save_crops", True),
        dpi=getattr(args, "dpi", None),
        unit=getattr(args, "unit", None),
        debug=getattr(args, "debug", False),
        recursive=getattr(args, "recursive", False),
        ann=getattr(args, "ann", ""),
    )


def _cmd_run(args: argparse.Namespace) -> None:
    from hires.pipeline.seg_pipeline import SegmentationPipeline
    SegmentationPipeline(_settings_from_args(args)).run()


def _cmd_chunk(args: argparse.Namespace) -> None:
    from hires.processing.chunker import ImageChunker
    ImageChunker(args.source).slice(
        save_folder=args.output,
        chunk_size=tuple(args.chunk_size),
        overlap=args.overlap,
    )


def _cmd_plot(args: argparse.Namespace) -> None:
    from hires.pipeline.chunk_pipeline import PlottingPipeline
    PlottingPipeline(_settings_from_args(args)).run()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="hires",
        formatter_class=_Fmt,
        description="HiReS — high-resolution segmentation toolkit",
        epilog=(
            "commands:\n"
            "  run     Full segmentation pipeline (chunk → predict → NMS → outputs)\n"
            "  chunk   Slice images into overlapping tiles\n"
            "  plot    Overlay annotations on images\n"
            "\n"
            "Run 'hires <command> --help' for per-command options.\n"
        ),
    )
    subs = parser.add_subparsers(dest="command", metavar="COMMAND")
    subs.required = True

    _build_run_parser(subs.add_parser(
        "run",
        help="Run the full segmentation pipeline on an image or directory",
        formatter_class=_Fmt,
    ))
    _build_chunk_parser(subs.add_parser(
        "chunk",
        help="Slice images into overlapping chunks",
        formatter_class=_Fmt,
    ))
    _build_plot_parser(subs.add_parser(
        "plot",
        help="Overlay existing annotations on images and save visualisations",
        formatter_class=_Fmt,
    ))

    args = parser.parse_args(argv)

    dispatch = {
        "run": _cmd_run,
        "chunk": _cmd_chunk,
        "plot": _cmd_plot,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
