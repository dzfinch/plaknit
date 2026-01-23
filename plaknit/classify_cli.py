"""Command-line interface for Random Forest training and prediction."""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

from .classify import predict_rf, train_rf


def _add_common_smoothing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--smooth",
        choices=["none", "mrf"],
        default="none",
        help="Post-process predictions. 'mrf' enables Potts-MRF ICM smoothing.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Smoothness strength for MRF (higher = smoother).",
    )
    parser.add_argument(
        "--neighborhood",
        type=int,
        choices=[4, 8],
        default=4,
        help="Neighborhood system for MRF smoothing.",
    )
    parser.add_argument(
        "--icm-iters",
        type=int,
        default=3,
        help="ICM iterations for MRF smoothing.",
    )
    parser.add_argument(
        "--block-overlap",
        type=int,
        default=0,
        help="Overlap (pixels) to reduce seams between blocks when smoothing.",
    )


def _parse_block_shape(values: Optional[Sequence[int]]) -> Optional[tuple[int, int]]:
    if values is None:
        return None
    if len(values) != 2:
        raise argparse.ArgumentTypeError("block shape must be two integers")
    return int(values[0]), int(values[1])


def _flatten_image_args(image_args: Sequence[Sequence[str]]) -> List[str]:
    images: List[str] = []
    for group in image_args:
        images.extend(group)
    return images


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="plaknit classify",
        description="Train or apply a Random Forest classifier to raster stacks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a Random Forest model.")
    train_parser.add_argument(
        "--image",
        required=True,
        nargs="+",
        action="append",
        help=(
            "Raster input(s): pass one or more GeoTIFF/VRT paths after --image, "
            "or repeat --image. Directories are expanded to TIFFs."
        ),
    )
    train_parser.add_argument(
        "--band-indices",
        nargs="+",
        type=int,
        help="Optional 1-based band indices to use from the stacked inputs.",
    )
    train_parser.add_argument(
        "--labels", required=True, help="Vector labels (e.g., Shapefile/GeoPackage)."
    )
    train_parser.add_argument(
        "--label-column",
        required=True,
        help="Column in labels containing class names/ids.",
    )
    train_parser.add_argument(
        "--model-out", required=True, help="Path to save the trained model (.joblib)."
    )
    train_parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="Number of trees (default: 500).",
    )
    train_parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Max tree depth (default: None).",
    )
    train_parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Parallel jobs for training (default: -1 = all cores).",
    )
    train_parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.3,
        help="Fraction of samples held out for evaluation (default: 0.3).",
    )

    predict_parser = subparsers.add_parser(
        "predict", help="Apply a trained model to classify a raster stack."
    )
    predict_parser.add_argument(
        "--image",
        required=True,
        nargs="+",
        action="append",
        help=(
            "Raster input(s): pass one or more GeoTIFF/VRT paths after --image, "
            "or repeat --image. Directories are expanded to TIFFs."
        ),
    )
    predict_parser.add_argument(
        "--band-indices",
        nargs="+",
        type=int,
        help="Optional 1-based band indices to use from the stacked inputs.",
    )
    predict_parser.add_argument("--model", required=True, help="Trained model path.")
    predict_parser.add_argument(
        "--output", required=True, help="Path for classified GeoTIFF."
    )
    predict_parser.add_argument(
        "--block-shape",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="Override block/window shape for reading (height width).",
    )
    predict_parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel workers for block prediction (default: 1). Use -1 for all cores.",
    )
    _add_common_smoothing_args(predict_parser)

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "train":
        image_paths = _flatten_image_args(args.image)
        train_rf(
            image_path=image_paths,
            shapefile_path=args.labels,
            label_column=args.label_column,
            model_out=args.model_out,
            band_indices=args.band_indices,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            n_jobs=args.jobs,
            test_fraction=args.test_fraction,
        )
        return 0

    block_shape = _parse_block_shape(args.block_shape)
    image_paths = _flatten_image_args(args.image)
    predict_rf(
        image_path=image_paths,
        model_path=args.model,
        output_path=args.output,
        band_indices=args.band_indices,
        block_shape=block_shape,
        smooth=args.smooth,
        beta=args.beta,
        neighborhood=args.neighborhood,
        icm_iters=args.icm_iters,
        block_overlap=args.block_overlap,
        jobs=args.jobs,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
