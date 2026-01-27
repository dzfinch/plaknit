"""Command-line interface for Random Forest training and prediction."""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

from .classify import predict_rf, smooth_probs, train_rf


def _add_common_smoothing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--smooth",
        choices=["none", "mrf", "bayes"],
        default="none",
        help=(
            "Post-process predictions. 'mrf' enables Potts-MRF ICM smoothing; "
            "'bayes' enables empirical Bayes smoothing (SITS)."
        ),
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
        "--bayes-window-size",
        type=int,
        default=7,
        help="Window size (odd >=3) for Bayesian smoothing.",
    )
    parser.add_argument(
        "--bayes-neigh-fraction",
        type=float,
        default=0.5,
        help="Fraction of neighbors used for Bayesian smoothing (0-1].",
    )
    parser.add_argument(
        "--bayes-smoothness",
        type=float,
        default=20.0,
        help="Smoothness parameter (sigma^2) for Bayesian smoothing.",
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
        default=1,
        help="Parallel jobs for training (default: -1 = all cores).",
    )
    train_parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.3,
        help="Fraction of samples held out for evaluation (default: 0.3).",
    )
    train_parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help=(
            "Grid size in pixels for spatially diverse sampling; keeps at most one "
            "sample per class per grid cell."
        ),
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
        "--probs-out",
        help="Optional output path for class probability GeoTIFF.",
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

    smooth_parser = subparsers.add_parser(
        "smooth", help="Smooth a class-probability raster."
    )
    smooth_parser.add_argument(
        "--probs",
        required=True,
        help="Path to class-probability GeoTIFF (bands=classes).",
    )
    smooth_parser.add_argument(
        "--output", required=True, help="Path for smoothed GeoTIFF."
    )
    class_group = smooth_parser.add_mutually_exclusive_group()
    class_group.add_argument("--model", help="Optional model to map class values.")
    class_group.add_argument(
        "--class-values",
        nargs="+",
        type=float,
        help="Explicit class values matching the probability bands.",
    )
    smooth_parser.add_argument(
        "--block-shape",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="Override block/window shape for reading (height width).",
    )
    smooth_parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel workers for block smoothing (default: 1). Use -1 for all cores.",
    )
    _add_common_smoothing_args(smooth_parser)
    smooth_parser.set_defaults(smooth="mrf")

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
            grid_size=args.grid_size,
        )
        return 0

    if args.command == "smooth":
        block_shape = _parse_block_shape(args.block_shape)
        smooth_probs(
            probability_path=args.probs,
            output_path=args.output,
            model_path=args.model,
            class_values=args.class_values,
            block_shape=block_shape,
            smooth=args.smooth,
            beta=args.beta,
            neighborhood=args.neighborhood,
            icm_iters=args.icm_iters,
            bayes_window_size=args.bayes_window_size,
            bayes_neigh_fraction=args.bayes_neigh_fraction,
            bayes_smoothness=args.bayes_smoothness,
            block_overlap=args.block_overlap,
            jobs=args.jobs,
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
        bayes_window_size=args.bayes_window_size,
        bayes_neigh_fraction=args.bayes_neigh_fraction,
        bayes_smoothness=args.bayes_smoothness,
        probs_out=args.probs_out,
        block_overlap=args.block_overlap,
        jobs=args.jobs,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
