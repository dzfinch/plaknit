"""Command-line interface for Boosted Regression Tree (BRT) ensemble training and prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

from ..models.brt import train_brt
from ..models.ensemble import BRTEnsemble
from ..processing.evaluation import (
    generate_partial_dependence,
    _write_partial_dependence_csv,
    plot_partial_dependence,
)
from ..data.raster import _log


def _add_common_smoothing_args(parser: argparse.ArgumentParser) -> None:
    """Add common smoothing arguments to a parser."""
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
    """Parse block shape from command-line arguments."""
    if values is None:
        return None
    if len(values) != 2:
        raise argparse.ArgumentTypeError("block shape must be two integers")
    return int(values[0]), int(values[1])


def _flatten_image_args(image_args: Sequence[Sequence[str]]) -> List[str]:
    """Flatten nested image arguments."""
    images: List[str] = []
    for group in image_args:
        images.extend(group)
    return images


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Dispatch to ensemble train/predict sub-commands."""
    parser = argparse.ArgumentParser(
        prog="plaknit brt",
        description="Train or apply a Boosted Regression Tree (BRT) classifier to raster stacks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train sub-command
    train_parser = subparsers.add_parser(
        "train", help="Train one BRT model or an ensemble."
    )
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
        "--output",
        required=True,
        help="Model file when --n-models=1; ensemble directory when --n-models>1.",
    )
    train_parser.add_argument(
        "--n-models",
        type=int,
        default=1,
        help="Number of BRT models; 1 trains a single model (default: 1).",
    )
    train_parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of boosting rounds per model (default: 200).",
    )
    train_parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Max tree depth (default: 6).",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for boosting (default: 0.1).",
    )
    train_parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="Fraction of samples per tree (default: 1.0).",
    )
    train_parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=1.0,
        help="Fraction of features per tree (default: 1.0).",
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
    train_parser.add_argument(
        "--pseudo-absence-ratio",
        type=float,
        default=1.0,
        help="Pseudo-absence samples per presence sample (default: 1.0).",
    )
    train_parser.add_argument(
        "--training-buffer-meters",
        type=float,
        default=0.0,
        help="Buffer training geometries by this many meters (default: 0).",
    )
    train_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Base random seed for reproducibility (default: 42).",
    )
    train_parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration if available (XGBoost + CUDA).",
    )
    train_parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Concurrent ensemble model fits (default: 1; GPU training remains serial).",
    )
    train_parser.add_argument(
        "--pdp-csv",
        type=str,
        default=None,
        help=(
            "Generate partial-dependence CSVs after training. "
            "Pass '-1' to generate for all bands, or a comma-separated list of 1-based band indices, e.g. '1,3,5'. "
            "CSV files are written into the model/ensemble output directory as 'pdp_band_<band>.csv'."
        ),
    )

    # Predict sub-command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Apply a trained BRT ensemble to classify a raster stack.",
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
    predict_parser.add_argument(
        "--ensemble-dir",
        required=True,
        help="Directory containing the trained ensemble models and metadata.",
    )
    predict_parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory for mean_probabilities.tif and, for ensembles with at least "
            "two models, lower_probabilities.tif and upper_probabilities.tif."
        ),
    )
    predict_parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level for lower and upper probability bounds (default: 0.95).",
    )
    predict_parser.add_argument(
        "--model-summary-out",
        help="Optional CSV path summarizing each ensemble member's seed and ROC AUC.",
    )
    predict_parser.add_argument(
        "--feature-importance-out",
        help="Optional CSV path for unweighted ensemble predictor importance.",
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
        help=(
            "Parallel workers for block prediction (default: 1). Use -1 for all "
            "cores on CPU, or up to the number of visible GPUs with --gpu."
        ),
    )
    predict_parser.add_argument(
        "--gpu",
        action="store_true",
        help=(
            "Use CUDA for XGBoost prediction when available. With --jobs > 1, "
            "models are distributed across persistent GPU workers."
        ),
    )
    predict_parser.add_argument(
        "--block-overlap",
        type=int,
        default=0,
        help="Overlap in pixels used while processing raster blocks (default: 0).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "train":
        image_paths = _flatten_image_args(args.image)
        if args.n_models == 1:
            model = train_brt(
                image_path=image_paths,
                shapefile_path=args.labels,
                label_column=args.label_column,
                model_out=args.output,
                band_indices=args.band_indices,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                random_state=args.random_state,
                test_fraction=args.test_fraction,
                grid_size=args.grid_size,
                pseudo_absence_ratio=args.pseudo_absence_ratio,
                training_buffer_meters=args.training_buffer_meters,
                gpu=args.gpu,
                n_jobs=args.jobs,
            )
            # Optionally generate PDP CSVs for single model
            if args.pdp_csv is not None:
                out_dir = Path(args.output).parent
                _log(f"[cyan]Generating PDP CSV(s) into {out_dir}...")
                try:
                    pdp_arg = args.pdp_csv.strip()
                    if pdp_arg == "-1":
                        band_ids = getattr(model, "band_indices", None)
                        if band_ids is None:
                            n_feats = model.test_samples_.shape[1] if getattr(model, "test_samples_", None) is not None else 0
                            band_ids = list(range(1, n_feats + 1))
                        requested = list(band_ids)
                    else:
                        requested = [int(s) for s in pdp_arg.split(",") if s.strip()]

                    if not requested:
                        _log("[yellow]No PDP bands requested; skipping PDP generation.")
                    else:
                        for band in requested:
                            band_indices = getattr(model, "band_indices", None)
                            if band_indices is None:
                                feature_index = band - 1
                            else:
                                try:
                                    feature_index = int(band_indices.index(band))
                                except ValueError:
                                    _log(f"[yellow]Band {band} not found in model.band_indices; skipping.")
                                    continue
                            if getattr(model, "test_samples_", None) is None:
                                _log("[yellow]Model lacks test samples; PDP requires sample matrix; skipping.")
                                break
                            grid, mean_probs = generate_partial_dependence([model], feature_index)
                            out_path = out_dir / f"pdp_band_{band}.csv"
                            _write_partial_dependence_csv(out_path, feature_index, grid, mean_probs)
                            _log(f"[green]Wrote PDP CSV for band {band} to {out_path}")
                            try:
                                png_path = out_path.with_suffix(".png")
                                plot_partial_dependence(png_path, grid, mean_probs, title=f"PDP band {band}")
                            except Exception as exc:
                                _log(f"[yellow]PDP plotting failed: {exc}")
                except Exception as exc:
                    _log(f"[yellow]PDP generation failed: {exc}")
        elif args.n_models > 1:
            ensemble = BRTEnsemble(
                n_models=args.n_models,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                random_state=args.random_state,
                pseudo_absence_ratio=args.pseudo_absence_ratio,
                training_buffer_meters=args.training_buffer_meters,
                test_fraction=args.test_fraction,
                gpu=args.gpu,
            )
            ensemble.fit(
                image_path=image_paths,
                shapefile_path=args.labels,
                label_column=args.label_column,
                ensemble_dir=args.output,
                band_indices=args.band_indices,
                grid_size=args.grid_size,
                jobs=args.jobs,
            )
            # Optionally generate PDP CSVs for the ensemble
            if args.pdp_csv is not None:
                out_dir = Path(args.output)
                _log(f"[cyan]Generating PDP CSV(s) into {out_dir}...")
                try:
                    pdp_arg = args.pdp_csv.strip()
                    if pdp_arg == "-1":
                        model0 = ensemble.models_[0] if ensemble.models_ else None
                        if model0 is None:
                            _log("[yellow]No models available for PDP generation; skipping.")
                        else:
                            band_ids = getattr(model0, "band_indices", None)
                            if band_ids is None:
                                n_feats = model0.test_samples_.shape[1] if getattr(model0, "test_samples_", None) is not None else 0
                                requested = list(range(1, n_feats + 1))
                            else:
                                requested = list(band_ids)
                    else:
                        requested = [int(s) for s in pdp_arg.split(",") if s.strip()]

                    if not requested:
                        _log("[yellow]No PDP bands requested; skipping PDP generation.")
                    else:
                        model0 = ensemble.models_[0] if ensemble.models_ else None
                        if model0 is None:
                            _log("[yellow]No models available for PDP generation; skipping.")
                        elif getattr(model0, "test_samples_", None) is None:
                            _log("[yellow]Models lack test samples; PDP requires sample matrix; skipping.")
                        else:
                            for band in requested:
                                band_indices = getattr(model0, "band_indices", None)
                                if band_indices is None:
                                    feature_index = band - 1
                                else:
                                    try:
                                        feature_index = int(band_indices.index(band))
                                    except ValueError:
                                        _log(f"[yellow]Band {band} not found in model.band_indices; skipping.")
                                        continue
                                grid, mean_probs = generate_partial_dependence(ensemble.models_, feature_index)
                                out_path = out_dir / f"pdp_band_{band}.csv"
                                _write_partial_dependence_csv(out_path, feature_index, grid, mean_probs)
                                _log(f"[green]Wrote PDP CSV for band {band} to {out_path}")
                                try:
                                    png_path = out_path.with_suffix(".png")
                                    plot_partial_dependence(png_path, grid, mean_probs, title=f"PDP band {band}")
                                except Exception as exc:
                                    _log(f"[yellow]PDP plotting failed: {exc}")
                except Exception as exc:
                    _log(f"[yellow]PDP generation failed: {exc}")
        else:
            raise ValueError("n_models must be at least 1.")
        return 0

    if args.command == "predict":
        image_paths = _flatten_image_args(args.image)
        block_shape = _parse_block_shape(args.block_shape)
        ensemble = BRTEnsemble()  # Dummy instance to call predict classmethod
        ensemble.predict(
            image_path=image_paths,
            ensemble_dir=args.ensemble_dir,
            output_dir=args.output_dir,
            band_indices=args.band_indices,
            block_shape=block_shape,
            ci_level=args.ci_level,
            model_summary_out=args.model_summary_out,
            feature_importance_out=args.feature_importance_out,
            block_overlap=args.block_overlap,
            jobs=args.jobs,
            gpu=args.gpu,
        )
        return 0

    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
