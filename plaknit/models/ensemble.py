"""Ensemble of Boosted Regression Tree models with probability summaries."""

from __future__ import annotations

import atexit
import concurrent.futures
import contextlib
import csv
import json
import multiprocessing
import os
import queue
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import joblib
import numpy as np
import rasterio
from rasterio import windows
from scipy import stats

from . import brt
from ..data.raster import (
    _RasterStack,
    _log,
    _nodata_pixel_mask,
    _open_raster_stack,
)
from ..processing.evaluation import (
    _log_holdout_metrics,
    _write_holdout_outputs,
    _write_point_gpkg,
)

PathLike = Union[str, Path]
WindowTuple = Tuple[int, int, int, int]
_WorkerResult = Tuple[int, List[int], np.ndarray]


def _assign_models_to_workers(n_models: int, n_workers: int) -> List[List[int]]:
    """Assign every model exactly once across workers in round-robin order."""
    if n_models < 1:
        raise ValueError("n_models must be at least 1.")
    if n_workers < 1:
        raise ValueError("n_workers must be at least 1.")
    worker_count = min(n_models, n_workers)
    return [
        [
            model_idx
            for model_idx in range(n_models)
            if model_idx % worker_count == worker_id
        ]
        for worker_id in range(worker_count)
    ]


def _write_model_summary_csv(
    path: Path,
    model_paths: Sequence[Path],
    metadata: Dict[str, Any],
) -> None:
    """Write a CSV summarizing each ensemble member's seed and holdout AUC."""
    test_auc = metadata.get("test_auc", [])
    base_seed = metadata.get("random_state", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_index",
                "model_path",
                "random_seed",
                "test_auc",
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
            ]
        )
        for idx, model_path in enumerate(model_paths):
            auc = test_auc[idx] if idx < len(test_auc) else ""
            writer.writerow(
                [
                    idx,
                    model_path.name,
                    base_seed + idx,
                    auc,
                    metadata.get("n_estimators", ""),
                    metadata.get("max_depth", ""),
                    metadata.get("learning_rate", ""),
                    metadata.get("subsample", ""),
                    metadata.get("colsample_bytree", ""),
                ]
            )


def _write_feature_importance_csv(path: Path, models: Sequence[Any]) -> None:
    """Write unweighted split-importance summary statistics for ensemble bands."""
    importance_by_band: Dict[int, List[float]] = {}
    member_importances: List[Dict[int, float]] = []
    for model in models:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            member_importances.append({})
            continue
        band_indices = getattr(model, "band_indices", None)
        if band_indices is None:
            band_indices = list(range(1, len(importances) + 1))
        values = {
            int(band_index): float(importance)
            for band_index, importance in zip(band_indices, importances)
        }
        member_importances.append(values)
        for band_index in values:
            importance_by_band.setdefault(band_index, [])

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "band_index",
                "mean_importance",
                "std_importance",
                "min_importance",
                "max_importance",
                "models_using_feature",
            ]
        )
        rows = []
        for band_index in importance_by_band:
            values = np.asarray(
                [member.get(band_index, 0.0) for member in member_importances],
                dtype="float64",
            )
            rows.append(
                (
                    band_index,
                    float(values.mean()),
                    float(values.std()),
                    float(values.min()),
                    float(values.max()),
                    int(np.count_nonzero(values)),
                )
            )
        for row in sorted(rows, key=lambda row: (-row[1], row[0])):
            writer.writerow(row)


def _aggregate_ensemble_probabilities(
    member_probs: np.ndarray,
    ci_t_crit: Optional[float],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Aggregate complete ensemble probabilities for one output window."""
    mean = member_probs.mean(axis=0)
    if ci_t_crit is None or member_probs.shape[0] < 2:
        return mean, None, None
    std = member_probs.std(axis=0, ddof=1)
    half_width = ci_t_crit * std / np.sqrt(member_probs.shape[0])
    return (
        mean,
        np.clip(mean - half_width, 0.0, 1.0),
        np.clip(mean + half_width, 0.0, 1.0),
    )


def _predict_ensemble_block(
    stack: _RasterStack,
    models: List[Any],
    win: windows.Window,
    *,
    classes: np.ndarray,
    block_overlap: int,
    ci_t_crit: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Predict member probabilities and unweighted summaries for one block.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
        ``(member_probabilities, mean, lower_ci, upper_ci)``. Member
        probabilities have shape ``(M, C, H, W)``; all summary arrays have
        shape ``(C, H, W)``. CI arrays are ``None`` for a one-member ensemble.
    """
    n_models = len(models)
    num_classes = len(classes)
    read_window, write_slice = brt._window_with_overlap(
        win,
        block_overlap=block_overlap,
        raster_width=stack.width,
        raster_height=stack.height,
    )

    def _empty_result() -> (
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
    ):
        empty_probs = np.full(
            (num_classes, int(win.height), int(win.width)), np.nan, dtype="float32"
        )
        return (
            np.full(
                (n_models, num_classes, int(win.height), int(win.width)),
                np.nan,
                dtype="float32",
            ),
            empty_probs,
            empty_probs.copy() if n_models > 1 else None,
            empty_probs.copy() if n_models > 1 else None,
        )

    block = stack.read(window=read_window, out_dtype="float32")
    if block.size == 0:
        return _empty_result()

    samples = block.reshape(stack.count, -1).T
    valid = ~_nodata_pixel_mask(samples, stack.nodata_values)
    if not np.any(valid):
        return _empty_result()

    # (n_models, valid_count, num_classes)
    probs_stack = np.stack(
        [
            model.predict_proba(samples[valid]).astype("float32", copy=False)
            for model in models
        ],
        axis=0,
    )
    mean, lower, upper = _aggregate_ensemble_probabilities(probs_stack, ci_t_crit)

    def _scatter(values: np.ndarray) -> np.ndarray:
        full = np.full((samples.shape[0], num_classes), np.nan, dtype="float32")
        full[valid] = values
        return full.reshape(
            int(read_window.height), int(read_window.width), num_classes
        )

    mean_cube = _scatter(mean)

    lower_cube = _scatter(lower) if lower is not None else None
    upper_cube = _scatter(upper) if upper is not None else None

    member_cube = np.full(
        (n_models, samples.shape[0], num_classes), np.nan, dtype="float32"
    )
    member_cube[:, valid] = probs_stack
    member_cube = member_cube.reshape(
        n_models, int(read_window.height), int(read_window.width), num_classes
    )
    member_out = member_cube[:, write_slice[0], write_slice[1]].transpose(0, 3, 1, 2)
    mean_out = mean_cube[write_slice[0], write_slice[1]].transpose(2, 0, 1)
    lower_out = (
        lower_cube[write_slice[0], write_slice[1]].transpose(2, 0, 1)
        if lower_cube is not None
        else None
    )
    upper_out = (
        upper_cube[write_slice[0], write_slice[1]].transpose(2, 0, 1)
        if upper_cube is not None
        else None
    )
    return member_out, mean_out, lower_out, upper_out


_ENSEMBLE_PREDICT_STACK: Optional[_RasterStack] = None
_ENSEMBLE_PREDICT_MODELS: Optional[List[Any]] = None
_ENSEMBLE_PREDICT_CLASSES: Optional[np.ndarray] = None
_ENSEMBLE_PREDICT_CI_T_CRIT: Optional[float] = None


def _close_ensemble_predict_worker() -> None:
    global _ENSEMBLE_PREDICT_STACK
    if _ENSEMBLE_PREDICT_STACK is not None:
        _ENSEMBLE_PREDICT_STACK.__exit__(None, None, None)
        _ENSEMBLE_PREDICT_STACK = None


def _init_ensemble_predict_worker(
    image_paths: List[str],
    band_indices: Optional[Sequence[int]],
    model_paths: List[str],
    classes: List[Any],
    ci_t_crit: Optional[float],
    gpu: bool,
) -> None:
    global _ENSEMBLE_PREDICT_STACK
    global _ENSEMBLE_PREDICT_MODELS
    global _ENSEMBLE_PREDICT_CLASSES
    global _ENSEMBLE_PREDICT_CI_T_CRIT

    stack = _open_raster_stack(image_paths, band_indices=band_indices)
    stack.__enter__()
    _ENSEMBLE_PREDICT_STACK = stack
    _ENSEMBLE_PREDICT_MODELS = [joblib.load(path) for path in model_paths]
    for model in _ENSEMBLE_PREDICT_MODELS:
        brt._configure_prediction_device(model, gpu)
    _ENSEMBLE_PREDICT_CLASSES = np.asarray(classes)
    _ENSEMBLE_PREDICT_CI_T_CRIT = ci_t_crit
    atexit.register(_close_ensemble_predict_worker)


def _predict_ensemble_block_worker(
    win_tuple: WindowTuple, block_overlap: int
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if _ENSEMBLE_PREDICT_STACK is None or _ENSEMBLE_PREDICT_MODELS is None:
        raise RuntimeError("Ensemble prediction worker not initialized.")
    if _ENSEMBLE_PREDICT_CLASSES is None:
        raise RuntimeError("Ensemble prediction worker missing classes.")

    win = brt._tuple_to_window(win_tuple)
    return _predict_ensemble_block(
        _ENSEMBLE_PREDICT_STACK,
        _ENSEMBLE_PREDICT_MODELS,
        win,
        classes=_ENSEMBLE_PREDICT_CLASSES,
        block_overlap=block_overlap,
        ci_t_crit=_ENSEMBLE_PREDICT_CI_T_CRIT,
    )


def _ensemble_gpu_worker(
    image_paths: List[str],
    band_indices: Optional[Sequence[int]],
    model_paths: List[str],
    model_indices: List[int],
    classes: List[Any],
    device_id: int,
    task_queue: Any,
    result_queue: Any,
) -> None:
    """Run assigned ensemble models on one persistent CUDA worker."""
    try:
        with _open_raster_stack(image_paths, band_indices=band_indices) as stack:
            models = [joblib.load(model_paths[idx]) for idx in model_indices]
            for model in models:
                brt._configure_prediction_device(model, True, device_id=device_id)

            while True:
                task = task_queue.get()
                if task is None:
                    return
                window_id, win_tuple, block_overlap = task
                member_probs, _, _, _ = _predict_ensemble_block(
                    stack,
                    models,
                    brt._tuple_to_window(win_tuple),
                    classes=np.asarray(classes),
                    block_overlap=block_overlap,
                    ci_t_crit=None,
                )
                result_queue.put(("result", window_id, model_indices, member_probs))
    except BaseException as exc:
        result_queue.put(("error", repr(exc)))


class BRTEnsemble:
    """Ensemble of independent BRT models with probability summaries.

    This class trains multiple independent BRT models with different random seeds,
    then writes unweighted mean probability and confidence-interval rasters.

    Parameters
    ----------
    n_models
        Number of BRT models in the ensemble. Default 5.
    random_state
        Base random seed. Each model gets seed+i where i is the model index.
    n_estimators
        Number of boosting rounds per BRT model. Default 200.
    max_depth
        Maximum tree depth. Default 6.
    learning_rate
        Learning rate (eta). Default 0.1.
    subsample
        Fraction of samples per boosting round. Default 0.8.
    colsample_bytree
        Fraction of features per tree. Default 0.8.
    test_fraction
        Fraction of training samples held out for evaluation. Default 0.3.
    gpu
        If True, attempt GPU acceleration. Default True.

    Attributes
    ----------
    models_
        List of trained XGBClassifier instances.
    metadata_
        Dict containing training configuration and per-model accuracies.
    """

    def __init__(
        self,
        n_models: int = 5,
        random_state: int = 13,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        test_fraction: float = 0.3,
        pseudo_absence_ratio: float = 1.0,
        training_buffer_meters: float = 0.0,
        gpu: bool = True,
    ):
        self.n_models = n_models
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.test_fraction = test_fraction
        self.pseudo_absence_ratio = pseudo_absence_ratio
        self.training_buffer_meters = training_buffer_meters
        self.gpu = gpu
        self.models_: List[Any] = []
        self.metadata_: Dict[str, Any] = {}

    def fit(
        self,
        image_path: Union[PathLike, Iterable[PathLike]],
        shapefile_path: PathLike,
        label_column: str,
        ensemble_dir: PathLike,
        *,
        band_indices: Optional[Sequence[int]] = None,
        test_fraction: Optional[float] = None,
        grid_size: Optional[int] = None,
        tree_method: Optional[str] = None,
        jobs: int = 1,
    ) -> BRTEnsemble:
        """Train an ensemble of BRT models.

        Parameters
        ----------
        image_path
            Path to raster file, directory of GeoTIFFs, or iterable of aligned rasters.
        shapefile_path
            Path to training polygon GeoPackage or Shapefile.
        label_column
            Name of the column in shapefile_path containing class labels.
        ensemble_dir
            Directory where ensemble models and metadata will be saved.
        band_indices
            Optional 1-based band indices to select subset of input bands.
        test_fraction
            Optional holdout fraction override for this training run. When omitted,
            the value provided to the constructor is used.
        grid_size
            Optional pixel grid size for spatial thinning of training samples.
        tree_method
            Explicitly set tree method ('hist', 'gpu_hist', etc.). If None, auto-selects.
        jobs
            Number of ensemble members to fit concurrently. CPU training uses
            bounded workers; GPU training remains serial by default.

        Returns
        -------
        self
        """
        if self.n_models < 1:
            raise ValueError("n_models must be at least 1.")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")
        if jobs is None or jobs <= 0:
            jobs = min(self.n_models, os.cpu_count() or 1)
        jobs = min(jobs, self.n_models)
        if self.gpu and jobs > 1:
            _log("[yellow]GPU ensemble training is serial; using jobs=1.")
            jobs = 1
        if test_fraction is not None:
            self.test_fraction = test_fraction
        ensemble_path = Path(ensemble_dir)
        ensemble_path.mkdir(parents=True, exist_ok=True)

        _log(
            f"[bold cyan]Training BRT ensemble ({self.n_models} models) "
            f"to {ensemble_path}..."
        )

        self.models_ = []
        test_auc = []
        prepared = brt._prepare_brt_training_data(
            image_path,
            shapefile_path,
            label_column,
            band_indices=band_indices,
            training_buffer_meters=self.training_buffer_meters,
        )

        def train_member(idx: int) -> Tuple[int, Any]:
            model_seed = self.random_state + idx
            model_path = ensemble_path / f"brt_{idx}.joblib"

            _log(
                f"[bold cyan]Training model {idx + 1}/{self.n_models} (seed={model_seed})..."
            )
            model = brt.train_brt(
                image_path,
                shapefile_path,
                label_column,
                model_path,
                band_indices=band_indices,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=model_seed,
                test_fraction=self.test_fraction,
                grid_size=grid_size,
                pseudo_absence_ratio=self.pseudo_absence_ratio,
                training_buffer_meters=self.training_buffer_meters,
                gpu=self.gpu,
                tree_method=tree_method,
                _prepared=prepared,
                n_jobs=max(1, (os.cpu_count() or 1) // jobs),
            )
            return idx, model

        if jobs == 1:
            member_results = [train_member(idx) for idx in range(self.n_models)]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
                member_results = list(executor.map(train_member, range(self.n_models)))

        for idx, model in member_results:
            self.models_.append(model)

            metrics = _log_holdout_metrics(model)
            if metrics is not None and metrics.get("auc") is not None:
                auc = float(metrics["auc"])
            else:
                auc = None
            test_auc.append(auc)
            if auc is None:
                _log(f"[yellow]Model {idx} holdout ROC AUC unavailable.")
            else:
                _log(f"[green]Model {idx} holdout ROC AUC: {auc:.4f}")

        # Save metadata
        self.metadata_ = {
            "n_models": self.n_models,
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "test_fraction": self.test_fraction,
            "pseudo_absence_ratio": self.pseudo_absence_ratio,
            "training_buffer_meters": self.training_buffer_meters,
            "jobs": jobs,
            "band_indices": list(band_indices) if band_indices is not None else None,
            "grid_size": grid_size,
            "test_auc": test_auc,
        }
        metadata_path = ensemble_path / "ensemble_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata_, f, indent=2)
        _log(f"[green]Ensemble metadata saved to {metadata_path}")
        _log("[green]Ensemble training complete.")

        return self

    def predict(
        self,
        image_path: Union[PathLike, Iterable[PathLike]],
        ensemble_dir: PathLike,
        output_dir: PathLike,
        *,
        band_indices: Optional[Sequence[int]] = None,
        block_shape: Optional[Tuple[int, int]] = None,
        ci_level: float = 0.95,
        model_summary_out: Optional[PathLike] = None,
        feature_importance_out: Optional[PathLike] = None,
        block_overlap: int = 0,
        jobs: int = 1,
        gpu: bool = False,
    ) -> Path:
        """Write unweighted ensemble probability summaries for a raster stack.

        Parameters
        ----------
        image_path
            Path to raster file, directory of GeoTIFFs, or iterable of aligned rasters.
        ensemble_dir
            Directory containing ensemble models and metadata (from fit()).
        output_dir
            Directory where probability summary rasters will be saved.
        band_indices
            Optional 1-based band indices to select subset of input bands.
        block_shape
            Optional (height, width) tuple for custom block size.
        ci_level
            Confidence level for probability bounds (default 0.95).
        model_summary_out
            Optional CSV path summarizing each ensemble member (seed, test
            ROC AUC, and hyperparameters).
        feature_importance_out
            Optional CSV path for unweighted split-importance summary statistics
            across ensemble members.
        block_overlap
            Number of pixels to overlap blocks for smoothing edge artifacts.
        jobs
            Number of parallel worker processes. In GPU mode, this is capped by
            the number of visible CUDA devices and ensemble models.
        gpu
            If True, configure each XGBoost model for CUDA prediction. Worker
            processes run concurrently when ``jobs`` is greater than one.

        Returns
        -------
        Path
            Path to the mean probability raster.
        """
        if not 0.0 < ci_level < 1.0:
            raise ValueError("ci_level must be between 0 and 1 (exclusive).")
        if jobs is None:
            jobs = 1
        if jobs <= 0:
            jobs = max(1, os.cpu_count() or 1)
        ensemble_path = Path(ensemble_dir)

        # Load metadata
        metadata_path = ensemble_path / "ensemble_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Ensemble metadata not found at {metadata_path}")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        self.metadata_ = metadata

        # Load all models
        model_paths = sorted(ensemble_path.glob("brt_*.joblib"))
        expected_models = metadata.get("n_models", metadata.get("n_estimators", 1))
        if len(model_paths) != expected_models:
            raise ValueError(
                f"Expected {expected_models} models but found {len(model_paths)}"
            )

        models = [joblib.load(path) for path in model_paths]
        cuda_devices = brt._available_cuda_devices() if gpu else []
        if gpu and not cuda_devices:
            raise RuntimeError(
                "GPU prediction requested, but no visible CUDA devices were found."
            )
        if gpu and jobs <= 1:
            for model in models:
                brt._configure_prediction_device(model, True, device_id=0)
        self.models_ = models

        # Verify class consistency
        class_sets = [set(getattr(model, "classes_", [])) for model in models]
        if not all(cs == class_sets[0] for cs in class_sets):
            raise ValueError("Ensemble models have inconsistent class definitions.")

        if model_summary_out is not None:
            summary_path = Path(model_summary_out)
            _write_model_summary_csv(summary_path, model_paths, metadata)
            _log(f"[green]Model summary saved to {summary_path}")
        if feature_importance_out is not None:
            importance_path = Path(feature_importance_out)
            _write_feature_importance_csv(importance_path, models)
            _log(f"[green]Feature importance summary saved to {importance_path}")

        ci_t_crit: Optional[float] = None
        if len(models) > 1:
            ci_t_crit = float(stats.t.ppf(0.5 + ci_level / 2, df=len(models) - 1))
        else:
            _log(
                "[yellow]One-member ensemble; writing mean probabilities without "
                "confidence intervals."
            )

        # Setup output
        _log("[bold cyan]Predicting with ensemble...")
        output_directory = Path(output_dir)
        output_directory.mkdir(parents=True, exist_ok=True)
        mean_path = output_directory / "mean_probabilities.tif"
        lower_path = output_directory / "lower_probabilities.tif"
        upper_path = output_directory / "upper_probabilities.tif"

        # Infer band indices from metadata or input
        band_indices_to_use = band_indices
        if band_indices_to_use is None and metadata.get("band_indices"):
            band_indices_to_use = metadata["band_indices"]

        with _open_raster_stack(image_path, band_indices=band_indices_to_use) as stack:
            assert stack.template is not None

            classes = class_sets[0]
            classes_arr = np.asarray(sorted(classes))
            probs_profile = brt._prepare_prob_profile(
                stack.profile, count=len(classes_arr)
            )
            probs_profile["driver"] = "GTiff"

            with tempfile.TemporaryDirectory(prefix="plaknit_brt_") as temp_dir:
                member_paths = [
                    Path(temp_dir) / f"model_{idx:03d}_probabilities.tif"
                    for idx in range(len(models))
                ]
                with contextlib.ExitStack() as stack_ctx:
                    mean_dst = stack_ctx.enter_context(
                        rasterio.open(mean_path, "w", **probs_profile)
                    )
                    lower_dst = (
                        stack_ctx.enter_context(
                            rasterio.open(lower_path, "w", **probs_profile)
                        )
                        if ci_t_crit is not None
                        else None
                    )
                    upper_dst = (
                        stack_ctx.enter_context(
                            rasterio.open(upper_path, "w", **probs_profile)
                        )
                        if ci_t_crit is not None
                        else None
                    )
                    member_dsts = [
                        stack_ctx.enter_context(
                            rasterio.open(path, "w", **probs_profile)
                        )
                        for path in member_paths
                    ]

                    if block_shape:
                        block_h, block_w = block_shape

                        def custom_windows() -> Iterable[windows.Window]:
                            for row_off in range(0, stack.height, block_h):
                                for col_off in range(0, stack.width, block_w):
                                    yield windows.Window(
                                        col_off=col_off,
                                        row_off=row_off,
                                        width=min(block_w, stack.width - col_off),
                                        height=min(block_h, stack.height - row_off),
                                    )

                        window_iter: Iterable[windows.Window] = custom_windows()
                    else:
                        window_iter = (win for _, win in stack.block_windows(1))

                    def _write_block_result(
                        win: windows.Window,
                        member_probs: np.ndarray,
                        mean_probs: np.ndarray,
                        lower_probs: Optional[np.ndarray],
                        upper_probs: Optional[np.ndarray],
                    ) -> None:
                        for member_dst, probs in zip(member_dsts, member_probs):
                            member_dst.write(probs, window=win)
                        mean_dst.write(mean_probs, window=win)
                        if lower_dst is not None and lower_probs is not None:
                            lower_dst.write(lower_probs, window=win)
                        if upper_dst is not None and upper_probs is not None:
                            upper_dst.write(upper_probs, window=win)

                    if gpu and jobs > 1:
                        worker_count = min(jobs, len(models), len(cuda_devices))
                        assignments = _assign_models_to_workers(
                            len(models), worker_count
                        )
                        context = multiprocessing.get_context("spawn")
                        task_queues = [context.Queue(maxsize=2) for _ in assignments]
                        result_queue = context.Queue()
                        workers = [
                            context.Process(
                                target=_ensemble_gpu_worker,
                                args=(
                                    [str(path) for path in stack.paths],
                                    band_indices_to_use,
                                    [str(path) for path in model_paths],
                                    assignment,
                                    classes_arr.tolist(),
                                    cuda_devices[worker_id],
                                    task_queue,
                                    result_queue,
                                ),
                            )
                            for worker_id, (assignment, task_queue) in enumerate(
                                zip(assignments, task_queues)
                            )
                        ]
                        for worker in workers:
                            worker.start()
                        try:
                            for window_id, win in enumerate(window_iter):
                                task = (
                                    window_id,
                                    brt._window_to_tuple(win),
                                    block_overlap,
                                )
                                for task_queue in task_queues:
                                    task_queue.put(task)

                                partial_results: Dict[int, np.ndarray] = {}
                                for _ in workers:
                                    try:
                                        message = result_queue.get(timeout=300)
                                    except queue.Empty as exc:
                                        dead = [
                                            worker.pid
                                            for worker in workers
                                            if not worker.is_alive()
                                        ]
                                        raise RuntimeError(
                                            "GPU prediction worker stopped before "
                                            f"completing window {window_id}; pids={dead}."
                                        ) from exc
                                    if message[0] == "error":
                                        raise RuntimeError(
                                            f"GPU prediction worker failed: {message[1]}"
                                        )
                                    _, returned_window_id, indices, member_probs = (
                                        message
                                    )
                                    if returned_window_id != window_id:
                                        raise RuntimeError(
                                            "GPU prediction returned an unexpected window."
                                        )
                                    for local_idx, model_idx in enumerate(indices):
                                        if model_idx in partial_results:
                                            raise RuntimeError(
                                                f"Duplicate GPU prediction for model {model_idx}."
                                            )
                                        partial_results[model_idx] = member_probs[
                                            local_idx
                                        ]

                                if len(partial_results) != len(models):
                                    raise RuntimeError(
                                        "GPU prediction returned an incomplete ensemble "
                                        f"for window {window_id}."
                                    )
                                full_member_probs = np.stack(
                                    [
                                        partial_results[idx]
                                        for idx in range(len(models))
                                    ],
                                    axis=0,
                                )
                                mean_probs, lower_probs, upper_probs = (
                                    _aggregate_ensemble_probabilities(
                                        full_member_probs, ci_t_crit
                                    )
                                )
                                _write_block_result(
                                    win,
                                    full_member_probs,
                                    mean_probs,
                                    lower_probs,
                                    upper_probs,
                                )
                        finally:
                            for task_queue in task_queues:
                                task_queue.put(None)
                            for worker in workers:
                                worker.join(timeout=10)
                                if worker.is_alive():
                                    worker.terminate()
                                    worker.join()
                    elif jobs > 1:
                        max_workers = jobs
                        max_pending = max_workers * 2
                        image_paths = [str(path) for path in stack.paths]
                        model_path_strs = [str(path) for path in model_paths]
                        with concurrent.futures.ProcessPoolExecutor(
                            max_workers=max_workers,
                            initializer=_init_ensemble_predict_worker,
                            initargs=(
                                image_paths,
                                band_indices_to_use,
                                model_path_strs,
                                classes_arr.tolist(),
                                ci_t_crit,
                                gpu,
                            ),
                        ) as executor:
                            futures: Dict[concurrent.futures.Future, windows.Window] = (
                                {}
                            )
                            for win in window_iter:
                                future = executor.submit(
                                    _predict_ensemble_block_worker,
                                    brt._window_to_tuple(win),
                                    block_overlap,
                                )
                                futures[future] = win
                                if len(futures) >= max_pending:
                                    done, _ = concurrent.futures.wait(
                                        futures,
                                        return_when=concurrent.futures.FIRST_COMPLETED,
                                    )
                                    for finished in done:
                                        (
                                            member_probs,
                                            mean_probs,
                                            lower_probs,
                                            upper_probs,
                                        ) = finished.result()
                                        _write_block_result(
                                            futures[finished],
                                            member_probs,
                                            mean_probs,
                                            lower_probs,
                                            upper_probs,
                                        )
                                        del futures[finished]

                            for finished in concurrent.futures.as_completed(futures):
                                member_probs, mean_probs, lower_probs, upper_probs = (
                                    finished.result()
                                )
                                _write_block_result(
                                    futures[finished],
                                    member_probs,
                                    mean_probs,
                                    lower_probs,
                                    upper_probs,
                                )
                    else:
                        for win in window_iter:
                            member_probs, mean_probs, lower_probs, upper_probs = (
                                _predict_ensemble_block(
                                    stack,
                                    models,
                                    win,
                                    classes=classes_arr,
                                    block_overlap=block_overlap,
                                    ci_t_crit=ci_t_crit,
                                )
                            )
                            _write_block_result(
                                win, member_probs, mean_probs, lower_probs, upper_probs
                            )

        _log(f"[green]Mean probabilities saved to {mean_path}")
        if ci_t_crit is not None:
            _log(f"[green]Lower probability bounds saved to {lower_path}")
            _log(f"[green]Upper probability bounds saved to {upper_path}")
        return mean_path
