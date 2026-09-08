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
import time
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
        [model_idx for model_idx in range(n_models) if model_idx % worker_count == worker_id]
        for worker_id in range(worker_count)
    ]


def _prediction_checkpoint(
    processed: int,
    total: int,
    started_at: float,
    next_checkpoint: int,
) -> int:
    """Log throttled prediction progress and return the next checkpoint."""
    if processed < next_checkpoint and processed < total:
        return next_checkpoint
    elapsed = time.perf_counter() - started_at
    percent = 100.0 * processed / total if total else 100.0
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = (total - processed) / rate if rate > 0 else 0.0
    _log(
        f"[cyan]Prediction progress: {processed:,}/{total:,} windows "
        f"({percent:.0f}%), {rate:.2f} windows/s, "
        f"ETA {remaining:.0f}s."
    )
    return min(total, processed + max(1, total // 10))


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


def _aggregate_ensemble_statistics(
    probability_sum: np.ndarray,
    probability_sum_sq: np.ndarray,
    valid_counts: np.ndarray,
    model_count: int,
    ci_t_crit: Optional[float],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Aggregate partial GPU statistics without rebuilding member probabilities."""
    complete = valid_counts == model_count
    mean = np.full(probability_sum.shape, np.nan, dtype="float32")
    mean[complete] = (
        probability_sum[complete] / float(model_count)
    ).astype("float32", copy=False)
    if ci_t_crit is None or model_count < 2:
        return mean, None, None

    variance = np.full(probability_sum.shape, np.nan, dtype="float64")
    variance[complete] = (
        probability_sum_sq[complete]
        - (probability_sum[complete] ** 2) / float(model_count)
    ) / float(model_count - 1)
    variance[complete] = np.maximum(variance[complete], 0.0)
    half_width = np.full(probability_sum.shape, np.nan, dtype="float32")
    half_width[complete] = (
        ci_t_crit
        * np.sqrt(variance[complete] / float(model_count))
    ).astype("float32", copy=False)
    return (
        mean,
        np.clip(mean - half_width, 0.0, 1.0),
        np.clip(mean + half_width, 0.0, 1.0),
    )


def _predict_ensemble_block_statistics(
    stack: _RasterStack,
    models: List[Any],
    win: windows.Window,
    *,
    classes: np.ndarray,
    block_overlap: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a block and return partial probability statistics."""
    read_window, write_slice = brt._window_with_overlap(
        win,
        block_overlap=block_overlap,
        raster_width=stack.width,
        raster_height=stack.height,
    )
    block = stack.read(window=read_window, out_dtype="float32")
    return _predict_ensemble_array_statistics(
        block,
        models,
        stack.nodata_values,
        write_slice,
        int(win.height),
        int(win.width),
        classes,
    )


def _predict_ensemble_array_statistics(
    block: np.ndarray,
    models: List[Any],
    nodata_values: Sequence[Optional[float]],
    write_slice: Tuple[slice, slice],
    output_height: int,
    output_width: int,
    classes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return partial statistics from a block read by the parent process."""
    num_classes = len(classes)
    output_shape = (num_classes, output_height, output_width)
    if block.size == 0:
        return (
            np.zeros(output_shape, dtype="float64"),
            np.zeros(output_shape, dtype="float64"),
            np.zeros(output_shape, dtype="int16"),
        )

    samples = block.reshape(block.shape[0], -1).T
    valid = ~_nodata_pixel_mask(samples, nodata_values)
    probability_sum = np.zeros(output_shape, dtype="float64")
    probability_sum_sq = np.zeros(output_shape, dtype="float64")
    valid_counts = np.zeros(output_shape, dtype="int16")
    if np.any(valid):
        for model in models:
            probabilities = model.predict_proba(samples[valid]).astype(
                "float64", copy=False
            )
            probability_cube = np.full(
                (samples.shape[0], num_classes), np.nan, dtype="float64"
            )
            probability_cube[valid] = probabilities
            probability_cube = probability_cube.reshape(
                block.shape[1], block.shape[2], num_classes
            ).transpose(2, 0, 1)
            probability_cube = probability_cube[:, write_slice[0], write_slice[1]]
            finite = np.isfinite(probability_cube)
            probability_sum[finite] += probability_cube[finite]
            probability_sum_sq[finite] += probability_cube[finite] ** 2
            valid_counts += finite.astype("int16")

    return probability_sum, probability_sum_sq, valid_counts


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
    mean, lower, upper = _aggregate_ensemble_probabilities(
        probs_stack, ci_t_crit
    )

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
    model_paths: List[str],
    model_indices: List[int],
    classes: List[Any],
    nodata_values: List[Optional[float]],
    device_id: int,
    task_queue: Any,
    result_queue: Any,
) -> None:
    """Run assigned ensemble models on one persistent CUDA worker."""
    try:
        models = [joblib.load(model_paths[idx]) for idx in model_indices]
        expected_classes = set(classes)
        if any(set(getattr(model, "classes_", [])) != expected_classes for model in models):
            raise ValueError("Ensemble models have inconsistent class definitions.")
        for model in models:
            brt._configure_prediction_device(model, True, device_id=device_id)

        while True:
            task = task_queue.get()
            if task is None:
                return
            window_id, win_tuple, block, write_slice = task
            win = brt._tuple_to_window(win_tuple)
            probability_sum, probability_sum_sq, valid_counts = (
                _predict_ensemble_array_statistics(
                    block,
                    models,
                    nodata_values,
                    write_slice,
                    int(win.height),
                    int(win.width),
                    np.asarray(classes),
                )
            )
            result_queue.put(
                (
                    "result",
                    window_id,
                    model_indices,
                    probability_sum,
                    probability_sum_sq,
                    valid_counts,
                )
            )
    except BaseException as exc:
        result_queue.put(("error", repr(exc)))


def _ensemble_cpu_worker(
    model_paths: List[str],
    model_indices: List[int],
    classes: List[Any],
    nodata_values: List[Optional[float]],
    task_queue: Any,
    result_queue: Any,
) -> None:
    """Run assigned ensemble models on parent-read blocks in one CPU worker."""
    try:
        models = [joblib.load(model_paths[idx]) for idx in model_indices]
        expected_classes = set(classes)
        if any(
            set(getattr(model, "classes_", [])) != expected_classes
            for model in models
        ):
            raise ValueError("Ensemble models have inconsistent class definitions.")

        while True:
            task = task_queue.get()
            if task is None:
                return
            window_id, win_tuple, block, write_slice = task
            win = brt._tuple_to_window(win_tuple)
            probability_sum, probability_sum_sq, valid_counts = (
                _predict_ensemble_array_statistics(
                    block,
                    models,
                    nodata_values,
                    write_slice,
                    int(win.height),
                    int(win.width),
                    np.asarray(classes),
                )
            )
            result_queue.put(
                (
                    "result",
                    window_id,
                    model_indices,
                    probability_sum,
                    probability_sum_sq,
                    valid_counts,
                )
            )
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
        if block_overlap < 0:
            raise ValueError("block_overlap must be non-negative.")
        if block_shape is not None:
            if len(block_shape) != 2:
                raise ValueError("block_shape must contain (height, width).")
            if block_shape[0] <= 0 or block_shape[1] <= 0:
                raise ValueError("block_shape dimensions must be positive.")
        ensemble_path = Path(ensemble_dir)

        # Load metadata
        metadata_path = ensemble_path / "ensemble_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Ensemble metadata not found at {metadata_path}")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        self.metadata_ = metadata
        _log(f"[cyan]Loaded ensemble metadata from {metadata_path}.")

        # Load all models
        model_paths = sorted(ensemble_path.glob("brt_*.joblib"))
        expected_models = metadata.get("n_models", metadata.get("n_estimators", 1))
        if len(model_paths) != expected_models:
            raise ValueError(
                f"Expected {expected_models} models but found {len(model_paths)}"
            )
        n_models = len(model_paths)

        parallel_prediction = jobs > 1
        load_all_models = not parallel_prediction or feature_importance_out is not None
        models = [
            joblib.load(path)
            for path in (model_paths if load_all_models else model_paths[:1])
        ]
        _log(
            f"[cyan]Loaded {len(models)} ensemble model"
            f"{'s' if len(models) != 1 else ''} in the parent process."
        )
        cuda_devices = brt._available_cuda_devices() if gpu else []
        if gpu and not cuda_devices:
            raise RuntimeError(
                "GPU prediction requested, but no visible CUDA devices were found."
            )
        if gpu and jobs <= 1:
            for model in models:
                brt._configure_prediction_device(model, True, device_id=0)
            _log("[cyan]GPU prediction enabled on CUDA device 0.")
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
        if n_models > 1:
            ci_t_crit = float(stats.t.ppf(0.5 + ci_level / 2, df=n_models - 1))
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
            _log(
                f"[cyan]Opened raster stack: {stack.width:,}x{stack.height:,}, "
                f"{stack.count} bands."
            )

            classes = class_sets[0]
            classes_arr = np.asarray(sorted(classes))
            probs_profile = brt._prepare_prob_profile(
                stack.profile, count=len(classes_arr)
            )
            probs_profile["driver"] = "GTiff"

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

                with contextlib.nullcontext():
                    if block_shape:
                        block_h, block_w = block_shape
                        total_windows = (
                            (stack.height + block_h - 1) // block_h
                        ) * ((stack.width + block_w - 1) // block_w)

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
                        block_h, block_w = stack.template.block_shapes[0]
                        total_windows = (
                            (stack.height + block_h - 1) // block_h
                        ) * ((stack.width + block_w - 1) // block_w)
                        window_iter = (win for _, win in stack.block_windows(1))

                    _log(
                        f"[cyan]Processing {total_windows:,} windows with "
                        f"jobs={jobs}, block_overlap={block_overlap}."
                    )
                    prediction_started_at = time.perf_counter()
                    processed_windows = 0
                    next_checkpoint = 1

                    def _write_block_result(
                        win: windows.Window,
                        member_probs: Optional[np.ndarray],
                        mean_probs: np.ndarray,
                        lower_probs: Optional[np.ndarray],
                        upper_probs: Optional[np.ndarray],
                    ) -> None:
                        mean_dst.write(mean_probs, window=win)
                        if lower_dst is not None and lower_probs is not None:
                            lower_dst.write(lower_probs, window=win)
                        if upper_dst is not None and upper_probs is not None:
                            upper_dst.write(upper_probs, window=win)

                    if gpu and jobs > 1:
                        worker_count = min(jobs, n_models, len(cuda_devices))
                        assignments = _assign_models_to_workers(
                            n_models, worker_count
                        )
                        _log(
                            f"[cyan]Starting {worker_count} GPU workers on devices "
                            f"{cuda_devices[:worker_count]}; model assignments: "
                            f"{assignments}."
                        )
                        context = multiprocessing.get_context("spawn")
                        task_queues = [context.Queue(maxsize=2) for _ in assignments]
                        result_queue = context.Queue()
                        workers = [
                            context.Process(
                                target=_ensemble_gpu_worker,
                                args=(
                                    [str(path) for path in model_paths],
                                    assignment,
                                    classes_arr.tolist(),
                                    list(stack.nodata_values),
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
                        _log("[cyan]GPU workers started; prediction is underway.")
                        try:
                            pipeline_depth = 2
                            pending_windows: Dict[int, windows.Window] = {}
                            partial_results_by_window: Dict[
                                int, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]
                            ] = {}
                            next_window_id = 0
                            exhausted = False

                            def submit_gpu_window() -> bool:
                                nonlocal next_window_id, exhausted
                                if exhausted:
                                    return False
                                try:
                                    win = next(window_iter)
                                except StopIteration:
                                    exhausted = True
                                    return False
                                read_window, write_slice = brt._window_with_overlap(
                                    win,
                                    block_overlap=block_overlap,
                                    raster_width=stack.width,
                                    raster_height=stack.height,
                                )
                                task = (
                                    next_window_id,
                                    brt._window_to_tuple(win),
                                    stack.read(window=read_window, out_dtype="float32"),
                                    write_slice,
                                )
                                for task_queue in task_queues:
                                    task_queue.put(task)
                                pending_windows[next_window_id] = win
                                next_window_id += 1
                                return True

                            while len(pending_windows) < pipeline_depth:
                                if not submit_gpu_window():
                                    break

                            while pending_windows:
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
                                        f"completing a window; pids={dead}."
                                    ) from exc
                                if message[0] == "error":
                                    raise RuntimeError(
                                        f"GPU prediction worker failed: {message[1]}"
                                    )
                                (
                                    _,
                                    returned_window_id,
                                    indices,
                                    probability_sum,
                                    probability_sum_sq,
                                    valid_counts,
                                ) = message
                                if returned_window_id not in pending_windows:
                                    raise RuntimeError(
                                        "GPU prediction returned an unexpected window."
                                    )
                                partial_results = partial_results_by_window.setdefault(
                                    returned_window_id, {}
                                )
                                result_key = tuple(indices)
                                if result_key in partial_results:
                                    raise RuntimeError(
                                        "Duplicate GPU prediction from a worker."
                                    )
                                partial_results[result_key] = (
                                    probability_sum,
                                    probability_sum_sq,
                                    valid_counts,
                                )

                                if sum(
                                    len(model_indices)
                                    for model_indices in partial_results
                                ) == n_models:
                                    returned_models = [
                                        model_idx
                                        for model_indices in partial_results
                                        for model_idx in model_indices
                                    ]
                                    if sorted(returned_models) != list(
                                        range(n_models)
                                    ):
                                        raise RuntimeError(
                                            "GPU prediction returned an incomplete "
                                            "or duplicate model assignment."
                                        )
                                    win = pending_windows.pop(returned_window_id)
                                    partial_results_by_window.pop(returned_window_id)
                                    sums = [
                                        result[0] for result in partial_results.values()
                                    ]
                                    sum_squares = [
                                        result[1] for result in partial_results.values()
                                    ]
                                    counts = [
                                        result[2] for result in partial_results.values()
                                    ]
                                    mean_probs, lower_probs, upper_probs = (
                                        _aggregate_ensemble_statistics(
                                            np.sum(sums, axis=0),
                                            np.sum(sum_squares, axis=0),
                                            np.sum(counts, axis=0),
                                            n_models,
                                            ci_t_crit,
                                        )
                                    )
                                    _write_block_result(
                                        win,
                                        None,
                                        mean_probs,
                                        lower_probs,
                                        upper_probs,
                                    )
                                    processed_windows += 1
                                    next_checkpoint = _prediction_checkpoint(
                                        processed_windows,
                                        total_windows,
                                        prediction_started_at,
                                        next_checkpoint,
                                    )
                                    submit_gpu_window()
                        finally:
                            for task_queue in task_queues:
                                task_queue.put(None)
                            for worker in workers:
                                worker.join(timeout=10)
                                if worker.is_alive():
                                    worker.terminate()
                                    worker.join()
                    elif jobs > 1:
                        _log(
                            f"[cyan]Starting {jobs} CPU prediction workers; "
                            "prediction is underway."
                        )
                        worker_count = min(jobs, n_models)
                        assignments = _assign_models_to_workers(n_models, worker_count)
                        context = multiprocessing.get_context("spawn")
                        task_queues = [context.Queue(maxsize=2) for _ in assignments]
                        result_queue = context.Queue()
                        workers = [
                            context.Process(
                                target=_ensemble_cpu_worker,
                                args=(
                                    [str(path) for path in model_paths],
                                    assignment,
                                    classes_arr.tolist(),
                                    list(stack.nodata_values),
                                    task_queue,
                                    result_queue,
                                ),
                            )
                            for assignment, task_queue in zip(assignments, task_queues)
                        ]
                        for worker in workers:
                            worker.start()
                        try:
                            pipeline_depth = 2
                            pending_windows: Dict[int, windows.Window] = {}
                            partial_results_by_window: Dict[
                                int, Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray, np.ndarray]]
                            ] = {}
                            next_window_id = 0
                            exhausted = False

                            def submit_cpu_window() -> bool:
                                nonlocal next_window_id, exhausted
                                if exhausted:
                                    return False
                                try:
                                    win = next(window_iter)
                                except StopIteration:
                                    exhausted = True
                                    return False
                                read_window, write_slice = brt._window_with_overlap(
                                    win,
                                    block_overlap=block_overlap,
                                    raster_width=stack.width,
                                    raster_height=stack.height,
                                )
                                task = (
                                    next_window_id,
                                    brt._window_to_tuple(win),
                                    stack.read(window=read_window, out_dtype="float32"),
                                    write_slice,
                                )
                                for task_queue in task_queues:
                                    task_queue.put(task)
                                pending_windows[next_window_id] = win
                                next_window_id += 1
                                return True

                            while len(pending_windows) < pipeline_depth:
                                if not submit_cpu_window():
                                    break

                            while pending_windows:
                                try:
                                    message = result_queue.get(timeout=300)
                                except queue.Empty as exc:
                                    dead = [
                                        worker.pid
                                        for worker in workers
                                        if not worker.is_alive()
                                    ]
                                    raise RuntimeError(
                                        "CPU prediction worker stopped before "
                                        f"completing a window; pids={dead}."
                                    ) from exc
                                if message[0] == "error":
                                    raise RuntimeError(
                                        f"CPU prediction worker failed: {message[1]}"
                                    )
                                (
                                    _, returned_window_id, indices,
                                    probability_sum, probability_sum_sq, valid_counts,
                                ) = message
                                if returned_window_id not in pending_windows:
                                    raise RuntimeError(
                                        "CPU prediction returned an unexpected window."
                                    )
                                partial_results = partial_results_by_window.setdefault(
                                    returned_window_id, {}
                                )
                                result_key = tuple(indices)
                                if result_key in partial_results:
                                    raise RuntimeError(
                                        "Duplicate CPU prediction from a worker."
                                    )
                                partial_results[result_key] = (
                                    probability_sum, probability_sum_sq, valid_counts
                                )
                                if sum(
                                    len(model_indices)
                                    for model_indices in partial_results
                                ) == n_models:
                                    returned_models = [
                                        model_idx
                                        for model_indices in partial_results
                                        for model_idx in model_indices
                                    ]
                                    if sorted(returned_models) != list(range(n_models)):
                                        raise RuntimeError(
                                            "CPU prediction returned an incomplete "
                                            "or duplicate model assignment."
                                        )
                                    win = pending_windows.pop(returned_window_id)
                                    partial_results_by_window.pop(returned_window_id)
                                    mean_probs, lower_probs, upper_probs = (
                                        _aggregate_ensemble_statistics(
                                            np.sum(
                                                [result[0] for result in partial_results.values()],
                                                axis=0,
                                            ),
                                            np.sum(
                                                [result[1] for result in partial_results.values()],
                                                axis=0,
                                            ),
                                            np.sum(
                                                [result[2] for result in partial_results.values()],
                                                axis=0,
                                            ),
                                            n_models,
                                            ci_t_crit,
                                        )
                                    )
                                    _write_block_result(
                                        win, None, mean_probs, lower_probs, upper_probs
                                    )
                                    processed_windows += 1
                                    next_checkpoint = _prediction_checkpoint(
                                        processed_windows,
                                        total_windows,
                                        prediction_started_at,
                                        next_checkpoint,
                                    )
                                    submit_cpu_window()
                        finally:
                            for task_queue in task_queues:
                                task_queue.put(None)
                            for worker in workers:
                                worker.join(timeout=10)
                                if worker.is_alive():
                                    worker.terminate()
                                    worker.join()
                    else:
                        _log("[cyan]Running single-process prediction.")
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
                            processed_windows += 1
                            next_checkpoint = _prediction_checkpoint(
                                processed_windows,
                                total_windows,
                                prediction_started_at,
                                next_checkpoint,
                            )

                    elapsed = time.perf_counter() - prediction_started_at
                    _log(
                        f"[green]Prediction processing complete: "
                        f"{processed_windows:,}/{total_windows:,} windows in "
                        f"{elapsed:.1f}s."
                    )

        _log(f"[green]Mean probabilities saved to {mean_path}")
        if ci_t_crit is not None:
            _log(f"[green]Lower probability bounds saved to {lower_path}")
            _log(f"[green]Upper probability bounds saved to {upper_path}")
        return mean_path
