"""Boosted Regression Tree (XGBoost) training and inference utilities for raster stacks."""

from __future__ import annotations

import atexit
import concurrent.futures
import contextlib
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import joblib
import numpy as np
import rasterio
from rasterio import windows

from ..data.raster import (
    _RasterStack,
    _align_raster_to_grid,
    _expand_raster_inputs,
    _log,
    _nodata_pixel_mask,
    _normalize_nodata,
    _open_raster_stack,
)
from ..data.sampling import (
    _collect_training_samples,
    _collect_pseudo_absence_candidate_pool,
    _collect_pseudo_absence_samples,
    _compute_class_band_stats,
    _grid_thin_samples,
    _split_train_test,
)
from ..processing.evaluation import (
    _collect_holdout_metrics,
    _format_confusion_matrix,
    _log_holdout_metrics,
    _metrics_path,
    _misclassified_ids,
    _write_holdout_outputs,
    _write_point_gpkg,
)

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    raise ImportError(
        "xgboost is required for BRT models. Install with: "
        "pip install xgboost  (or pip install 'plaknit[gpu]' for GPU support)"
    )

try:  # pragma: no cover - optional rich dependency
    from rich.console import Console
except ImportError:  # pragma: no cover - fallback logging
    console = None
else:  # pragma: no cover
    console = Console()

PathLike = Union[str, Path]
WindowTuple = Tuple[int, int, int, int]


@dataclass
class _PreparedBRTData:
    X_presence: np.ndarray
    y_presence: np.ndarray
    presence_ids: np.ndarray
    presence_rows: np.ndarray
    presence_cols: np.ndarray
    candidate_features: np.ndarray
    candidate_rows: np.ndarray
    candidate_cols: np.ndarray
    train_shape: Tuple[int, int]
    train_transform: Any
    train_crs: Any
    decoder: Dict[int, int]


def _prepare_brt_training_data(
    image_path: Union[PathLike, Iterable[PathLike]],
    shapefile_path: PathLike,
    label_column: str,
    *,
    band_indices: Optional[Sequence[int]],
    training_buffer_meters: float,
    collect_candidates: bool = True,
) -> _PreparedBRTData:
    """Prepare deterministic BRT inputs shared by ensemble members."""

    with _open_raster_stack(image_path, band_indices=band_indices) as stack:
        assert stack.template is not None
        gdf = gpd.read_file(shapefile_path)
        if label_column not in gdf.columns:
            raise ValueError(f"Column '{label_column}' not found in training data.")
        if stack.crs is None:
            raise ValueError("Raster must have a valid CRS.")
        if gdf.crs is None:
            warnings.warn(
                "Vector training data lacks CRS. Assuming raster CRS.", UserWarning
            )
            gdf.set_crs(stack.crs, inplace=True)
        else:
            gdf = gdf.to_crs(stack.crs)

        label_values = set(gdf[label_column].dropna().tolist())
        if (
            not label_values
            or not label_values.issubset({0, 1})
            or 1 not in label_values
        ):
            raise ValueError(
                "BRT training requires presence labels coded as 1 and optional "
                "absence labels coded as 0."
            )
        code_column = "__plaknit_label_code__"
        gdf[code_column] = gdf[label_column].astype("int32")
        X, y, sample_ids, sample_rows, sample_cols = _collect_training_samples(
            stack, gdf, code_column, buffer_meters=training_buffer_meters
        )
        y = y.astype("int32", copy=False)
        if collect_candidates:
            (
                candidate_features,
                candidate_rows,
                candidate_cols,
            ) = _collect_pseudo_absence_candidate_pool(
                stack, gdf, buffer_meters=training_buffer_meters
            )
        else:
            candidate_features = np.empty((0, stack.count), dtype="float32")
            candidate_rows = np.empty(0, dtype="int32")
            candidate_cols = np.empty(0, dtype="int32")
        return _PreparedBRTData(
            X_presence=X,
            y_presence=y,
            presence_ids=sample_ids,
            presence_rows=sample_rows,
            presence_cols=sample_cols,
            candidate_features=candidate_features,
            candidate_rows=candidate_rows,
            candidate_cols=candidate_cols,
            train_shape=(stack.height, stack.width),
            train_transform=stack.transform,
            train_crs=stack.crs,
            decoder={0: 0, 1: 1},
        )


def _detect_gpu_available() -> bool:
    """Detect if GPU is available for XGBoost."""
    try:
        import cupy as cp  # noqa: F401

        return True
    except ImportError:
        return False


def train_brt(
    image_path: Union[PathLike, Iterable[PathLike]],
    shapefile_path: PathLike,
    label_column: str,
    model_out: PathLike,
    *,
    band_indices: Optional[Sequence[int]] = None,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 1,
    test_fraction: float = 0.3,
    grid_size: Optional[int] = None,
    pseudo_absence_ratio: float = 1.0,
    training_buffer_meters: float = 0.0,
    absence_buffer_meters: float = 0.0,
    gpu: bool = True,
    tree_method: Optional[str] = None,
    n_jobs: Optional[int] = None,
    _prepared: Optional[_PreparedBRTData] = None,
) -> xgb.XGBClassifier:
    """Train a Boosted Regression Tree (XGBoost) classifier on raster pixels under training polygons.

    The `image_path` can be a single multi-band raster, a directory of GeoTIFFs
    (expanded), or an iterable of coregistered rasters (elevation, NDVI,
    spectral bands, etc.). Use `band_indices` (1-based) to select a subset of
    stacked bands for training. A configurable fraction of samples is held out
    for evaluation and persisted with the model.

    BRT models are typically faster and more accurate than Random Forest, with
    optional GPU acceleration. See `plaknit.train_rf()` for RF-specific details.

    Parameters
    ----------
    image_path
        Path to raster file, directory of GeoTIFFs, or iterable of aligned rasters.
    shapefile_path
        Path to training polygon GeoPackage or Shapefile.
    label_column
        Name of the column in shapefile_path containing class labels.
    model_out
        Path where trained model will be saved (joblib format).
    band_indices
        Optional 1-based band indices to select subset of input bands.
    n_estimators
        Number of boosting rounds. Default 200 (typically 100-500 for raster classification).
    max_depth
        Maximum tree depth. Default 6 (deeper trees increase accuracy but risk overfitting).
    learning_rate
        Learning rate (eta). Default 0.1 (lower values ~0.01-0.05 for more stable models).
    subsample
        Fraction of samples per boosting round. Default 0.8.
    colsample_bytree
        Fraction of features per tree. Default 0.8.
    random_state
        Random seed for reproducibility.
    test_fraction
        Fraction of training samples held out for evaluation. Default 0.3.
    grid_size
        Optional pixel grid size for spatial thinning of training samples. Larger values
        reduce training set size for memory efficiency.
    pseudo_absence_ratio
        Number of generated pseudo-absence samples per presence sample. Default 1.0.
    training_buffer_meters
        Distance in meters around training geometries used to extract training pixels.
        The same area is excluded from pseudo-absence sampling. Default 0.
    absence_buffer_meters
        Deprecated alias for ``training_buffer_meters``. Default 0.
    gpu
        If True, attempt to use GPU acceleration via XGBoost's CUDA support. Gracefully
        falls back to CPU if GPU unavailable.
    tree_method
        Explicitly set tree method ('hist', 'gpu_hist', 'approx', etc.). If None, auto-selects
        based on gpu flag.
    n_jobs
        Number of XGBoost threads used by this model. If None, XGBoost selects
        its default thread count.

    Returns
    -------
    xgb.XGBClassifier
        Fitted classifier with metadata stored as attributes (label_decoder, band_indices,
        train_shape_, train_transform_, train_crs_, etc.).
    """

    _log("[bold cyan]Loading training data...")
    if training_buffer_meters and absence_buffer_meters:
        raise ValueError(
            "Specify only one of training_buffer_meters or absence_buffer_meters."
        )
    effective_buffer_meters = training_buffer_meters or absence_buffer_meters
    prepared = _prepared or _prepare_brt_training_data(
        image_path,
        shapefile_path,
        label_column,
        band_indices=band_indices,
        training_buffer_meters=effective_buffer_meters,
    )
    X = prepared.X_presence.copy()
    y = prepared.y_presence.copy()
    sample_ids = prepared.presence_ids.copy()
    sample_rows = prepared.presence_rows.copy()
    sample_cols = prepared.presence_cols.copy()
    candidate_features = prepared.candidate_features
    candidate_rows = prepared.candidate_rows
    candidate_cols = prepared.candidate_cols
    requested = int(np.floor(np.count_nonzero(y == 1) * pseudo_absence_ratio))
    if pseudo_absence_ratio < 0:
        raise ValueError("pseudo-absence ratio must be non-negative.")
    if requested == 0 and pseudo_absence_ratio > 0 and np.any(y == 1):
        requested = 1
    if requested > candidate_features.shape[0]:
        raise ValueError(
            f"Requested {requested} pseudo-absence samples, but only "
            f"{candidate_features.shape[0]} valid candidates are available."
        )
    if requested:
        rng = np.random.default_rng(random_state)
        selected = np.sort(
            rng.choice(candidate_features.shape[0], size=requested, replace=False)
        )
        X = np.vstack((X, candidate_features[selected]))
        y = np.concatenate((y, np.zeros(requested, dtype="int32")))
        sample_ids = np.concatenate(
            (
                sample_ids,
                np.asarray(
                    [f"pseudo_absence_{idx}" for idx in range(requested)], dtype=object
                ),
            )
        )
        sample_rows = np.concatenate((sample_rows, candidate_rows[selected]))
        sample_cols = np.concatenate((sample_cols, candidate_cols[selected]))
    generated_absence_count = requested
    train_shape = prepared.train_shape
    train_transform = prepared.train_transform
    train_crs = prepared.train_crs
    decoder = prepared.decoder

    if grid_size is not None:
        if grid_size < 1:
            raise ValueError("grid_size must be >= 1.")
        if grid_size > 1:
            _log(
                f"[bold cyan]Applying grid sampling (size={grid_size} px) for spatial "
                "diversity..."
            )
            before = X.shape[0]
            X, y, sample_ids, sample_rows, sample_cols = _grid_thin_samples(
                X,
                y,
                sample_ids,
                sample_rows,
                sample_cols,
                grid_size=grid_size,
                random_state=random_state,
            )
            _log(f"[bold cyan]Grid sampling kept {X.shape[0]:,} of {before:,} samples.")

    X_train, y_train, X_test, y_test, _extra_train, extra_test = _split_train_test(
        X,
        y,
        test_fraction=test_fraction,
        random_state=random_state,
        extra=[sample_ids, sample_rows, sample_cols],
    )
    if X_test is not None:
        _log(f"[bold cyan]Holding out {X_test.shape[0]:,} samples " "for evaluation.")

    # Configure XGBoost tree method and device
    if tree_method is None:
        if gpu and _detect_gpu_available():
            tree_method = "gpu_hist"
            _log("[green]GPU detected; using gpu_hist tree method.")
        else:
            tree_method = "hist"
            if gpu:
                _log(
                    "[yellow]GPU requested but not available; falling back to CPU (hist)."
                )

    _log(
        f"[bold cyan]Training BRT (XGBoost) on {X_train.shape[0]:,} samples "
        f"({X_train.shape[1]} bands, {n_estimators} rounds, max_depth={max_depth})..."
    )
    brt = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        tree_method=tree_method,
        n_jobs=n_jobs,
        eval_metric="auc",
        verbosity=0,
    )
    brt.fit(X_train, y_train)

    # Attach metadata (matching RF interface)
    brt.label_decoder = decoder  # type: ignore[attr-defined]
    if band_indices is not None:
        brt.band_indices = list(band_indices)  # type: ignore[attr-defined]
    brt.test_samples_ = X_test  # type: ignore[attr-defined]
    brt.test_labels_ = y_test  # type: ignore[attr-defined]
    brt.test_fraction_ = test_fraction  # type: ignore[attr-defined]
    brt.train_shape_ = train_shape  # type: ignore[attr-defined]
    brt.train_transform_ = train_transform  # type: ignore[attr-defined]
    brt.train_crs_ = train_crs  # type: ignore[attr-defined]
    brt.train_grid_size_ = grid_size  # type: ignore[attr-defined]
    brt.pseudo_absence_ratio_ = pseudo_absence_ratio  # type: ignore[attr-defined]
    brt.training_buffer_meters_ = effective_buffer_meters  # type: ignore[attr-defined]
    brt.absence_buffer_meters_ = effective_buffer_meters  # type: ignore[attr-defined]
    brt.pseudo_absence_count_ = generated_absence_count  # type: ignore[attr-defined]
    if extra_test is not None:
        brt.test_ids_ = extra_test[0]  # type: ignore[attr-defined]
        brt.test_rows_ = extra_test[1]  # type: ignore[attr-defined]
        brt.test_cols_ = extra_test[2]  # type: ignore[attr-defined]
    else:
        brt.test_ids_ = None  # type: ignore[attr-defined]
        brt.test_rows_ = None  # type: ignore[attr-defined]
        brt.test_cols_ = None  # type: ignore[attr-defined]

    if decoder:
        mapping_preview = ", ".join(
            f"{code}:{label}" for code, label in list(decoder.items())[:10]
        )
        _log(
            f"[green]Label codes => classes: {mapping_preview}"
            + (" ..." if len(decoder) > 10 else "")
        )

    classes = getattr(brt, "classes_", np.unique(y_train))
    class_stats = _compute_class_band_stats(X_train, y_train, classes=classes)
    band_ids = (
        list(band_indices)
        if band_indices is not None
        else list(range(1, X_train.shape[1] + 1))
    )
    class_stats["band_indices"] = np.asarray(band_ids, dtype=np.int64)
    brt.class_band_stats_ = class_stats  # type: ignore[attr-defined]

    missing = class_stats["counts"] == 0
    if np.any(missing):
        missing_classes = classes[missing]
        _log(
            "[yellow]Some classes have no training samples after sampling: "
            + ", ".join(str(val) for val in missing_classes)
        )
    _log("[green]Training complete. Saving model...")

    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    points_base = model_out.with_suffix("")
    if extra_test is not None and _extra_train is not None:
        train_ids, train_rows, train_cols = _extra_train
        val_ids, val_rows, val_cols = extra_test
        train_pred = brt.predict(X_train)
        _write_point_gpkg(
            points_base.with_name(f"{points_base.name}_train_points.gpkg"),
            train_rows,
            train_cols,
            y_train,
            train_pred,
            train_ids,
            transform=train_transform,
            crs=train_crs,
        )
        if X_test is not None and val_rows is not None:
            val_pred = brt.predict(X_test)
            _write_point_gpkg(
                points_base.with_name(f"{points_base.name}_val_points.gpkg"),
                val_rows,
                val_cols,
                y_test,
                val_pred,
                val_ids,
                transform=train_transform,
                crs=train_crs,
            )
    else:
        all_pred = brt.predict(X)
        _write_point_gpkg(
            points_base.with_name(f"{points_base.name}_train_points.gpkg"),
            sample_rows,
            sample_cols,
            y,
            all_pred,
            sample_ids,
            transform=train_transform,
            crs=train_crs,
        )

    joblib.dump(brt, model_out)
    _log(f"[green]Model saved to {model_out}")
    return brt


def _prepare_output_profile(
    profile: dict, dtype: str, nodata_value: Union[int, float]
) -> dict:
    profile = profile.copy()
    profile.update(count=1, dtype=dtype, nodata=nodata_value)
    return profile


def _prepare_prob_profile(profile: dict, *, count: int) -> dict:
    profile = profile.copy()
    profile.update(count=count, dtype="float32", nodata=np.nan)
    return profile


def _write_binary_class_outputs(
    classified_path: Path,
    output_dir: PathLike,
    class_values: Sequence[Union[int, float]],
) -> List[Path]:
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_directory / f"class_{class_value}.tif" for class_value in class_values
    ]

    with rasterio.open(classified_path) as src:
        profile = src.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0)
        with contextlib.ExitStack() as stack_ctx:
            destinations = [
                stack_ctx.enter_context(rasterio.open(path, "w", **profile))
                for path in outputs
            ]
            for _, window in src.block_windows(1):
                classified = src.read(1, window=window)
                for class_value, destination in zip(class_values, destinations):
                    binary = (classified == class_value).astype("uint8")
                    destination.write(binary, 1, window=window)

    return outputs


def _window_to_tuple(win: windows.Window) -> WindowTuple:
    return (
        int(win.col_off),
        int(win.row_off),
        int(win.width),
        int(win.height),
    )


def _tuple_to_window(win_tuple: WindowTuple) -> windows.Window:
    col_off, row_off, width, height = win_tuple
    return windows.Window(col_off=col_off, row_off=row_off, width=width, height=height)


def _window_with_overlap(
    win: windows.Window,
    *,
    block_overlap: int,
    raster_width: int,
    raster_height: int,
) -> Tuple[windows.Window, Tuple[slice, slice]]:
    if block_overlap <= 0:
        return win, (slice(None), slice(None))

    col_off = max(0, int(win.col_off) - block_overlap)
    row_off = max(0, int(win.row_off) - block_overlap)
    width = min(int(win.width) + 2 * block_overlap, raster_width - col_off)
    height = min(int(win.height) + 2 * block_overlap, raster_height - row_off)
    read_window = windows.Window(
        col_off=col_off, row_off=row_off, width=width, height=height
    )
    write_slice = (
        slice(int(win.row_off - row_off), int(win.row_off - row_off + win.height)),
        slice(int(win.col_off - col_off), int(win.col_off - col_off + win.width)),
    )
    return read_window, write_slice


def _predict_block(
    stack: "_RasterStack",
    model: xgb.XGBClassifier,
    win: windows.Window,
    *,
    out_dtype: str,
    nodata_value: Union[int, float],
    return_probs: bool,
    block_overlap: int,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Predict on a block of raster data."""
    read_window, write_slice = _window_with_overlap(
        win,
        block_overlap=block_overlap,
        raster_width=stack.width,
        raster_height=stack.height,
    )
    block = stack.read(window=read_window, out_dtype="float32")
    if block.size == 0:
        empty_pred = np.full(
            (int(win.height), int(win.width)), nodata_value, dtype=out_dtype
        )
        if not return_probs:
            return empty_pred
        class_values = getattr(model, "classes_", None)
        num_classes = len(class_values) if class_values is not None else 1
        empty_probs = np.full(
            (num_classes, int(win.height), int(win.width)),
            np.nan,
            dtype="float32",
        )
        return empty_pred, empty_probs

    samples = block.reshape(stack.count, -1).T
    valid = ~_nodata_pixel_mask(samples, stack.nodata_values)

    predictions = np.full(samples.shape[0], nodata_value, dtype=out_dtype)
    class_values = getattr(model, "classes_", None)
    if class_values is not None:
        class_values = np.asarray(class_values)
    num_classes = len(class_values) if class_values is not None else 1
    prob_out: Optional[np.ndarray] = None

    if np.any(valid):
        block_valid = valid.reshape(int(read_window.height), int(read_window.width))
        if return_probs:
            probs = model.predict_proba(samples[valid]).astype("float32", copy=False)
            num_classes = probs.shape[1]
            prob_cube = np.full(
                (samples.shape[0], num_classes), np.nan, dtype="float32"
            )
            prob_cube[valid] = probs
            prob_cube = prob_cube.reshape(
                int(read_window.height), int(read_window.width), num_classes
            )
            prob_out = prob_cube[write_slice[0], write_slice[1]].transpose(2, 0, 1)
            # Use argmax of probabilities for classification
            masked_probs = np.where(block_valid[:, :, None], prob_cube, -np.inf)
            best_idx = masked_probs.argmax(axis=2)
            if class_values is not None:
                best = class_values[best_idx]
            else:
                best = best_idx
            best[~block_valid] = nodata_value
            predictions = best[write_slice[0], write_slice[1]].astype(
                out_dtype, copy=False
            )
        else:
            preds = model.predict(samples[valid])
            predictions[valid] = preds.astype(out_dtype, copy=False)
            predictions = predictions.reshape(
                int(read_window.height), int(read_window.width)
            )
            predictions = predictions[write_slice[0], write_slice[1]]
    else:
        predictions = predictions.reshape(
            int(read_window.height), int(read_window.width)
        )
        predictions = predictions[write_slice[0], write_slice[1]]
        if return_probs:
            prob_out = np.full(
                (num_classes, int(win.height), int(win.width)),
                np.nan,
                dtype="float32",
            )

    if not return_probs:
        return predictions
    if prob_out is None:
        prob_out = np.full(
            (num_classes, int(win.height), int(win.width)),
            np.nan,
            dtype="float32",
        )
    return predictions, prob_out


_PREDICT_STACK: Optional[_RasterStack] = None
_PREDICT_MODEL: Optional[xgb.XGBClassifier] = None
_PREDICT_OUT_DTYPE: Optional[str] = None
_PREDICT_NODATA: Optional[Union[int, float]] = None
_PREDICT_RETURN_PROBS: bool = False


def _close_predict_worker() -> None:
    global _PREDICT_STACK
    if _PREDICT_STACK is not None:
        _PREDICT_STACK.__exit__(None, None, None)
        _PREDICT_STACK = None


def _init_predict_worker(
    image_paths: List[str],
    band_indices: Optional[Sequence[int]],
    model_path: str,
    out_dtype: str,
    nodata_value: Union[int, float],
    return_probs: bool,
) -> None:
    global _PREDICT_STACK
    global _PREDICT_MODEL
    global _PREDICT_OUT_DTYPE
    global _PREDICT_NODATA
    global _PREDICT_RETURN_PROBS

    stack = _open_raster_stack(image_paths, band_indices=band_indices)
    stack.__enter__()
    _PREDICT_STACK = stack
    _PREDICT_MODEL = joblib.load(model_path)
    _PREDICT_OUT_DTYPE = out_dtype
    _PREDICT_NODATA = nodata_value
    _PREDICT_RETURN_PROBS = return_probs
    atexit.register(_close_predict_worker)


def _predict_block_worker(
    win_tuple: WindowTuple, block_overlap: int
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if _PREDICT_STACK is None or _PREDICT_MODEL is None:
        raise RuntimeError("Prediction worker not initialized.")
    if _PREDICT_OUT_DTYPE is None or _PREDICT_NODATA is None:
        raise RuntimeError("Prediction worker missing output settings.")

    win = _tuple_to_window(win_tuple)
    return _predict_block(
        _PREDICT_STACK,
        _PREDICT_MODEL,
        win,
        out_dtype=_PREDICT_OUT_DTYPE,
        nodata_value=_PREDICT_NODATA,
        return_probs=_PREDICT_RETURN_PROBS,
        block_overlap=block_overlap,
    )


def predict_brt(
    image_path: Union[PathLike, Iterable[PathLike]],
    model_path: PathLike,
    output_path: PathLike,
    *,
    band_indices: Optional[Sequence[int]] = None,
    block_shape: Optional[Tuple[int, int]] = None,
    probs_out: Optional[PathLike] = None,
    binary_out: Optional[PathLike] = None,
    block_overlap: int = 0,
    jobs: int = 1,
) -> Path:
    """Apply a trained BRT (XGBoost) model to a raster stack and write a classified GeoTIFF.

    The `image_path` can be a single raster, a directory of GeoTIFFs, or an
    iterable of aligned rasters. `jobs` controls block-level parallelism via
    worker processes. Use `band_indices` (1-based) to select a subset of stacked
    bands for prediction. If `probs_out` is set, prediction writes a multi-band
    GeoTIFF of class probabilities (XGBoost posteriors). If `binary_out` is set,
    write one uint8 binary GeoTIFF per class into that directory. If the model
    includes holdout samples, prediction logs a confusion matrix and feature
    importance.

    Note: BRT prediction does not support spatial smoothing (MRF/Bayes). For
    smoothing, use `plaknit.smooth_probs()` on the probability outputs.
    """

    _log("[bold cyan]Loading model...")
    model: xgb.XGBClassifier = joblib.load(model_path)
    classes = getattr(model, "classes_", None)
    classes_dtype = getattr(classes, "dtype", np.int32)
    if np.issubdtype(classes_dtype, np.integer):
        out_dtype = "int16"
        nodata_value: Union[int, float] = -1
    else:
        out_dtype = "float32"
        nodata_value = np.nan

    if block_overlap < 0:
        raise ValueError("block_overlap must be non-negative.")

    if jobs is None:
        jobs = 1
    if jobs <= 0:
        jobs = max(1, os.cpu_count() or 1)

    selected_band_indices: Optional[List[int]]
    if band_indices is not None:
        selected_band_indices = list(band_indices)
    else:
        model_band_indices = getattr(model, "band_indices", None)
        selected_band_indices = list(model_band_indices) if model_band_indices else None

    out_path = Path(output_path)
    if out_path.suffix.lower() == ".vrt":
        out_path = out_path.with_suffix(".tif")
        _log("[yellow]Output path ended with .vrt; writing GeoTIFF to .tif instead.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    probs_path: Optional[Path] = None
    if probs_out is not None:
        probs_path = Path(probs_out)
        if probs_path.suffix.lower() == ".vrt":
            probs_path = probs_path.with_suffix(".tif")
            _log(
                "[yellow]Probability output path ended with .vrt; writing GeoTIFF to "
                ".tif instead."
            )
        probs_path.parent.mkdir(parents=True, exist_ok=True)

    with _open_raster_stack(image_path, band_indices=selected_band_indices) as stack:
        assert stack.template is not None
        expected_features = getattr(model, "n_features_in_", None)
        if expected_features is not None and expected_features != stack.count:
            raise ValueError(
                "Model expects "
                f"{expected_features} bands, but input stack has {stack.count}. "
                "Train a matching model or pass --band-indices to select "
                "the same bands used during training."
            )
        metrics = _log_holdout_metrics(model)

        profile = _prepare_output_profile(stack.profile, out_dtype, nodata_value)
        probs_profile: Optional[dict] = None
        if probs_path is not None:
            num_classes = len(classes) if classes is not None else 1
            probs_profile = _prepare_prob_profile(stack.profile, count=num_classes)
        # Ensure we write a GeoTIFF even when reading from a VRT source.
        profile["driver"] = "GTiff"
        if probs_profile is not None:
            probs_profile["driver"] = "GTiff"
        return_probs = probs_path is not None

        with contextlib.ExitStack() as stack_ctx:
            dst = stack_ctx.enter_context(rasterio.open(out_path, "w", **profile))
            probs_dst = (
                stack_ctx.enter_context(rasterio.open(probs_path, "w", **probs_profile))
                if probs_path is not None and probs_profile is not None
                else None
            )
            _log("[bold cyan]Predicting classes...")
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

            if jobs > 1:
                max_workers = jobs
                max_pending = max_workers * 2
                image_paths = [str(path) for path in stack.paths]
                model_path_str = str(model_path)
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=_init_predict_worker,
                    initargs=(
                        image_paths,
                        selected_band_indices,
                        model_path_str,
                        out_dtype,
                        nodata_value,
                        return_probs,
                    ),
                ) as executor:
                    futures: Dict[concurrent.futures.Future, windows.Window] = {}
                    for win in window_iter:
                        future = executor.submit(
                            _predict_block_worker,
                            _window_to_tuple(win),
                            block_overlap,
                        )
                        futures[future] = win
                        if len(futures) >= max_pending:
                            done, _ = concurrent.futures.wait(
                                futures, return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            for finished in done:
                                result = finished.result()
                                if return_probs and probs_dst is not None:
                                    predictions, probs = result
                                    dst.write(predictions, 1, window=futures[finished])
                                    probs_dst.write(probs, window=futures[finished])
                                else:
                                    predictions = result
                                    dst.write(predictions, 1, window=futures[finished])
                                del futures[finished]

                    for finished in concurrent.futures.as_completed(futures):
                        result = finished.result()
                        if return_probs and probs_dst is not None:
                            predictions, probs = result
                            dst.write(predictions, 1, window=futures[finished])
                            probs_dst.write(probs, window=futures[finished])
                        else:
                            predictions = result
                            dst.write(predictions, 1, window=futures[finished])
            else:
                for win in window_iter:
                    result = _predict_block(
                        stack,
                        model,
                        win,
                        out_dtype=out_dtype,
                        nodata_value=nodata_value,
                        return_probs=return_probs,
                        block_overlap=block_overlap,
                    )
                    if return_probs and probs_dst is not None:
                        predictions, probs = result
                        dst.write(predictions, 1, window=win)
                        probs_dst.write(probs, window=win)
                    else:
                        predictions = result
                        dst.write(predictions, 1, window=win)

    if metrics is not None:
        _write_holdout_outputs(
            metrics,
            out_path,
            smoothed=None,
            smoothed_note="BRT prediction does not include spatial smoothing.",
            smooth="none",
        )

    if binary_out is not None:
        class_values = getattr(model, "classes_", None)
        if class_values is None:
            raise ValueError("Model must define classes_ to write binary outputs.")
        binary_paths = _write_binary_class_outputs(out_path, binary_out, class_values)
        for binary_path in binary_paths:
            _log(f"[green]Binary class mask saved to {binary_path}")

    _log(f"[green]Classification saved to {out_path}")
    return out_path
