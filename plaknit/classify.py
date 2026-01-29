"""Random Forest training and inference utilities for raster stacks."""

from __future__ import annotations

import atexit
import concurrent.futures
import contextlib
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import joblib
import numpy as np
import rasterio
from rasterio import features, windows
from rasterio.features import geometry_window
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

try:  # pragma: no cover - optional rich dependency
    from rich.console import Console
except ImportError:  # pragma: no cover - fallback logging
    console = None
else:  # pragma: no cover
    console = Console()

PathLike = Union[str, Path]
NeighborOffsets = Tuple[Tuple[int, int], ...]
WindowTuple = Tuple[int, int, int, int]


def _log(message: str) -> None:
    if console is not None:
        console.log(message)
    else:
        print(message)


def _normalize_nodata(
    nodata_values: Optional[Union[float, Iterable[Optional[float]]]],
    band_count: int,
) -> List[Optional[float]]:
    """Normalize nodata to a list per band."""

    if nodata_values is None:
        return [None] * band_count

    if isinstance(nodata_values, Iterable) and not isinstance(
        nodata_values, (str, bytes)
    ):
        nodata_list = list(nodata_values)
    else:
        nodata_list = [nodata_values]

    if len(nodata_list) == 1 and band_count > 1:
        nodata_list *= band_count
    elif len(nodata_list) < band_count:
        nodata_list.extend([None] * (band_count - len(nodata_list)))

    return nodata_list[:band_count]


def _nodata_pixel_mask(
    samples: np.ndarray,
    nodata_values: Optional[Union[float, Iterable[Optional[float]]]],
) -> np.ndarray:
    """Return a boolean mask of pixels touching nodata for any band."""

    nodata_per_band = _normalize_nodata(nodata_values, samples.shape[1])
    if all(v is None for v in nodata_per_band):
        return np.zeros(samples.shape[0], dtype=bool)

    mask = np.zeros(samples.shape[0], dtype=bool)
    for band_idx, nd_val in enumerate(nodata_per_band):
        if nd_val is None:
            continue
        if np.isnan(nd_val):
            mask |= np.isnan(samples[:, band_idx])
        else:
            mask |= samples[:, band_idx] == nd_val
    return mask


def _neighbor_offsets(neighborhood: int) -> NeighborOffsets:
    if neighborhood == 4:
        return ((-1, 0), (1, 0), (0, -1), (0, 1))
    if neighborhood == 8:
        return (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        )
    raise ValueError("neighborhood must be 4 or 8.")


def _icm_smooth(
    log_probs: np.ndarray,
    init_labels: np.ndarray,
    valid_mask: np.ndarray,
    *,
    beta: float,
    neighborhood: int,
    iters: int,
) -> np.ndarray:
    """Iterated Conditional Modes for a Potts MRF prior."""

    h, w, num_classes = log_probs.shape
    labels = init_labels.copy()
    offsets = _neighbor_offsets(neighborhood)

    for _ in range(max(1, iters)):
        counts = np.zeros((h, w, num_classes), dtype=np.int16)
        for dr, dc in offsets:
            neighbor = np.full_like(labels, -1)
            if dr >= 0:
                r_src = slice(0, h - dr)
                r_dst = slice(dr, h)
            else:
                r_src = slice(-dr, h)
                r_dst = slice(0, h + dr)
            if dc >= 0:
                c_src = slice(0, w - dc)
                c_dst = slice(dc, w)
            else:
                c_src = slice(-dc, w)
                c_dst = slice(0, w + dc)

            neighbor[r_dst, c_dst] = labels[r_src, c_src]
            neighbor_valid = np.zeros_like(valid_mask, dtype=bool)
            neighbor_valid[r_dst, c_dst] = valid_mask[r_src, c_src]

            for k in range(num_classes):
                counts[:, :, k] += ((neighbor == k) & neighbor_valid).astype(np.int16)

        energies = log_probs + beta * counts
        best = energies.argmax(axis=2)
        labels[valid_mask] = best[valid_mask]

    return labels


def _bayes_smooth(
    prob_cube: np.ndarray,
    valid_mask: np.ndarray,
    *,
    window_size: int,
    neigh_fraction: float,
    smoothness: Union[float, Sequence[float]],
) -> np.ndarray:
    """Empirical Bayes smoothing of class probabilities (SITS book)."""

    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("bayes_window_size must be an odd integer >= 3.")
    if not (0 < neigh_fraction <= 1):
        raise ValueError("bayes_neigh_fraction must be in (0, 1].")

    h, w, num_classes = prob_cube.shape
    if isinstance(smoothness, Sequence) and not isinstance(smoothness, (str, bytes)):
        smoothness_arr = np.asarray(smoothness, dtype="float32")
        if smoothness_arr.size != num_classes:
            raise ValueError("bayes_smoothness must match number of classes.")
    else:
        smoothness_arr = np.full(num_classes, float(smoothness), dtype="float32")

    if np.any(smoothness_arr < 0):
        raise ValueError("bayes_smoothness must be non-negative.")

    if np.all(smoothness_arr == 0):
        return prob_cube

    eps = np.float32(1e-6)
    probs = np.clip(prob_cube, eps, 1 - eps).astype("float32", copy=False)
    logits = np.log(probs) - np.log1p(-probs)

    pad = window_size // 2
    padded_probs = np.pad(
        probs, ((pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=np.nan
    )
    padded_logits = np.pad(
        logits,
        ((pad, pad), (pad, pad), (0, 0)),
        mode="constant",
        constant_values=np.nan,
    )
    padded_valid = np.pad(
        valid_mask, ((pad, pad), (pad, pad)), mode="constant", constant_values=False
    )

    out_logits = np.full_like(logits, np.nan, dtype="float32")

    for row in range(h):
        r_slice = slice(row, row + window_size)
        for col in range(w):
            if not valid_mask[row, col]:
                continue
            c_slice = slice(col, col + window_size)
            win_valid = padded_valid[r_slice, c_slice]
            if not np.any(win_valid):
                out_logits[row, col] = logits[row, col]
                continue

            for k in range(num_classes):
                win_probs = padded_probs[r_slice, c_slice, k]
                win_logits = padded_logits[r_slice, c_slice, k]
                mask = win_valid & ~np.isnan(win_probs)
                if not np.any(mask):
                    out_logits[row, col, k] = logits[row, col, k]
                    continue

                flat_probs = win_probs[mask]
                flat_logits = win_logits[mask]
                keep = max(1, int(np.ceil(neigh_fraction * flat_probs.size)))
                if keep < flat_probs.size:
                    idx = np.argpartition(flat_probs, -keep)[-keep:]
                    selected_logits = flat_logits[idx]
                else:
                    selected_logits = flat_logits

                mean_val = float(selected_logits.mean())
                if selected_logits.size > 1:
                    var_val = float(selected_logits.var(ddof=1))
                else:
                    var_val = 0.0

                sigma2 = float(smoothness_arr[k])
                denom = var_val + sigma2
                if denom <= 0:
                    out_logits[row, col, k] = logits[row, col, k]
                else:
                    out_logits[row, col, k] = (var_val / denom) * logits[
                        row, col, k
                    ] + (sigma2 / denom) * mean_val

    return 1 / (1 + np.exp(-out_logits))


class _RasterStack:
    """Lightweight reader that stacks multiple rasters band-wise."""

    def __init__(self, paths: List[Path], band_indices: Optional[Sequence[int]] = None):
        self.paths = paths
        self.datasets: List[rasterio.io.DatasetReader] = []
        self.count = 0
        self.nodata_values: List[Optional[float]] = []
        self.template: Optional[rasterio.io.DatasetReader] = None
        self._requested_band_indices = (
            list(band_indices) if band_indices is not None else None
        )
        self._selected_band_map: Optional[List[Tuple[int, int]]] = None
        self._all_band_map: List[Tuple[int, int]] = []

    def __enter__(self) -> "_RasterStack":
        self.datasets = [rasterio.open(p) for p in self.paths]
        if not self.datasets:
            raise ValueError("No raster paths were provided.")

        self.template = self.datasets[0]
        template_shape = (self.template.width, self.template.height)
        template_transform = self.template.transform
        template_crs = self.template.crs

        for ds in self.datasets[1:]:
            if (ds.width, ds.height) != template_shape:
                raise ValueError("All rasters must have the same dimensions.")
            if not np.allclose(ds.transform, template_transform):
                raise ValueError("All rasters must share the same transform/grid.")
            if (
                template_crs is not None
                and ds.crs is not None
                and ds.crs != template_crs
            ):
                raise ValueError("All rasters must share the same CRS.")

        for ds_idx, ds in enumerate(self.datasets):
            self.count += ds.count
            self.nodata_values.extend(_normalize_nodata(ds.nodatavals, ds.count))
            self._all_band_map.extend(
                (ds_idx, band_idx) for band_idx in range(1, ds.count + 1)
            )

        if self._requested_band_indices is not None:
            band_indices = [int(idx) for idx in self._requested_band_indices]
            if not band_indices:
                raise ValueError("band_indices must include at least one band.")
            if len(set(band_indices)) != len(band_indices):
                raise ValueError("band_indices must not contain duplicates.")
            total_count = self.count
            for idx in band_indices:
                if idx < 1 or idx > total_count:
                    raise ValueError(
                        f"band_indices must be between 1 and {total_count}."
                    )
            self._selected_band_map = [
                self._all_band_map[idx - 1] for idx in band_indices
            ]
            self.nodata_values = [self.nodata_values[idx - 1] for idx in band_indices]
            self.count = len(self._selected_band_map)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for ds in self.datasets:
            ds.close()

    @property
    def width(self) -> int:
        assert self.template is not None
        return self.template.width

    @property
    def height(self) -> int:
        assert self.template is not None
        return self.template.height

    @property
    def crs(self):
        assert self.template is not None
        return self.template.crs

    @property
    def transform(self):
        assert self.template is not None
        return self.template.transform

    @property
    def profile(self) -> dict:
        assert self.template is not None
        return self.template.profile

    def block_windows(self, bidx: int = 1):
        assert self.template is not None
        return self.template.block_windows(bidx)

    def read(self, *, window, out_dtype: str) -> np.ndarray:
        if self._selected_band_map is None:
            blocks: List[np.ndarray] = []
            for ds in self.datasets:
                blocks.append(ds.read(window=window, out_dtype=out_dtype))
            return np.concatenate(blocks, axis=0)

        height = int(window.height)
        width = int(window.width)
        out = np.empty((self.count, height, width), dtype=out_dtype)
        read_plan: Dict[int, List[Tuple[int, int]]] = {}
        for out_idx, (ds_idx, band_idx) in enumerate(self._selected_band_map):
            read_plan.setdefault(ds_idx, []).append((out_idx, band_idx))

        for ds_idx, selections in read_plan.items():
            ds = self.datasets[ds_idx]
            band_ids = [band_idx for _, band_idx in selections]
            data = ds.read(indexes=band_ids, window=window, out_dtype=out_dtype)
            if data.ndim == 2:
                data = data[np.newaxis, :, :]
            for local_idx, (out_idx, _) in enumerate(selections):
                out[out_idx] = data[local_idx]

        return out


def _expand_raster_inputs(
    image_path: Union[PathLike, Iterable[PathLike]],
) -> List[Path]:
    """Normalize raster inputs to a list of Paths.

    Accepts a single file/VRT, a directory (expands *.tif / *.tiff), or an
    iterable of mixed paths (files or directories). Directories must contain
    at least one GeoTIFF.
    """

    paths: List[Path] = []

    def add_path(p: Path) -> None:
        if p.is_dir():
            candidates = sorted([*p.glob("*.tif"), *p.glob("*.tiff")])
            if not candidates:
                raise ValueError(f"No GeoTIFFs found in directory: {p}")
            paths.extend(candidates)
        elif p.is_file():
            paths.append(p)
        else:
            raise ValueError(f"Raster path not found: {p}")

    if isinstance(image_path, Iterable) and not isinstance(
        image_path, (str, bytes, Path)
    ):
        for item in image_path:
            add_path(Path(item))
    else:
        add_path(Path(image_path))  # type: ignore[arg-type]

    if not paths:
        raise ValueError("No raster paths were provided.")

    return paths


def _open_raster_stack(
    image_path: Union[PathLike, Iterable[PathLike]],
    band_indices: Optional[Sequence[int]] = None,
) -> _RasterStack:
    return _RasterStack(_expand_raster_inputs(image_path), band_indices=band_indices)


def _collect_training_samples(
    stack: _RasterStack,
    gdf: gpd.GeoDataFrame,
    label_column: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-pixel samples under each training geometry.

    Returns features, labels, source geometry ids, row indices, and column indices.
    """

    feature_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    id_chunks: List[np.ndarray] = []
    row_chunks: List[np.ndarray] = []
    col_chunks: List[np.ndarray] = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        label_value = row[label_column]
        if label_value in (None, 0):
            continue

        try:
            assert stack.template is not None
            win = geometry_window(stack.template, [geom], north_up=True, rotated=False)
        except ValueError:
            _log(f"[yellow]Skipping geometry {idx}: outside raster bounds.")
            continue

        if win.width == 0 or win.height == 0:
            continue

        data = stack.read(window=win, out_dtype="float32")
        if data.size == 0:
            continue

        block_transform = windows.transform(win, stack.transform)
        label_block = features.rasterize(
            [(geom, label_value)],
            out_shape=(win.height, win.width),
            transform=block_transform,
            fill=0,
            dtype="int32",
        )

        label_flat = label_block.reshape(-1)
        valid = label_flat != 0
        if not np.any(valid):
            continue

        samples = data.reshape(stack.count, -1).T
        valid &= ~_nodata_pixel_mask(samples, stack.nodata_values)

        if not np.any(valid):
            continue

        feature_chunks.append(samples[valid])
        label_chunks.append(label_flat[valid])
        id_chunks.append(np.full(np.count_nonzero(valid), idx, dtype=object))

        valid_mask = valid.reshape(int(win.height), int(win.width))
        rows, cols = np.where(valid_mask)
        row_chunks.append(rows + int(win.row_off))
        col_chunks.append(cols + int(win.col_off))

    if not feature_chunks:
        raise ValueError("No training samples were extracted. Check label geometries.")

    features_arr = np.vstack(feature_chunks)
    labels_arr = np.concatenate(label_chunks)
    ids_arr = np.concatenate(id_chunks)
    rows_arr = np.concatenate(row_chunks).astype("int32", copy=False)
    cols_arr = np.concatenate(col_chunks).astype("int32", copy=False)
    return features_arr, labels_arr, ids_arr, rows_arr, cols_arr


def _grid_thin_samples(
    features: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    grid_size: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if grid_size <= 1:
        return features, labels, sample_ids, rows, cols

    rng = np.random.default_rng(random_state)
    indices = np.arange(labels.shape[0])
    rng.shuffle(indices)

    seen: set[Tuple[int, int, int]] = set()
    keep: List[int] = []
    for idx in indices:
        key = (
            int(labels[idx]),
            int(rows[idx] // grid_size),
            int(cols[idx] // grid_size),
        )
        if key in seen:
            continue
        seen.add(key)
        keep.append(int(idx))

    keep_idx = np.asarray(keep, dtype=np.int64)
    keep_idx.sort()
    return (
        features[keep_idx],
        labels[keep_idx],
        sample_ids[keep_idx],
        rows[keep_idx],
        cols[keep_idx],
    )


def _split_train_test(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    test_fraction: float,
    random_state: int,
    extra: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[List[np.ndarray]],
    Optional[List[np.ndarray]],
]:
    if test_fraction <= 0:
        return features, labels, None, None, None, None
    if test_fraction >= 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    extra = list(extra) if extra is not None else []
    unique_labels = np.unique(labels)
    stratify = labels if unique_labels.size > 1 else None
    indices = np.arange(labels.shape[0])
    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_fraction,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        _log("[yellow]Stratified split failed; falling back to unstratified holdout.")
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_fraction,
            random_state=random_state,
            stratify=None,
        )

    X_train = features[train_idx]
    y_train = labels[train_idx]
    X_test = features[test_idx]
    y_test = labels[test_idx]

    if extra:
        extra_train = [arr[train_idx] for arr in extra]
        extra_test = [arr[test_idx] for arr in extra]
    else:
        extra_train = None
        extra_test = None

    return X_train, y_train, X_test, y_test, extra_train, extra_test


def _compute_class_band_stats(
    features: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
) -> Dict[str, np.ndarray]:
    num_classes = classes.shape[0]
    num_bands = features.shape[1]
    counts = np.zeros(num_classes, dtype=np.int64)
    means = np.full((num_classes, num_bands), np.nan, dtype="float64")
    stds = np.full((num_classes, num_bands), np.nan, dtype="float64")

    for idx, class_value in enumerate(classes):
        mask = labels == class_value
        if not np.any(mask):
            continue
        class_samples = features[mask]
        counts[idx] = class_samples.shape[0]
        means[idx] = np.nanmean(class_samples, axis=0)
        stds[idx] = np.nanstd(class_samples, axis=0)

    return {"classes": classes, "counts": counts, "mean": means, "std": stds}


def _format_confusion_matrix(matrix: np.ndarray, labels: Sequence[str]) -> str:
    label_strings = [str(label) for label in labels]
    max_label = max((len(label) for label in label_strings), default=0)
    max_value = max((len(str(val)) for val in matrix.flatten()), default=1)
    width = max(max_label, max_value)

    header = " " * (width + 1) + " ".join(label.rjust(width) for label in label_strings)
    rows = [header]
    for label, row in zip(label_strings, matrix):
        row_values = " ".join(str(val).rjust(width) for val in row)
        rows.append(f"{label.rjust(width)} {row_values}")
    return "\n".join(rows)


def _collect_holdout_metrics(
    model: RandomForestClassifier,
) -> Optional[Dict[str, Any]]:
    test_samples = getattr(model, "test_samples_", None)
    test_labels = getattr(model, "test_labels_", None)
    if test_samples is None or test_labels is None or len(test_labels) == 0:
        return None

    predictions = model.predict(test_samples)
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = np.unique(test_labels)
    decoder = getattr(model, "label_decoder", None)
    if decoder:
        label_names = [str(decoder.get(int(code), code)) for code in classes]
    else:
        label_names = [str(code) for code in classes]

    matrix = confusion_matrix(test_labels, predictions, labels=classes)
    accuracy = accuracy_score(test_labels, predictions)

    importances = getattr(model, "feature_importances_", None)
    bands: Optional[List[Tuple[int, float]]]
    if importances is None:
        bands = None
    else:
        band_indices = getattr(model, "band_indices", None)
        if band_indices is None:
            band_indices = list(range(1, len(importances) + 1))
        bands = list(zip(band_indices, importances))
        bands.sort(key=lambda pair: pair[1], reverse=True)

    return {
        "sample_count": len(test_labels),
        "accuracy": float(accuracy),
        "labels": label_names,
        "classes": classes,
        "matrix": matrix,
        "band_importances": bands,
        "test_labels": test_labels,
        "predictions": predictions,
        "test_ids": getattr(model, "test_ids_", None),
        "test_rows": getattr(model, "test_rows_", None),
        "test_cols": getattr(model, "test_cols_", None),
        "train_shape": getattr(model, "train_shape_", None),
        "train_transform": getattr(model, "train_transform_", None),
        "train_crs": getattr(model, "train_crs_", None),
        "train_grid_size": getattr(model, "train_grid_size_", None),
        "class_band_stats": getattr(model, "class_band_stats_", None),
    }


def _log_holdout_metrics(model: RandomForestClassifier) -> Optional[Dict[str, Any]]:
    metrics = _collect_holdout_metrics(model)
    if metrics is None:
        _log("[yellow]Model has no holdout samples; skipping evaluation.")
        return None

    _log(
        f"[bold cyan]Holdout evaluation: {metrics['sample_count']:,} samples, "
        f"accuracy {metrics['accuracy']:.3f}"
    )
    _log("[bold cyan]Confusion matrix (rows=true, cols=pred):")
    _log(_format_confusion_matrix(metrics["matrix"], metrics["labels"]))

    bands = metrics["band_importances"]
    if bands is None:
        _log("[yellow]Model lacks feature_importances_; skipping band importance.")
    else:
        lines = ["[bold cyan]Band importance (sorted):"]
        for band_idx, importance in bands:
            lines.append(f"band {band_idx}: {importance:.6f}")
        _log("\n".join(lines))

    return metrics


def _metrics_path(out_path: Path) -> Path:
    base = out_path.with_suffix("")
    metrics_txt = base.with_name(f"{base.name}_metrics.txt")
    return metrics_txt


def _write_point_gpkg(
    out_path: Path,
    rows: np.ndarray,
    cols: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    sample_ids: Optional[np.ndarray],
    *,
    transform: Any,
    crs: Optional[Any],
) -> None:
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    data: Dict[str, Any] = {
        "true": np.asarray(true_labels),
        "pred": np.asarray(pred_labels),
    }
    if sample_ids is not None:
        data["source_id"] = np.asarray(sample_ids)

    geometry = gpd.points_from_xy(xs, ys, crs=crs)
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=crs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG", index=False)
    _log(f"[green]Wrote sample points to {out_path}.")


def _misclassified_ids(
    labels: np.ndarray,
    predictions: np.ndarray,
    sample_ids: Optional[np.ndarray],
) -> Optional[List[str]]:
    if sample_ids is None:
        return None
    mismatched = labels != predictions
    if not np.any(mismatched):
        return []
    unique_ids = np.unique(sample_ids[mismatched])
    return [str(value) for value in unique_ids]


def _collect_smoothed_holdout_metrics(
    metrics: Dict[str, Any],
    out_path: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    test_rows = metrics.get("test_rows")
    test_cols = metrics.get("test_cols")
    if test_rows is None or test_cols is None:
        return None, "Holdout sample locations missing from model."

    test_rows_arr = np.asarray(test_rows)
    test_cols_arr = np.asarray(test_cols)
    test_labels = metrics["test_labels"]
    test_ids = metrics.get("test_ids")

    with rasterio.open(out_path) as src:
        train_shape = metrics.get("train_shape")
        if train_shape and (src.height, src.width) != tuple(train_shape):
            return None, "Prediction grid shape differs from training grid."

        train_transform = metrics.get("train_transform")
        if train_transform is not None:
            if not np.allclose(tuple(src.transform), tuple(train_transform)):
                return None, "Prediction transform differs from training transform."

        train_crs = metrics.get("train_crs")
        if train_crs is not None and src.crs is not None and train_crs != src.crs:
            return None, "Prediction CRS differs from training CRS."

        in_bounds = (
            (test_rows_arr >= 0)
            & (test_rows_arr < src.height)
            & (test_cols_arr >= 0)
            & (test_cols_arr < src.width)
        )
        if not np.any(in_bounds):
            return None, "No holdout samples fall within prediction bounds."

        if not np.all(in_bounds):
            dropped_bounds = int(np.count_nonzero(~in_bounds))
        else:
            dropped_bounds = 0

        test_rows_arr = test_rows_arr[in_bounds]
        test_cols_arr = test_cols_arr[in_bounds]
        test_labels = test_labels[in_bounds]
        if test_ids is not None:
            test_ids = test_ids[in_bounds]

        coords = [
            rasterio.transform.xy(src.transform, int(row), int(col), offset="center")
            for row, col in zip(test_rows_arr, test_cols_arr)
        ]
        sampled = np.array([val[0] for val in src.sample(coords)])

        nodata = src.nodata
        if nodata is None:
            valid = np.ones(sampled.shape[0], dtype=bool)
        elif np.isnan(nodata):
            valid = ~np.isnan(sampled)
        else:
            valid = sampled != nodata

        if not np.any(valid):
            return None, "All holdout samples landed on nodata in predictions."

        dropped_nodata = int(np.count_nonzero(~valid))
        if dropped_nodata:
            test_labels = test_labels[valid]
            sampled = sampled[valid]
            if test_ids is not None:
                test_ids = test_ids[valid]

        classes = metrics["classes"]
        matrix = confusion_matrix(test_labels, sampled, labels=classes)
        accuracy = accuracy_score(test_labels, sampled)

        note = None
        total_dropped = dropped_bounds + dropped_nodata
        if total_dropped:
            note = (
                "Dropped "
                f"{total_dropped} holdout samples outside raster or on nodata."
            )

        return (
            {
                "sample_count": len(test_labels),
                "accuracy": float(accuracy),
                "matrix": matrix,
                "predictions": sampled,
                "test_labels": test_labels,
                "test_ids": test_ids,
                "note": note,
            },
            None,
        )


def _write_holdout_outputs(
    metrics: Dict[str, Any],
    out_path: Path,
    *,
    smoothed: Optional[Dict[str, Any]] = None,
    smoothed_note: Optional[str] = None,
    smooth: str = "none",
) -> None:
    metrics_txt = _metrics_path(out_path)
    labels = metrics["labels"]
    matrix = metrics["matrix"]

    lines = [
        f"Holdout samples: {metrics['sample_count']}",
        f"Raw accuracy: {metrics['accuracy']:.3f}",
        (
            f"Training grid sampling: {metrics['train_grid_size']} px per cell."
            if metrics.get("train_grid_size")
            else "Training grid sampling: none."
        ),
        "",
        "Raw confusion matrix (rows=true, cols=pred):",
        _format_confusion_matrix(matrix, labels),
    ]

    raw_ids = _misclassified_ids(
        metrics["test_labels"], metrics["predictions"], metrics.get("test_ids")
    )
    if raw_ids is None:
        lines.append("")
        lines.append("Misclassified validation IDs (raw): unavailable.")
    else:
        lines.append("")
        lines.append(
            "Misclassified validation IDs (raw): "
            + (", ".join(raw_ids) if raw_ids else "none")
        )

    lines.append("")
    lines.append(f"Smoothing mode: {smooth}")
    if smoothed is None:
        lines.append("Smoothed accuracy: unavailable.")
        if smoothed_note:
            lines.append(f"Smoothed metrics note: {smoothed_note}")
        lines.append("")
        lines.append("Smoothed confusion matrix: unavailable.")
    else:
        lines.append(f"Smoothed accuracy: {smoothed['accuracy']:.3f}")
        lines.append("")
        lines.append("Smoothed confusion matrix (rows=true, cols=pred):")
        lines.append(_format_confusion_matrix(smoothed["matrix"], labels))
        if smoothed.get("note"):
            lines.append(smoothed["note"])

        smoothed_ids = _misclassified_ids(
            smoothed["test_labels"],
            smoothed["predictions"],
            smoothed.get("test_ids"),
        )
        lines.append("")
        if smoothed_ids is None:
            lines.append("Misclassified validation IDs (smoothed): unavailable.")
        else:
            lines.append(
                "Misclassified validation IDs (smoothed): "
                + (", ".join(smoothed_ids) if smoothed_ids else "none")
            )

    bands = metrics["band_importances"]
    if bands is None:
        lines.append("")
        lines.append("Band importance unavailable (model has no feature_importances_).")
    else:
        lines.append("")
        lines.append("Band importance (sorted):")
        for band_idx, importance in bands:
            lines.append(f"band {band_idx}: {importance:.6f}")

    class_stats = metrics.get("class_band_stats")
    if class_stats is None:
        lines.append("")
        lines.append("Per-class band stats (train): unavailable.")
    else:
        stats_classes = class_stats.get("classes")
        counts = class_stats.get("counts")
        means = class_stats.get("mean")
        stds = class_stats.get("std")
        band_indices = class_stats.get("band_indices")
        if (
            stats_classes is None
            or counts is None
            or means is None
            or stds is None
            or band_indices is None
        ):
            lines.append("")
            lines.append("Per-class band stats (train): unavailable.")
        else:
            label_map = {cls: name for cls, name in zip(stats_classes, labels)}
            lines.append("")
            lines.append("Per-class band stats (train):")
            for idx, class_value in enumerate(stats_classes):
                label = label_map.get(class_value, str(class_value))
                lines.append(f"class {label} (n={int(counts[idx])}):")
                for band_idx, mean_val, std_val in zip(
                    band_indices, means[idx], stds[idx]
                ):
                    lines.append(
                        f"band {int(band_idx)}: mean={mean_val:.6f} "
                        f"std={std_val:.6f}"
                    )

    metrics_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(f"[green]Wrote holdout metrics to {metrics_txt}.")


def train_rf(
    image_path: Union[PathLike, Iterable[PathLike]],
    shapefile_path: PathLike,
    label_column: str,
    model_out: PathLike,
    *,
    band_indices: Optional[Sequence[int]] = None,
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    n_jobs: int = -1,
    random_state: int = 42,
    test_fraction: float = 0.3,
    grid_size: Optional[int] = None,
) -> RandomForestClassifier:
    """Train a Random Forest classifier on raster pixels under training polygons.

    The `image_path` can be a single multi-band raster, a directory of GeoTIFFs
    (expanded), or an iterable of coregistered rasters (elevation, NDVI,
    spectral bands, etc.). Use `band_indices` (1-based) to select a subset of
    stacked bands for training. A configurable fraction of samples is held out
    for evaluation and persisted with the model.
    """

    _log("[bold cyan]Loading training data...")
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

        label_cat = gdf[label_column].astype("category")
        code_column = "__plaknit_label_code__"
        gdf[code_column] = label_cat.cat.codes + 1

        categories = list(label_cat.cat.categories)
        decoder = {idx + 1: value for idx, value in enumerate(categories)}

        X, y, sample_ids, sample_rows, sample_cols = _collect_training_samples(
            stack, gdf, code_column
        )
        y = y.astype("int32", copy=False)
        train_shape = (stack.height, stack.width)
        train_transform = stack.transform
        train_crs = stack.crs

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

    _log(
        f"[bold cyan]Training RandomForest on {X_train.shape[0]:,} samples "
        f"({X_train.shape[1]} bands)..."
    )
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=random_state,
        oob_score=False,
    )
    rf.fit(X_train, y_train)
    rf.label_decoder = decoder  # type: ignore[attr-defined]
    if band_indices is not None:
        rf.band_indices = list(band_indices)  # type: ignore[attr-defined]
    rf.test_samples_ = X_test  # type: ignore[attr-defined]
    rf.test_labels_ = y_test  # type: ignore[attr-defined]
    rf.test_fraction_ = test_fraction  # type: ignore[attr-defined]
    rf.train_shape_ = train_shape  # type: ignore[attr-defined]
    rf.train_transform_ = train_transform  # type: ignore[attr-defined]
    rf.train_crs_ = train_crs  # type: ignore[attr-defined]
    rf.train_grid_size_ = grid_size  # type: ignore[attr-defined]
    if extra_test is not None:
        rf.test_ids_ = extra_test[0]  # type: ignore[attr-defined]
        rf.test_rows_ = extra_test[1]  # type: ignore[attr-defined]
        rf.test_cols_ = extra_test[2]  # type: ignore[attr-defined]
    else:
        rf.test_ids_ = None  # type: ignore[attr-defined]
        rf.test_rows_ = None  # type: ignore[attr-defined]
        rf.test_cols_ = None  # type: ignore[attr-defined]
    if decoder:
        mapping_preview = ", ".join(
            f"{code}:{label}" for code, label in list(decoder.items())[:10]
        )
        _log(
            f"[green]Label codes => classes: {mapping_preview}"
            + (" ..." if len(decoder) > 10 else "")
        )

    classes = getattr(rf, "classes_", np.unique(y_train))
    class_stats = _compute_class_band_stats(X_train, y_train, classes=classes)
    band_ids = (
        list(band_indices)
        if band_indices is not None
        else list(range(1, X_train.shape[1] + 1))
    )
    class_stats["band_indices"] = np.asarray(band_ids, dtype=np.int64)
    rf.class_band_stats_ = class_stats  # type: ignore[attr-defined]

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
        train_pred = rf.predict(X_train)
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
            val_pred = rf.predict(X_test)
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
        all_pred = rf.predict(X)
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

    joblib.dump(rf, model_out)
    _log(f"[green]Model saved to {model_out}")
    return rf


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
    model: RandomForestClassifier,
    win: windows.Window,
    *,
    out_dtype: str,
    nodata_value: Union[int, float],
    smooth: str,
    beta: float,
    neighborhood: int,
    icm_iters: int,
    bayes_window_size: int,
    bayes_neigh_fraction: float,
    bayes_smoothness: Union[float, Sequence[float]],
    return_probs: bool,
    block_overlap: int,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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
        need_probs = return_probs or smooth != "none"
        if need_probs:
            probs = model.predict_proba(samples[valid]).astype("float32", copy=False)
            num_classes = probs.shape[1]
            prob_cube = np.full(
                (samples.shape[0], num_classes), np.nan, dtype="float32"
            )
            prob_cube[valid] = probs
            prob_cube = prob_cube.reshape(
                int(read_window.height), int(read_window.width), num_classes
            )
            if return_probs:
                prob_out = prob_cube[write_slice[0], write_slice[1]].transpose(2, 0, 1)

        if smooth == "none":
            if need_probs:
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
        elif smooth == "mrf":
            log_probs = np.full(
                (samples.shape[0], num_classes), -np.inf, dtype="float32"
            )
            log_probs[valid] = np.log(probs + 1e-9).astype("float32")
            log_probs = log_probs.reshape(
                int(read_window.height), int(read_window.width), num_classes
            )
            init_labels = np.full(
                (int(read_window.height), int(read_window.width)),
                -1,
                dtype=np.int32,
            )
            init_labels[block_valid] = log_probs.argmax(axis=2)[block_valid]

            smoothed = _icm_smooth(
                log_probs,
                init_labels,
                block_valid,
                beta=beta,
                neighborhood=neighborhood,
                iters=icm_iters,
            )
            if class_values is not None:
                smoothed = class_values[smoothed]
            smoothed[~block_valid] = nodata_value
            smoothed = smoothed[write_slice[0], write_slice[1]].astype(
                out_dtype, copy=False
            )
            predictions = smoothed
        else:
            smoothed_probs = _bayes_smooth(
                prob_cube,
                block_valid,
                window_size=bayes_window_size,
                neigh_fraction=bayes_neigh_fraction,
                smoothness=bayes_smoothness,
            )
            masked_probs = np.where(block_valid[:, :, None], smoothed_probs, -np.inf)
            best_idx = masked_probs.argmax(axis=2)
            if class_values is not None:
                best = class_values[best_idx]
            else:
                best = best_idx
            best[~block_valid] = nodata_value
            best = best[write_slice[0], write_slice[1]].astype(out_dtype, copy=False)
            predictions = best
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
_PREDICT_MODEL: Optional[RandomForestClassifier] = None
_PREDICT_OUT_DTYPE: Optional[str] = None
_PREDICT_NODATA: Optional[Union[int, float]] = None
_PREDICT_SMOOTH: str = "none"
_PREDICT_BETA: float = 1.0
_PREDICT_NEIGHBORHOOD: int = 4
_PREDICT_ICM_ITERS: int = 3
_PREDICT_BAYES_WINDOW_SIZE: int = 7
_PREDICT_BAYES_NEIGH_FRACTION: float = 0.5
_PREDICT_BAYES_SMOOTHNESS: Union[float, Sequence[float]] = 20.0
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
    smooth: str,
    beta: float,
    neighborhood: int,
    icm_iters: int,
    bayes_window_size: int,
    bayes_neigh_fraction: float,
    bayes_smoothness: Union[float, Sequence[float]],
    return_probs: bool,
) -> None:
    global _PREDICT_STACK
    global _PREDICT_MODEL
    global _PREDICT_OUT_DTYPE
    global _PREDICT_NODATA
    global _PREDICT_SMOOTH
    global _PREDICT_BETA
    global _PREDICT_NEIGHBORHOOD
    global _PREDICT_ICM_ITERS
    global _PREDICT_BAYES_WINDOW_SIZE
    global _PREDICT_BAYES_NEIGH_FRACTION
    global _PREDICT_BAYES_SMOOTHNESS
    global _PREDICT_RETURN_PROBS

    stack = _open_raster_stack(image_paths, band_indices=band_indices)
    stack.__enter__()
    _PREDICT_STACK = stack
    _PREDICT_MODEL = joblib.load(model_path)
    _PREDICT_OUT_DTYPE = out_dtype
    _PREDICT_NODATA = nodata_value
    _PREDICT_SMOOTH = smooth
    _PREDICT_BETA = beta
    _PREDICT_NEIGHBORHOOD = neighborhood
    _PREDICT_ICM_ITERS = icm_iters
    _PREDICT_BAYES_WINDOW_SIZE = bayes_window_size
    _PREDICT_BAYES_NEIGH_FRACTION = bayes_neigh_fraction
    _PREDICT_BAYES_SMOOTHNESS = bayes_smoothness
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
        smooth=_PREDICT_SMOOTH,
        beta=_PREDICT_BETA,
        neighborhood=_PREDICT_NEIGHBORHOOD,
        icm_iters=_PREDICT_ICM_ITERS,
        bayes_window_size=_PREDICT_BAYES_WINDOW_SIZE,
        bayes_neigh_fraction=_PREDICT_BAYES_NEIGH_FRACTION,
        bayes_smoothness=_PREDICT_BAYES_SMOOTHNESS,
        return_probs=_PREDICT_RETURN_PROBS,
        block_overlap=block_overlap,
    )


def predict_rf(
    image_path: Union[PathLike, Iterable[PathLike]],
    model_path: PathLike,
    output_path: PathLike,
    *,
    band_indices: Optional[Sequence[int]] = None,
    block_shape: Optional[Tuple[int, int]] = None,
    smooth: str = "none",
    beta: float = 1.0,
    neighborhood: int = 4,
    icm_iters: int = 3,
    bayes_window_size: int = 7,
    bayes_neigh_fraction: float = 0.5,
    bayes_smoothness: Union[float, Sequence[float]] = 20.0,
    probs_out: Optional[PathLike] = None,
    block_overlap: int = 0,
    jobs: int = 1,
) -> Path:
    """Apply a trained Random Forest to a raster stack and write a classified GeoTIFF.

    The `image_path` can be a single raster, a directory of GeoTIFFs, or an
    iterable of aligned rasters. Optional Potts-MRF smoothing (`smooth="mrf"`)
    uses RF posteriors + ICM to reduce speckle. Bayesian smoothing
    (`smooth="bayes"`) applies empirical Bayes shrinkage to class probabilities
    using non-isotropic neighborhoods. `jobs` controls block-level
    parallelism via worker processes. Use `band_indices` (1-based) to select a
    subset of stacked bands for prediction. If `probs_out` is set, prediction
    writes a multi-band GeoTIFF of class probabilities (raw RF posteriors).
    If the model includes holdout samples, prediction logs a confusion matrix
    and band importance.
    """

    _log("[bold cyan]Loading model...")
    model: RandomForestClassifier = joblib.load(model_path)
    classes = getattr(model, "classes_", None)
    classes_dtype = getattr(classes, "dtype", np.int32)
    if np.issubdtype(classes_dtype, np.integer):
        out_dtype = "int16"
        nodata_value: Union[int, float] = -1
    else:
        out_dtype = "float32"
        nodata_value = np.nan

    smooth = smooth.lower()
    if smooth not in {"none", "mrf", "bayes"}:
        raise ValueError("smooth must be 'none', 'mrf', or 'bayes'.")

    if smooth == "bayes":
        if bayes_window_size < 3 or bayes_window_size % 2 == 0:
            raise ValueError("bayes_window_size must be an odd integer >= 3.")
        if not (0 < bayes_neigh_fraction <= 1):
            raise ValueError("bayes_neigh_fraction must be in (0, 1].")

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
                model_jobs = getattr(model, "n_jobs", None)
                if model_jobs not in (None, 1):
                    _log(
                        "[yellow]Block-level parallelism requested but model n_jobs="
                        f"{model_jobs}. This may oversubscribe CPU; consider setting "
                        "model.n_jobs=1 before prediction."
                    )

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
                        smooth,
                        beta,
                        neighborhood,
                        icm_iters,
                        bayes_window_size,
                        bayes_neigh_fraction,
                        bayes_smoothness,
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
                        smooth=smooth,
                        beta=beta,
                        neighborhood=neighborhood,
                        icm_iters=icm_iters,
                        bayes_window_size=bayes_window_size,
                        bayes_neigh_fraction=bayes_neigh_fraction,
                        bayes_smoothness=bayes_smoothness,
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
        if smooth == "none":
            smoothed_metrics = {
                "sample_count": metrics["sample_count"],
                "accuracy": metrics["accuracy"],
                "matrix": metrics["matrix"],
                "predictions": metrics["predictions"],
                "test_labels": metrics["test_labels"],
                "test_ids": metrics.get("test_ids"),
                "note": "Smoothing disabled; metrics match raw predictions.",
            }
            smoothed_note = None
        else:
            smoothed_metrics, smoothed_note = _collect_smoothed_holdout_metrics(
                metrics, out_path
            )
        _write_holdout_outputs(
            metrics,
            out_path,
            smoothed=smoothed_metrics,
            smoothed_note=smoothed_note,
            smooth=smooth,
        )

    _log(f"[green]Classification saved to {out_path}")
    return out_path


def _smooth_prob_block(
    dataset: rasterio.io.DatasetReader,
    win: windows.Window,
    *,
    out_dtype: str,
    nodata_value: Union[int, float],
    smooth: str,
    beta: float,
    neighborhood: int,
    icm_iters: int,
    bayes_window_size: int,
    bayes_neigh_fraction: float,
    bayes_smoothness: Union[float, Sequence[float]],
    block_overlap: int,
    class_values: Optional[np.ndarray],
) -> np.ndarray:
    read_window, write_slice = _window_with_overlap(
        win,
        block_overlap=block_overlap,
        raster_width=dataset.width,
        raster_height=dataset.height,
    )
    data = dataset.read(window=read_window, out_dtype="float32")
    if data.size == 0:
        return np.full((int(win.height), int(win.width)), nodata_value, dtype=out_dtype)

    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    prob_cube = data.transpose(1, 2, 0)
    h, w, num_classes = prob_cube.shape
    samples = prob_cube.reshape(-1, num_classes)

    nodata_mask = _nodata_pixel_mask(samples, dataset.nodatavals)
    nodata_mask |= ~np.all(np.isfinite(samples), axis=1)
    valid = ~nodata_mask

    if not np.any(valid):
        return np.full((int(win.height), int(win.width)), nodata_value, dtype=out_dtype)

    block_valid = valid.reshape(h, w)
    if smooth == "none":
        masked_probs = np.where(block_valid[:, :, None], prob_cube, -np.inf)
        best_idx = masked_probs.argmax(axis=2)
        if class_values is not None:
            best = class_values[best_idx]
        else:
            best = best_idx
    elif smooth == "mrf":
        log_probs = np.full((samples.shape[0], num_classes), -np.inf, dtype="float32")
        log_probs[valid] = np.log(np.clip(samples[valid], 1e-9, 1.0))
        log_probs = log_probs.reshape(h, w, num_classes)
        init_labels = np.full((h, w), -1, dtype=np.int32)
        init_labels[block_valid] = log_probs.argmax(axis=2)[block_valid]
        smoothed = _icm_smooth(
            log_probs,
            init_labels,
            block_valid,
            beta=beta,
            neighborhood=neighborhood,
            iters=icm_iters,
        )
        if class_values is not None:
            best = class_values[smoothed]
        else:
            best = smoothed
    else:
        smoothed_probs = _bayes_smooth(
            prob_cube,
            block_valid,
            window_size=bayes_window_size,
            neigh_fraction=bayes_neigh_fraction,
            smoothness=bayes_smoothness,
        )
        masked_probs = np.where(block_valid[:, :, None], smoothed_probs, -np.inf)
        best_idx = masked_probs.argmax(axis=2)
        if class_values is not None:
            best = class_values[best_idx]
        else:
            best = best_idx

    best[~block_valid] = nodata_value
    best = best[write_slice[0], write_slice[1]].astype(out_dtype, copy=False)
    return best


_SMOOTH_DATASET: Optional[rasterio.io.DatasetReader] = None
_SMOOTH_OUT_DTYPE: Optional[str] = None
_SMOOTH_NODATA: Optional[Union[int, float]] = None
_SMOOTH_SMOOTH: str = "none"
_SMOOTH_BETA: float = 1.0
_SMOOTH_NEIGHBORHOOD: int = 4
_SMOOTH_ICM_ITERS: int = 3
_SMOOTH_BAYES_WINDOW_SIZE: int = 7
_SMOOTH_BAYES_NEIGH_FRACTION: float = 0.5
_SMOOTH_BAYES_SMOOTHNESS: Union[float, Sequence[float]] = 20.0
_SMOOTH_CLASS_VALUES: Optional[np.ndarray] = None


def _close_smooth_worker() -> None:
    global _SMOOTH_DATASET
    if _SMOOTH_DATASET is not None:
        _SMOOTH_DATASET.close()
        _SMOOTH_DATASET = None


def _init_smooth_worker(
    prob_path: str,
    out_dtype: str,
    nodata_value: Union[int, float],
    smooth: str,
    beta: float,
    neighborhood: int,
    icm_iters: int,
    bayes_window_size: int,
    bayes_neigh_fraction: float,
    bayes_smoothness: Union[float, Sequence[float]],
    class_values: Optional[Sequence[Union[int, float]]],
) -> None:
    global _SMOOTH_DATASET
    global _SMOOTH_OUT_DTYPE
    global _SMOOTH_NODATA
    global _SMOOTH_SMOOTH
    global _SMOOTH_BETA
    global _SMOOTH_NEIGHBORHOOD
    global _SMOOTH_ICM_ITERS
    global _SMOOTH_BAYES_WINDOW_SIZE
    global _SMOOTH_BAYES_NEIGH_FRACTION
    global _SMOOTH_BAYES_SMOOTHNESS
    global _SMOOTH_CLASS_VALUES

    _SMOOTH_DATASET = rasterio.open(prob_path)
    _SMOOTH_OUT_DTYPE = out_dtype
    _SMOOTH_NODATA = nodata_value
    _SMOOTH_SMOOTH = smooth
    _SMOOTH_BETA = beta
    _SMOOTH_NEIGHBORHOOD = neighborhood
    _SMOOTH_ICM_ITERS = icm_iters
    _SMOOTH_BAYES_WINDOW_SIZE = bayes_window_size
    _SMOOTH_BAYES_NEIGH_FRACTION = bayes_neigh_fraction
    _SMOOTH_BAYES_SMOOTHNESS = bayes_smoothness
    _SMOOTH_CLASS_VALUES = (
        np.asarray(class_values) if class_values is not None else None
    )
    atexit.register(_close_smooth_worker)


def _smooth_block_worker(win_tuple: WindowTuple, block_overlap: int) -> np.ndarray:
    if _SMOOTH_DATASET is None:
        raise RuntimeError("Smoothing worker not initialized.")
    if _SMOOTH_OUT_DTYPE is None or _SMOOTH_NODATA is None:
        raise RuntimeError("Smoothing worker missing output settings.")

    win = _tuple_to_window(win_tuple)
    return _smooth_prob_block(
        _SMOOTH_DATASET,
        win,
        out_dtype=_SMOOTH_OUT_DTYPE,
        nodata_value=_SMOOTH_NODATA,
        smooth=_SMOOTH_SMOOTH,
        beta=_SMOOTH_BETA,
        neighborhood=_SMOOTH_NEIGHBORHOOD,
        icm_iters=_SMOOTH_ICM_ITERS,
        bayes_window_size=_SMOOTH_BAYES_WINDOW_SIZE,
        bayes_neigh_fraction=_SMOOTH_BAYES_NEIGH_FRACTION,
        bayes_smoothness=_SMOOTH_BAYES_SMOOTHNESS,
        block_overlap=block_overlap,
        class_values=_SMOOTH_CLASS_VALUES,
    )


def smooth_probs(
    probability_path: PathLike,
    output_path: PathLike,
    *,
    model_path: Optional[PathLike] = None,
    class_values: Optional[Sequence[Union[int, float]]] = None,
    block_shape: Optional[Tuple[int, int]] = None,
    smooth: str = "mrf",
    beta: float = 1.0,
    neighborhood: int = 4,
    icm_iters: int = 3,
    bayes_window_size: int = 7,
    bayes_neigh_fraction: float = 0.5,
    bayes_smoothness: Union[float, Sequence[float]] = 20.0,
    block_overlap: int = 0,
    jobs: int = 1,
) -> Path:
    """Smooth a class-probability raster and write a classified GeoTIFF."""

    smooth = smooth.lower()
    if smooth not in {"none", "mrf", "bayes"}:
        raise ValueError("smooth must be 'none', 'mrf', or 'bayes'.")

    if model_path is not None and class_values is not None:
        raise ValueError("Provide either model_path or class_values, not both.")

    if smooth == "bayes":
        if bayes_window_size < 3 or bayes_window_size % 2 == 0:
            raise ValueError("bayes_window_size must be an odd integer >= 3.")
        if not (0 < bayes_neigh_fraction <= 1):
            raise ValueError("bayes_neigh_fraction must be in (0, 1].")

    if block_overlap < 0:
        raise ValueError("block_overlap must be non-negative.")

    if jobs is None:
        jobs = 1
    if jobs <= 0:
        jobs = max(1, os.cpu_count() or 1)

    out_path = Path(output_path)
    if out_path.suffix.lower() == ".vrt":
        out_path = out_path.with_suffix(".tif")
        _log("[yellow]Output path ended with .vrt; writing GeoTIFF to .tif instead.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_classes: Optional[np.ndarray] = None
    if model_path is not None:
        model: RandomForestClassifier = joblib.load(model_path)
        model_classes = np.asarray(getattr(model, "classes_", None))

    with rasterio.open(probability_path) as src:
        num_classes = src.count
        if num_classes < 1:
            raise ValueError("Probability raster must have at least one band.")

        if class_values is not None:
            class_values_arr = np.asarray(class_values)
        elif model_classes is not None:
            class_values_arr = model_classes
        else:
            class_values_arr = np.arange(num_classes, dtype="int16")

        if class_values_arr.size != num_classes:
            raise ValueError(
                "Number of class values must match probability raster bands."
            )

        if np.issubdtype(class_values_arr.dtype, np.floating):
            rounded = np.round(class_values_arr)
            if np.allclose(class_values_arr, rounded):
                class_values_arr = rounded.astype("int64")

        if np.issubdtype(class_values_arr.dtype, np.integer):
            out_dtype = "int16"
            nodata_value: Union[int, float] = -1
        else:
            out_dtype = "float32"
            nodata_value = np.nan

        profile = _prepare_output_profile(src.profile, out_dtype, nodata_value)
        profile["driver"] = "GTiff"

        if block_shape:
            block_h, block_w = block_shape

            def custom_windows() -> Iterable[windows.Window]:
                for row_off in range(0, src.height, block_h):
                    for col_off in range(0, src.width, block_w):
                        yield windows.Window(
                            col_off=col_off,
                            row_off=row_off,
                            width=min(block_w, src.width - col_off),
                            height=min(block_h, src.height - row_off),
                        )

            window_iter: Iterable[windows.Window] = custom_windows()
        else:
            window_iter = (win for _, win in src.block_windows(1))

        with rasterio.open(out_path, "w", **profile) as dst:
            _log("[bold cyan]Smoothing probabilities...")
            if jobs > 1:
                max_workers = jobs
                max_pending = max_workers * 2
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=_init_smooth_worker,
                    initargs=(
                        str(probability_path),
                        out_dtype,
                        nodata_value,
                        smooth,
                        beta,
                        neighborhood,
                        icm_iters,
                        bayes_window_size,
                        bayes_neigh_fraction,
                        bayes_smoothness,
                        class_values_arr.tolist(),
                    ),
                ) as executor:
                    futures: Dict[concurrent.futures.Future, windows.Window] = {}
                    for win in window_iter:
                        future = executor.submit(
                            _smooth_block_worker,
                            _window_to_tuple(win),
                            block_overlap,
                        )
                        futures[future] = win
                        if len(futures) >= max_pending:
                            done, _ = concurrent.futures.wait(
                                futures, return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            for finished in done:
                                smoothed = finished.result()
                                dst.write(smoothed, 1, window=futures[finished])
                                del futures[finished]

                    for finished in concurrent.futures.as_completed(futures):
                        smoothed = finished.result()
                        dst.write(smoothed, 1, window=futures[finished])
            else:
                for win in window_iter:
                    smoothed = _smooth_prob_block(
                        src,
                        win,
                        out_dtype=out_dtype,
                        nodata_value=nodata_value,
                        smooth=smooth,
                        beta=beta,
                        neighborhood=neighborhood,
                        icm_iters=icm_iters,
                        bayes_window_size=bayes_window_size,
                        bayes_neigh_fraction=bayes_neigh_fraction,
                        bayes_smoothness=bayes_smoothness,
                        block_overlap=block_overlap,
                        class_values=class_values_arr,
                    )
                    dst.write(smoothed, 1, window=win)

    _log(f"[green]Smoothed classification saved to {out_path}")
    return out_path
