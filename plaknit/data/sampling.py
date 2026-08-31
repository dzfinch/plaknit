"""Sampling, splitting, and label utilities shared across model implementations."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features, windows
from rasterio.features import geometry_window
from sklearn.model_selection import train_test_split

from .raster import _RasterStack, _log, _nodata_pixel_mask


def _collect_training_samples(
    stack: _RasterStack,
    gdf: gpd.GeoDataFrame,
    label_column: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-pixel samples under each training geometry."""

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
