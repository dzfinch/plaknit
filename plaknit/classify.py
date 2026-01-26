"""Random Forest training and inference utilities for raster stacks."""

from __future__ import annotations

import atexit
import concurrent.futures
import csv
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract per-pixel samples under each training geometry."""

    feature_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []

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

    if not feature_chunks:
        raise ValueError("No training samples were extracted. Check label geometries.")

    features_arr = np.vstack(feature_chunks)
    labels_arr = np.concatenate(label_chunks)
    return features_arr, labels_arr


def _split_train_test(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    test_fraction: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if test_fraction <= 0:
        return features, labels, None, None
    if test_fraction >= 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    unique_labels = np.unique(labels)
    stratify = labels if unique_labels.size > 1 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=test_fraction,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        _log("[yellow]Stratified split failed; falling back to unstratified holdout.")
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=test_fraction,
            random_state=random_state,
            stratify=None,
        )
    return X_train, y_train, X_test, y_test


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
        "matrix": matrix,
        "band_importances": bands,
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


def _metrics_paths(out_path: Path) -> Tuple[Path, Path]:
    base = out_path.with_suffix("")
    metrics_csv = base.with_name(f"{base.name}_metrics.csv")
    metrics_txt = base.with_name(f"{base.name}_metrics.txt")
    return metrics_csv, metrics_txt


def _write_holdout_outputs(metrics: Dict[str, Any], out_path: Path) -> None:
    metrics_csv, metrics_txt = _metrics_paths(out_path)
    labels = metrics["labels"]
    matrix = metrics["matrix"]

    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [int(val) for val in row])

    lines = [
        f"Holdout samples: {metrics['sample_count']}",
        f"Accuracy: {metrics['accuracy']:.3f}",
        "",
        "Confusion matrix (rows=true, cols=pred):",
        _format_confusion_matrix(matrix, labels),
    ]

    bands = metrics["band_importances"]
    if bands is None:
        lines.append("")
        lines.append("Band importance unavailable (model has no feature_importances_).")
    else:
        lines.append("")
        lines.append("Band importance (sorted):")
        for band_idx, importance in bands:
            lines.append(f"band {band_idx}: {importance:.6f}")

    metrics_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(f"[green]Wrote holdout metrics to {metrics_csv} and {metrics_txt}.")


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

        X, y = _collect_training_samples(stack, gdf, code_column)
        y = y.astype("int32", copy=False)

    X_train, y_train, X_test, y_test = _split_train_test(
        X, y, test_fraction=test_fraction, random_state=random_state
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
    if decoder:
        mapping_preview = ", ".join(
            f"{code}:{label}" for code, label in list(decoder.items())[:10]
        )
        _log(
            f"[green]Label codes => classes: {mapping_preview}"
            + (" ..." if len(decoder) > 10 else "")
        )
    _log("[green]Training complete. Saving model...")

    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, model_out)
    _log(f"[green]Model saved to {model_out}")
    return rf


def _prepare_output_profile(
    profile: dict, dtype: str, nodata_value: Union[int, float]
) -> dict:
    profile = profile.copy()
    profile.update(count=1, dtype=dtype, nodata=nodata_value)
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
    block_overlap: int,
) -> np.ndarray:
    read_window, write_slice = _window_with_overlap(
        win,
        block_overlap=block_overlap,
        raster_width=stack.width,
        raster_height=stack.height,
    )
    block = stack.read(window=read_window, out_dtype="float32")
    if block.size == 0:
        return np.full((int(win.height), int(win.width)), nodata_value, dtype=out_dtype)

    samples = block.reshape(stack.count, -1).T
    valid = ~_nodata_pixel_mask(samples, stack.nodata_values)

    predictions = np.full(samples.shape[0], nodata_value, dtype=out_dtype)
    if np.any(valid):
        if smooth == "none":
            preds = model.predict(samples[valid])
            predictions[valid] = preds.astype(out_dtype, copy=False)
            predictions = predictions.reshape(
                int(read_window.height), int(read_window.width)
            )
            predictions = predictions[write_slice[0], write_slice[1]]
        else:
            probs = model.predict_proba(samples[valid])
            num_classes = probs.shape[1]
            log_probs = np.full(
                (samples.shape[0], num_classes), -np.inf, dtype="float32"
            )
            log_probs[valid] = np.log(probs + 1e-9).astype("float32")

            log_probs = log_probs.reshape(
                int(read_window.height), int(read_window.width), num_classes
            )
            block_valid = valid.reshape(int(read_window.height), int(read_window.width))
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
            smoothed[~block_valid] = nodata_value
            smoothed = smoothed[write_slice[0], write_slice[1]].astype(
                out_dtype, copy=False
            )
            predictions = smoothed
    else:
        predictions = predictions.reshape(
            int(read_window.height), int(read_window.width)
        )
        predictions = predictions[write_slice[0], write_slice[1]]

    return predictions


_PREDICT_STACK: Optional[_RasterStack] = None
_PREDICT_MODEL: Optional[RandomForestClassifier] = None
_PREDICT_OUT_DTYPE: Optional[str] = None
_PREDICT_NODATA: Optional[Union[int, float]] = None
_PREDICT_SMOOTH: str = "none"
_PREDICT_BETA: float = 1.0
_PREDICT_NEIGHBORHOOD: int = 4
_PREDICT_ICM_ITERS: int = 3


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
) -> None:
    global _PREDICT_STACK
    global _PREDICT_MODEL
    global _PREDICT_OUT_DTYPE
    global _PREDICT_NODATA
    global _PREDICT_SMOOTH
    global _PREDICT_BETA
    global _PREDICT_NEIGHBORHOOD
    global _PREDICT_ICM_ITERS

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
    atexit.register(_close_predict_worker)


def _predict_block_worker(win_tuple: WindowTuple, block_overlap: int) -> np.ndarray:
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
    block_overlap: int = 0,
    jobs: int = 1,
) -> Path:
    """Apply a trained Random Forest to a raster stack and write a classified GeoTIFF.

    The `image_path` can be a single raster, a directory of GeoTIFFs, or an
    iterable of aligned rasters. Optional Potts-MRF smoothing (`smooth="mrf"`)
    uses RF posteriors + ICM to reduce speckle. `jobs` controls block-level
    parallelism via worker processes. Use `band_indices` (1-based) to select a
    subset of stacked bands for prediction. If the model includes holdout
    samples, prediction logs a confusion matrix and band importance.
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
    if smooth not in {"none", "mrf"}:
        raise ValueError("smooth must be 'none' or 'mrf'.")

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
        if metrics is not None:
            _write_holdout_outputs(metrics, out_path)

        profile = _prepare_output_profile(stack.profile, out_dtype, nodata_value)
        # Ensure we write a GeoTIFF even when reading from a VRT source.
        profile["driver"] = "GTiff"
        with rasterio.open(out_path, "w", **profile) as dst:
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
                                predictions = finished.result()
                                dst.write(predictions, 1, window=futures[finished])
                                del futures[finished]

                    for finished in concurrent.futures.as_completed(futures):
                        predictions = finished.result()
                        dst.write(predictions, 1, window=futures[finished])
            else:
                for win in window_iter:
                    predictions = _predict_block(
                        stack,
                        model,
                        win,
                        out_dtype=out_dtype,
                        nodata_value=nodata_value,
                        smooth=smooth,
                        beta=beta,
                        neighborhood=neighborhood,
                        icm_iters=icm_iters,
                        block_overlap=block_overlap,
                    )
                    dst.write(predictions, 1, window=win)

    _log(f"[green]Classification saved to {out_path}")
    return out_path
