"""Random Forest training and inference utilities for raster stacks."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import geopandas as gpd
import joblib
import numpy as np
import rasterio
from rasterio import features, windows
from rasterio.features import geometry_window
from sklearn.ensemble import RandomForestClassifier

try:  # pragma: no cover - optional rich dependency
    from rich.console import Console
except ImportError:  # pragma: no cover - fallback logging
    console = None
else:  # pragma: no cover
    console = Console()

PathLike = Union[str, Path]


def _log(message: str) -> None:
    if console is not None:
        console.log(message)
    else:
        print(message)


def _nodata_pixel_mask(
    samples: np.ndarray, nodata_value: Optional[float]
) -> np.ndarray:
    """Return a boolean mask of pixels touching nodata."""

    if nodata_value is None:
        return np.zeros(samples.shape[0], dtype=bool)
    if np.isnan(nodata_value):
        return np.any(np.isnan(samples), axis=1)
    return np.any(samples == nodata_value, axis=1)


def _collect_training_samples(
    dataset: rasterio.io.DatasetReader,
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
            win = geometry_window(dataset, [geom], north_up=True, rotated=False)
        except ValueError:
            _log(f"[yellow]Skipping geometry {idx}: outside raster bounds.")
            continue

        if win.width == 0 or win.height == 0:
            continue

        data = dataset.read(window=win, out_dtype="float32")
        if data.size == 0:
            continue

        block_transform = windows.transform(win, dataset.transform)
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

        samples = data.reshape(dataset.count, -1).T
        valid &= ~_nodata_pixel_mask(samples, dataset.nodata)

        if not np.any(valid):
            continue

        feature_chunks.append(samples[valid])
        label_chunks.append(label_flat[valid])

    if not feature_chunks:
        raise ValueError("No training samples were extracted. Check label geometries.")

    features_arr = np.vstack(feature_chunks)
    labels_arr = np.concatenate(label_chunks)
    return features_arr, labels_arr


def train_rf(
    image_path: PathLike,
    shapefile_path: PathLike,
    label_column: str,
    model_out: PathLike,
    *,
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    n_jobs: int = -1,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest classifier on raster pixels under training polygons."""

    _log("[bold cyan]Loading training data...")
    with rasterio.open(image_path) as src:
        gdf = gpd.read_file(shapefile_path)
        if label_column not in gdf.columns:
            raise ValueError(f"Column '{label_column}' not found in training data.")

        if src.crs is None:
            raise ValueError("Raster must have a valid CRS.")
        if gdf.crs is None:
            warnings.warn(
                "Vector training data lacks CRS. Assuming raster CRS.", UserWarning
            )
            gdf.set_crs(src.crs, inplace=True)
        else:
            gdf = gdf.to_crs(src.crs)

        label_cat = gdf[label_column].astype("category")
        code_column = "__plaknit_label_code__"
        gdf[code_column] = label_cat.cat.codes + 1

        categories = list(label_cat.cat.categories)
        decoder = {idx + 1: value for idx, value in enumerate(categories)}

        X, y = _collect_training_samples(src, gdf, code_column)
        y = y.astype("int32", copy=False)

    _log(
        f"[bold cyan]Training RandomForest on {X.shape[0]:,} samples ({X.shape[1]} bands)..."
    )
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=random_state,
        oob_score=False,
    )
    rf.fit(X, y)
    rf.label_decoder = decoder  # type: ignore[attr-defined]
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
    src: rasterio.io.DatasetReader, dtype: str, nodata_value: Union[int, float]
) -> dict:
    profile = src.profile.copy()
    profile.update(count=1, dtype=dtype, nodata=nodata_value)
    return profile


def predict_rf(
    image_path: PathLike,
    model_path: PathLike,
    output_path: PathLike,
    *,
    block_shape: Optional[Tuple[int, int]] = None,
) -> Path:
    """Apply a trained Random Forest to a raster stack and write a classified GeoTIFF."""

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

    with rasterio.open(image_path) as src:
        profile = _prepare_output_profile(src, out_dtype, nodata_value)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_path, "w", **profile) as dst:
            _log("[bold cyan]Predicting classes...")
            if block_shape:
                block_h, block_w = block_shape

                def custom_windows() -> (
                    Iterable[Tuple[Tuple[int, int], windows.Window]]
                ):
                    for row_off in range(0, src.height, block_h):
                        for col_off in range(0, src.width, block_w):
                            yield (
                                (row_off // block_h, col_off // block_w),
                                windows.Window(
                                    col_off=col_off,
                                    row_off=row_off,
                                    width=min(block_w, src.width - col_off),
                                    height=min(block_h, src.height - row_off),
                                ),
                            )

                window_iter: Iterable[Tuple[Tuple[int, int], windows.Window]] = (
                    custom_windows()
                )
            else:
                window_iter = src.block_windows(1)

            for _, win in window_iter:
                block = src.read(window=win, out_dtype="float32")
                if block.size == 0:
                    continue

                samples = block.reshape(src.count, -1).T
                valid = ~_nodata_pixel_mask(samples, src.nodata)

                predictions = np.full(
                    samples.shape[0], nodata_value, dtype=profile["dtype"]
                )
                if np.any(valid):
                    preds = model.predict(samples[valid])
                    predictions[valid] = preds.astype(profile["dtype"], copy=False)

                predictions = predictions.reshape(int(win.height), int(win.width))
                dst.write(predictions, 1, window=win)

    _log(f"[green]Classification saved to {out_path}")
    return out_path
