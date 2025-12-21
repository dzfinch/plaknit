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
NeighborOffsets = Tuple[Tuple[int, int], ...]


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

    def __init__(self, paths: List[Path]):
        self.paths = paths
        self.datasets: List[rasterio.io.DatasetReader] = []
        self.count = 0
        self.nodata_values: List[Optional[float]] = []
        self.template: Optional[rasterio.io.DatasetReader] = None

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

        for ds in self.datasets:
            self.count += ds.count
            self.nodata_values.extend(_normalize_nodata(ds.nodatavals, ds.count))

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
        blocks: List[np.ndarray] = []
        for ds in self.datasets:
            blocks.append(ds.read(window=window, out_dtype=out_dtype))
        return np.concatenate(blocks, axis=0)


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


def _open_raster_stack(image_path: Union[PathLike, Iterable[PathLike]]) -> _RasterStack:
    return _RasterStack(_expand_raster_inputs(image_path))


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
    """Train a Random Forest classifier on raster pixels under training polygons.

    The `image_path` can be a single multi-band raster, a directory of GeoTIFFs
    (expanded), or an iterable of coregistered rasters (elevation, NDVI,
    spectral bands, etc.).
    """

    _log("[bold cyan]Loading training data...")
    with _open_raster_stack(image_path) as stack:
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
    profile: dict, dtype: str, nodata_value: Union[int, float]
) -> dict:
    profile = profile.copy()
    profile.update(count=1, dtype=dtype, nodata=nodata_value)
    return profile


def predict_rf(
    image_path: PathLike,
    model_path: PathLike,
    output_path: PathLike,
    *,
    block_shape: Optional[Tuple[int, int]] = None,
    smooth: str = "none",
    beta: float = 1.0,
    neighborhood: int = 4,
    icm_iters: int = 3,
    block_overlap: int = 0,
) -> Path:
    """Apply a trained Random Forest to a raster stack and write a classified GeoTIFF.

    The `image_path` can be a single raster, a directory of GeoTIFFs, or an
    iterable of aligned rasters. Optional Potts-MRF smoothing (`smooth="mrf"`)
    uses RF posteriors + ICM to reduce speckle.
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

    with _open_raster_stack(image_path) as stack:
        assert stack.template is not None
        profile = _prepare_output_profile(stack.profile, out_dtype, nodata_value)
        # Ensure we write a GeoTIFF even when reading from a VRT source.
        profile["driver"] = "GTiff"
        out_path = Path(output_path)
        if out_path.suffix.lower() == ".vrt":
            out_path = out_path.with_suffix(".tif")
            _log(
                "[yellow]Output path ended with .vrt; writing GeoTIFF to .tif instead."
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_path, "w", **profile) as dst:
            _log("[bold cyan]Predicting classes...")
            if block_shape:
                block_h, block_w = block_shape

                def custom_windows() -> (
                    Iterable[Tuple[Tuple[int, int], windows.Window]]
                ):
                    for row_off in range(0, stack.height, block_h):
                        for col_off in range(0, stack.width, block_w):
                            yield (
                                (row_off // block_h, col_off // block_w),
                                windows.Window(
                                    col_off=col_off,
                                    row_off=row_off,
                                    width=min(block_w, stack.width - col_off),
                                    height=min(block_h, stack.height - row_off),
                                ),
                            )

                window_iter: Iterable[Tuple[Tuple[int, int], windows.Window]] = (
                    custom_windows()
                )
            else:
                window_iter = stack.block_windows(1)

            if block_overlap < 0:
                raise ValueError("block_overlap must be non-negative.")

            for _, win in window_iter:
                if block_overlap > 0:
                    col_off = max(0, int(win.col_off) - block_overlap)
                    row_off = max(0, int(win.row_off) - block_overlap)
                    width = min(
                        int(win.width) + 2 * block_overlap, stack.width - col_off
                    )
                    height = min(
                        int(win.height) + 2 * block_overlap, stack.height - row_off
                    )
                    read_window = windows.Window(
                        col_off=col_off, row_off=row_off, width=width, height=height
                    )
                    write_slice = (
                        slice(
                            int(win.row_off - row_off),
                            int(win.row_off - row_off + win.height),
                        ),
                        slice(
                            int(win.col_off - col_off),
                            int(win.col_off - col_off + win.width),
                        ),
                    )
                else:
                    read_window = win
                    write_slice = (slice(None), slice(None))

                block = stack.read(window=read_window, out_dtype="float32")
                if block.size == 0:
                    continue

                samples = block.reshape(stack.count, -1).T
                valid = ~_nodata_pixel_mask(samples, stack.nodata_values)

                predictions = np.full(
                    samples.shape[0], nodata_value, dtype=profile["dtype"]
                )
                if np.any(valid):
                    if smooth == "none":
                        preds = model.predict(samples[valid])
                        predictions[valid] = preds.astype(profile["dtype"], copy=False)
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
                        block_valid = valid.reshape(
                            int(read_window.height), int(read_window.width)
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
                        smoothed[~block_valid] = nodata_value
                        smoothed = smoothed[write_slice[0], write_slice[1]].astype(
                            profile["dtype"], copy=False
                        )
                        predictions = smoothed
                else:
                    predictions = predictions.reshape(
                        int(read_window.height), int(read_window.width)
                    )
                    predictions = predictions[write_slice[0], write_slice[1]]

                dst.write(predictions, 1, window=win)

    _log(f"[green]Classification saved to {out_path}")
    return out_path
