"""Raster loading and alignment helpers shared across model implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

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


def _align_raster_to_grid(
    dataset: rasterio.io.DatasetReader,
    template: rasterio.io.DatasetReader,
) -> rasterio.io.DatasetReader:
    """Return *dataset* on *template*'s grid, warping only when necessary."""

    same_grid = (
        dataset.width == template.width
        and dataset.height == template.height
        and dataset.crs == template.crs
        and np.allclose(dataset.transform, template.transform)
    )
    if same_grid:
        return dataset

    if dataset.crs is None or template.crs is None:
        raise ValueError(
            "Rasters on different grids must all have a CRS so they can be aligned."
        )

    return WarpedVRT(
        dataset,
        crs=template.crs,
        transform=template.transform,
        width=template.width,
        height=template.height,
        resampling=Resampling.nearest,
    )


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


class _CastingDataset:
    """Proxy dataset that casts reads to a target dtype while delegating metadata.

    This wraps a rasterio dataset and ensures `read(..., out_dtype=...)` will
    succeed by requesting the chosen `target_dtype`. It exposes a subset of the
    DatasetReader attributes used by `_RasterStack`.
    """

    def __init__(self, ds: rasterio.io.DatasetReader, target_dtype: str):
        self._ds = ds
        self.count = ds.count
        self.width = ds.width
        self.height = ds.height
        self.crs = ds.crs
        self.transform = ds.transform
        # profile copy with coerced dtype
        self.profile = ds.profile.copy()
        self.profile["dtype"] = target_dtype
        # expose nodatavals as a tuple of scalars or None
        raw = ds.nodatavals or ()
        if raw:
            self.nodatavals = tuple(
                None if v is None else np.asarray(v).astype(target_dtype).item()
                for v in raw
            )
        else:
            self.nodatavals = tuple([None] * self.count)

    @property
    def dtypes(self):
        # return a list-like of per-band dtype strings
        try:
            return list(self._ds.dtypes)
        except Exception:
            return [self.profile.get("dtype")] * self.count

    def block_windows(self, bidx: int = 1):
        return self._ds.block_windows(bidx)

    def read(
        self,
        *args,
        window=None,
        out_dtype: Optional[str] = None,
        indexes=None,
        **kwargs,
    ):
        dtype = out_dtype or self.profile.get("dtype")
        band_ids = (
            list(indexes) if indexes is not None else list(range(1, self.count + 1))
        )
        # Read one band at a time: rasterio refuses a single multi-band read
        # when the underlying dataset's bands have differing native dtypes,
        # so we can't just delegate a combined read here.
        bands = [
            self._ds.read(indexes=[band_id], window=window, **kwargs)[0]
            for band_id in band_ids
        ]
        return np.stack(bands, axis=0).astype(dtype, copy=False)

    def close(self):
        try:
            self._ds.close()
        except Exception:
            pass


class _RasterStack:
    """Lightweight reader that stacks multiple rasters band-wise."""

    def __init__(self, paths: List[Path], band_indices: Optional[Sequence[int]] = None):
        self.paths = paths
        self.datasets: List[rasterio.io.DatasetReader] = []
        self._source_datasets: List[rasterio.io.DatasetReader] = []
        self.count = 0
        self.nodata_values: List[Optional[float]] = []
        self.template: Optional[rasterio.io.DatasetReader] = None
        self._requested_band_indices = (
            list(band_indices) if band_indices is not None else None
        )
        self._selected_band_map: Optional[List[Tuple[int, int]]] = None
        self._all_band_map: List[Tuple[int, int]] = []

    def __enter__(self) -> "_RasterStack":
        self._source_datasets = [rasterio.open(p) for p in self.paths]
        self.datasets = list(self._source_datasets)
        if not self.datasets:
            raise ValueError("No raster paths were provided.")

        self.template = self.datasets[0]
        for index, ds in enumerate(self.datasets[1:], start=1):
            self.datasets[index] = _align_raster_to_grid(ds, self.template)

        # Detect mixed source dtypes and coerce to a common safe dtype when
        # necessary. Coercion is done via a thin proxy that requests reads in
        # the chosen dtype so downstream code receives a consistent array dtype.
        all_dtypes: List[str] = []
        for ds in self.datasets:
            try:
                all_dtypes.extend(list(ds.dtypes))
            except Exception:
                # fallback to profile dtype if dtypes not available
                all_dtypes.extend([ds.profile.get("dtype")] * getattr(ds, "count", 1))

        unique_dtypes = set(dt for dt in all_dtypes if dt is not None)
        if len(unique_dtypes) > 1:
            target_dtype = "float32"
            _log(f"[yellow]Input rasters have mixed dtypes; casting to {target_dtype}.")
            self.datasets = [_CastingDataset(ds, target_dtype) for ds in self.datasets]

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
        for ds in reversed(self.datasets):
            ds.close()
        for ds in self._source_datasets:
            if not any(ds is stack_ds for stack_ds in self.datasets):
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
    """Normalize raster inputs to a list of Paths."""

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
