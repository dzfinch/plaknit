"""Spectral index helpers built around rasterio."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import rasterio
from rasterio.io import DatasetReader

ArrayLike = Union[np.ndarray, "np.typing.ArrayLike"]  # type: ignore[attr-defined]
PathLike = Union[str, Path]


def normalized_difference(
    numerator: ArrayLike,
    denominator: ArrayLike,
    *,
    nodata_value: Optional[float] = None,
) -> np.ndarray:
    """Compute the normalized difference of two arrays."""

    num = np.asarray(numerator, dtype="float32")
    den = np.asarray(denominator, dtype="float32")
    if num.shape != den.shape:
        raise ValueError(
            f"numerator and denominator arrays must share the same shape, "
            f"got {num.shape} and {den.shape}"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        result = (num - den) / (num + den)

    if nodata_value is not None:
        mask = (num == nodata_value) | (den == nodata_value)
        result = np.where(mask, np.nan, result)

    return result


def _write_raster(
    sample_src: DatasetReader,
    array: np.ndarray,
    dst_path: PathLike,
    dtype: str = "float32",
) -> None:
    profile = sample_src.profile.copy()
    profile.update(count=1, dtype=dtype)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)


def normalized_difference_from_raster(
    dataset_path: PathLike,
    numerator_band: int,
    denominator_band: int,
    *,
    dst_path: Optional[PathLike] = None,
    dtype: str = "float32",
) -> np.ndarray:
    """Compute normalized difference from two bands inside the same raster."""

    with rasterio.open(dataset_path) as src:
        numerator = src.read(numerator_band, out_dtype="float32")
        denominator = src.read(denominator_band, out_dtype="float32")
        index = normalized_difference(numerator, denominator, nodata_value=src.nodata)

        if dst_path is not None:
            _write_raster(src, np.nan_to_num(index), dst_path, dtype=dtype)

    return index


def normalized_difference_from_files(
    numerator_path: PathLike,
    denominator_path: PathLike,
    *,
    numerator_band: int = 1,
    denominator_band: int = 1,
    dst_path: Optional[PathLike] = None,
    dtype: str = "float32",
) -> np.ndarray:
    """Compute normalized difference pulling bands from two different files."""

    with rasterio.open(numerator_path) as num_src, rasterio.open(
        denominator_path
    ) as den_src:
        if (
            num_src.width != den_src.width
            or num_src.height != den_src.height
            or num_src.transform != den_src.transform
        ):
            raise ValueError(
                "Input rasters must share the same shape and affine transform."
            )

        numerator = num_src.read(numerator_band, out_dtype="float32")
        denominator = den_src.read(denominator_band, out_dtype="float32")

        index = normalized_difference(
            numerator,
            denominator,
            nodata_value=num_src.nodata if num_src.nodata == den_src.nodata else None,
        )

        if dst_path is not None:
            _write_raster(num_src, np.nan_to_num(index), dst_path, dtype=dtype)

    return index
