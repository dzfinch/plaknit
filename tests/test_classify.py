"""Tests for classification raster-stack helpers."""

import numpy as np
import rasterio
from rasterio.transform import from_origin

from plaknit.classify import _open_raster_stack


def _write_raster(path, data, transform):
    """Write a single-band test raster in Web Mercator."""

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype=data.dtype,
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def test_raster_stack_aligns_nonmatching_grid_to_first_raster(tmp_path):
    template_path = tmp_path / "template.tif"
    source_path = tmp_path / "source.tif"
    template_transform = from_origin(0, 2, 1, 1)

    _write_raster(
        template_path,
        np.array([[1, 2], [3, 4]], dtype="uint8"),
        template_transform,
    )
    _write_raster(
        source_path,
        np.array(
            [[10, 10, 20, 20], [10, 10, 20, 20], [30, 30, 40, 40], [30, 30, 40, 40]],
            dtype="uint8",
        ),
        from_origin(0, 2, 0.5, 0.5),
    )

    with _open_raster_stack([template_path, source_path]) as stack:
        assert stack.width == 2
        assert stack.height == 2
        assert stack.transform == template_transform
        np.testing.assert_array_equal(
            stack.read(window=rasterio.windows.Window(0, 0, 2, 2), out_dtype="uint8"),
            np.array([[[1, 2], [3, 4]], [[10, 20], [30, 40]]], dtype="uint8"),
        )
