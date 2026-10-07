import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from plaknit.data.raster import _expand_raster_inputs, _normalize_nodata
from plaknit.data.raster import _open_raster_stack
from plaknit.data.sampling import (
    _collect_training_samples,
    _collect_pseudo_absence_samples,
    _split_train_test,
)


def test_expand_raster_inputs_accepts_file_and_dir(tmp_path):
    raster = tmp_path / "a.tif"
    raster.write_bytes(b"fake")

    assert _expand_raster_inputs(raster) == [raster]


def test_normalize_nodata_expands_single_value():
    assert _normalize_nodata(0.0, 3) == [0.0, 0.0, 0.0]


def test_split_train_test_preserves_extra_arrays():
    features = np.arange(20, dtype=float).reshape(10, 2)
    labels = np.array([0, 1] * 5)
    extra = [np.arange(10), np.arange(10, 20)]

    X_train, y_train, X_test, y_test, extra_train, extra_test = _split_train_test(
        features,
        labels,
        test_fraction=0.4,
        random_state=7,
        extra=extra,
    )

    assert X_train.shape[0] + X_test.shape[0] == features.shape[0]
    assert len(extra_train) == len(extra)
    assert len(extra_test) == len(extra)
    assert all(arr.shape[0] == X_train.shape[0] for arr in extra_train)
    assert all(arr.shape[0] == X_test.shape[0] for arr in extra_test)


def test_collect_pseudo_absences_is_deterministic_and_buffered(tmp_path):
    raster_path = tmp_path / "image.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=6,
        height=6,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0, 6, 1, 1),
    ) as dst:
        dst.write(np.ones((1, 6, 6), dtype="float32"))

    labels = gpd.GeoDataFrame(
        {"presence": [1], "geometry": [Point(0.5, 5.5)]},
        crs="EPSG:3857",
    )
    with _open_raster_stack(raster_path) as stack:
        first = _collect_pseudo_absence_samples(
            stack,
            labels,
            presence_count=4,
            ratio=1.0,
            buffer_meters=1.1,
            random_state=7,
        )
        second = _collect_pseudo_absence_samples(
            stack,
            labels,
            presence_count=4,
            ratio=1.0,
            buffer_meters=1.1,
            random_state=7,
        )

    np.testing.assert_array_equal(first[3], second[3])
    np.testing.assert_array_equal(first[4], second[4])
    assert first[0].shape == (4, 1)
    assert np.all(first[1] == 0)
    assert not np.any((first[3] == 0) & (first[4] == 0))


def test_training_buffer_extracts_surrounding_pixels(tmp_path):
    raster_path = tmp_path / "image.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=5,
        height=5,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0, 5, 1, 1),
    ) as dst:
        dst.write(np.arange(25, dtype="float32").reshape(1, 5, 5))

    labels = gpd.GeoDataFrame(
        {"presence": [1], "geometry": [Point(2.5, 2.5)]},
        crs="EPSG:3857",
    )
    with _open_raster_stack(raster_path) as stack:
        unbuffered = _collect_training_samples(stack, labels, "presence")
        buffered = _collect_training_samples(
            stack, labels, "presence", buffer_meters=1.1
        )

    assert unbuffered[0].shape[0] == 1
    assert buffered[0].shape[0] > unbuffered[0].shape[0]
    assert np.all(buffered[1] == 1)
