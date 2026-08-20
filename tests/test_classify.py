"""Tests for classification raster-stack helpers."""

import numpy as np
import joblib
import rasterio
from rasterio.transform import from_origin
from sklearn.tree import DecisionTreeClassifier

from plaknit.classify import _open_raster_stack, predict_rf


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


def test_predict_writes_binary_raster_for_each_class(tmp_path):
    image_path = tmp_path / "image.tif"
    model_path = tmp_path / "model.joblib"
    output_path = tmp_path / "classified.tif"
    binary_dir = tmp_path / "binary"
    transform = from_origin(0, 2, 1, 1)
    data = np.array([[1, 2], [3, 4]], dtype="uint8")
    _write_raster(image_path, data, transform)

    model = DecisionTreeClassifier(random_state=0)
    model.fit(np.array([[1], [2], [3], [4]]), np.array([10, 20, 20, 10]))
    joblib.dump(model, model_path)

    predict_rf(image_path, model_path, output_path, binary_out=binary_dir)

    with rasterio.open(output_path) as classified:
        classified_data = classified.read(1)
    for class_value in (10, 20):
        binary_path = binary_dir / f"class_{class_value}.tif"
        with rasterio.open(binary_path) as binary:
            assert binary.dtypes == ("uint8",)
            assert binary.nodata == 0
            np.testing.assert_array_equal(
                binary.read(1), (classified_data == class_value).astype("uint8")
            )
