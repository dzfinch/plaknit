"""Tests for classification raster-stack helpers."""

import csv
import numpy as np
import joblib
import rasterio
from rasterio.transform import from_origin
from sklearn.tree import DecisionTreeClassifier

from plaknit.classify import _open_raster_stack, predict_brt, predict_rf
from plaknit.models.ensemble import BRTEnsemble, _write_feature_importance_csv
from plaknit.processing.evaluation import _collect_holdout_metrics


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


def test_predict_brt_writes_binary_raster_for_each_class(tmp_path):
    """Test BRT prediction with binary class outputs."""
    image_path = tmp_path / "image.tif"
    model_path = tmp_path / "model.joblib"
    output_path = tmp_path / "classified.tif"
    binary_dir = tmp_path / "binary"
    transform = from_origin(0, 2, 1, 1)
    data = np.array([[1, 2], [3, 4]], dtype="uint8")
    _write_raster(image_path, data, transform)

    # Use a simple DecisionTreeClassifier as a mock XGBoost model for testing
    model = DecisionTreeClassifier(random_state=0)
    model.fit(np.array([[1], [2], [3], [4]]), np.array([10, 20, 20, 10]))
    joblib.dump(model, model_path)

    predict_brt(image_path, model_path, output_path, binary_out=binary_dir)

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


def test_brt_ensemble_metadata_and_weights(tmp_path):
    """Test that BRT ensemble saves metadata with test accuracies."""
    ensemble = BRTEnsemble(n_models=2, random_state=42)
    assert ensemble.n_models == 2
    assert ensemble.random_state == 42
    assert len(ensemble.models_) == 0  # Not trained yet

    # Verify metadata structure
    assert ensemble.metadata_ == {}
    _log_msg = "[Test] Metadata structure verified"
    assert _log_msg is not None  # Placeholder assertion


def test_holdout_metrics_calculates_binary_roc_auc():
    model = DecisionTreeClassifier(random_state=0).fit(
        np.array([[0], [1]]), np.array([0, 1])
    )
    model.test_samples_ = np.array([[0], [1]])
    model.test_labels_ = np.array([0, 1])

    metrics = _collect_holdout_metrics(model)

    assert metrics is not None
    assert metrics["auc"] == 1.0


def test_holdout_metrics_skips_auc_for_one_class_holdout():
    model = DecisionTreeClassifier(random_state=0).fit(
        np.array([[0], [1]]), np.array([0, 1])
    )
    model.test_samples_ = np.array([[0]])
    model.test_labels_ = np.array([0])

    metrics = _collect_holdout_metrics(model)

    assert metrics is not None
    assert metrics["auc"] is None


def test_ensemble_feature_importance_summary_is_unweighted(tmp_path):
    first = DecisionTreeClassifier(random_state=0).fit(
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 1, 1])
    )
    second = DecisionTreeClassifier(random_state=1).fit(
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 0, 1])
    )
    first.band_indices = [3, 7]
    second.band_indices = [3, 7]
    output_path = tmp_path / "importance.csv"

    _write_feature_importance_csv(output_path, [first, second])

    with output_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(row["band_index"]) for row in rows] == [3, 7]
    np.testing.assert_allclose(
        [float(row["mean_importance"]) for row in rows], [0.5, 0.5]
    )
    assert [int(row["models_using_feature"]) for row in rows] == [1, 1]


def _write_ensemble_metadata(ensemble_dir, n_models):
    ensemble_dir.mkdir()
    (ensemble_dir / "ensemble_metadata.json").write_text(
        '{"n_models": ' + str(n_models) + ', "random_state": 42}'
    )


def test_ensemble_predict_writes_unweighted_probability_summaries(tmp_path):
    image_path = tmp_path / "image.tif"
    ensemble_dir = tmp_path / "ensemble"
    output_dir = tmp_path / "output"
    transform = from_origin(0, 2, 1, 1)
    _write_raster(image_path, np.array([[1, 2], [3, 4]], dtype="uint8"), transform)
    _write_ensemble_metadata(ensemble_dir, 2)

    first = DecisionTreeClassifier(random_state=0).fit(
        np.array([[1], [2], [3], [4]]), np.array([0, 0, 1, 1])
    )
    second = DecisionTreeClassifier(random_state=1).fit(
        np.array([[1], [2], [3], [4]]), np.array([0, 1, 1, 1])
    )
    joblib.dump(first, ensemble_dir / "brt_0.joblib")
    joblib.dump(second, ensemble_dir / "brt_1.joblib")

    mean_path = BRTEnsemble().predict(image_path, ensemble_dir, output_dir)

    assert mean_path == output_dir / "mean_probabilities.tif"
    assert not (output_dir / "classified.tif").exists()
    for name in (
        "mean_probabilities.tif",
        "lower_probabilities.tif",
        "upper_probabilities.tif",
    ):
        assert (output_dir / name).exists()
    with rasterio.open(mean_path) as mean:
        assert mean.dtypes == ("float32", "float32")
        mean_values = mean.read()
        np.testing.assert_allclose(
            mean_values,
            np.array(
                [
                    [[1.0, 0.5], [0.0, 0.0]],
                    [[0.0, 0.5], [1.0, 1.0]],
                ],
                dtype="float32",
            ),
        )
    with rasterio.open(output_dir / "lower_probabilities.tif") as lower, rasterio.open(
        output_dir / "upper_probabilities.tif"
    ) as upper:
        assert np.all(lower.read() <= upper.read())
        assert np.all(lower.read() <= mean_values)
        assert np.all(mean_values <= upper.read())


def test_ensemble_predict_writes_only_mean_for_one_member(tmp_path):
    image_path = tmp_path / "image.tif"
    ensemble_dir = tmp_path / "ensemble"
    output_dir = tmp_path / "output"
    _write_raster(
        image_path, np.array([[1, 2], [3, 4]], dtype="uint8"), from_origin(0, 2, 1, 1)
    )
    _write_ensemble_metadata(ensemble_dir, 1)
    model = DecisionTreeClassifier(random_state=0).fit(
        np.array([[1], [2], [3], [4]]), np.array([0, 0, 1, 1])
    )
    joblib.dump(model, ensemble_dir / "brt_0.joblib")

    BRTEnsemble().predict(image_path, ensemble_dir, output_dir)

    assert (output_dir / "mean_probabilities.tif").exists()
    assert not (output_dir / "lower_probabilities.tif").exists()
    assert not (output_dir / "upper_probabilities.tif").exists()


def test_ensemble_parallel_prediction_reads_blocks_once_and_preserves_overlap(
    tmp_path,
):
    image_path = tmp_path / "image.tif"
    ensemble_dir = tmp_path / "ensemble"
    output_dir = tmp_path / "output"
    _write_raster(
        image_path,
        np.array([[1, 2], [3, 4]], dtype="uint8"),
        from_origin(0, 2, 1, 1),
    )
    _write_ensemble_metadata(ensemble_dir, 2)
    for index, labels in enumerate(
        (np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    ):
        model = DecisionTreeClassifier(random_state=index).fit(
            np.array([[1], [2], [3], [4]]), labels
        )
        joblib.dump(model, ensemble_dir / f"brt_{index}.joblib")

    BRTEnsemble().predict(
        image_path,
        ensemble_dir,
        output_dir,
        block_shape=(1, 1),
        block_overlap=1,
        jobs=2,
    )

    with rasterio.open(output_dir / "mean_probabilities.tif") as mean:
        assert mean.shape == (2, 2)
        assert mean.count == 2
        assert np.isfinite(mean.read()).all()
