import numpy as np

from plaknit.data.raster import _expand_raster_inputs, _normalize_nodata
from plaknit.data.sampling import _split_train_test


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
