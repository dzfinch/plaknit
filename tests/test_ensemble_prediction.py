import numpy as np
import pytest

from plaknit.models.ensemble import (
    _aggregate_ensemble_probabilities,
    _aggregate_ensemble_statistics,
    _assign_models_to_workers,
)


def test_assign_models_to_workers_round_robin() -> None:
    assert _assign_models_to_workers(5, 3) == [[0, 3], [1, 4], [2]]


def test_assign_models_to_workers_caps_workers_at_model_count() -> None:
    assert _assign_models_to_workers(2, 8) == [[0], [1]]


def test_assign_models_to_workers_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError):
        _assign_models_to_workers(0, 1)
    with pytest.raises(ValueError):
        _assign_models_to_workers(1, 0)


def test_aggregate_ensemble_probabilities_matches_sample_ci() -> None:
    member_probs = np.asarray(
        [
            [[[0.1, 0.9]]],
            [[[0.2, 0.8]]],
            [[[0.3, 0.7]]],
        ],
        dtype="float32",
    )
    mean, lower, upper = _aggregate_ensemble_probabilities(member_probs, 2.0)
    expected_mean = member_probs.mean(axis=0)
    expected_half_width = 2.0 * member_probs.std(axis=0, ddof=1) / np.sqrt(3)
    np.testing.assert_allclose(mean, expected_mean)
    np.testing.assert_allclose(lower, np.clip(expected_mean - expected_half_width, 0, 1))
    np.testing.assert_allclose(upper, np.clip(expected_mean + expected_half_width, 0, 1))


def test_aggregate_single_model_has_no_interval() -> None:
    member_probs = np.asarray([[[[0.4]]]], dtype="float32")
    mean, lower, upper = _aggregate_ensemble_probabilities(member_probs, None)
    np.testing.assert_array_equal(mean, member_probs[0])
    assert lower is None
    assert upper is None


def test_aggregate_statistics_matches_member_probabilities() -> None:
    member_probs = np.asarray(
        [
            [[[0.1, 0.9]]],
            [[[0.2, 0.8]]],
            [[[0.3, 0.7]]],
        ],
        dtype="float64",
    )
    sums = member_probs.sum(axis=0)
    sum_squares = np.square(member_probs).sum(axis=0)
    counts = np.full(sums.shape, member_probs.shape[0], dtype="int16")

    expected = _aggregate_ensemble_probabilities(member_probs.astype("float32"), 2.0)
    actual = _aggregate_ensemble_statistics(
        sums, sum_squares, counts, member_probs.shape[0], 2.0
    )
    for result, expected_result in zip(actual, expected):
        np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-6)


def test_aggregate_statistics_preserves_incomplete_pixels_as_nan() -> None:
    sums = np.asarray([[[0.3, 0.0]]], dtype="float64")
    sum_squares = np.asarray([[[0.05, 0.0]]], dtype="float64")
    counts = np.asarray([[[2, 1]]], dtype="int16")

    mean, lower, upper = _aggregate_ensemble_statistics(
        sums, sum_squares, counts, 2, 2.0
    )
    assert np.isfinite(mean[0, 0, 0])
    assert np.isnan(mean[0, 0, 1])
    assert np.isnan(lower[0, 0, 1])
    assert np.isnan(upper[0, 0, 1])