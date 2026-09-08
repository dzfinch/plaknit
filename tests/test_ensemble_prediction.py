import numpy as np
import pytest

from plaknit.models.ensemble import (
    _aggregate_ensemble_probabilities,
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