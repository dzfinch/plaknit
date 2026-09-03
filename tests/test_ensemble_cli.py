"""Tests for the BRT CLI."""

from __future__ import annotations

from plaknit.cli import ensemble_cli


def test_brt_train_help() -> None:
    """Test that brt train command shows help without errors."""
    try:
        ensemble_cli.main(["train", "--help"])
    except SystemExit as e:
        # --help should exit with code 0
        assert e.code == 0


def test_brt_predict_help() -> None:
    """Test that brt predict command shows help without errors."""
    try:
        ensemble_cli.main(["predict", "--help"])
    except SystemExit as e:
        # --help should exit with code 0
        assert e.code == 0


def test_brt_train_missing_required_args() -> None:
    """Test that brt train requires necessary arguments."""
    try:
        ensemble_cli.main(["train"])
    except SystemExit as e:
        # Missing required args should exit with code 2
        assert e.code == 2


def test_brt_predict_missing_required_args() -> None:
    """Test that brt predict requires necessary arguments."""
    try:
        ensemble_cli.main(["predict"])
    except SystemExit as e:
        # Missing required args should exit with code 2
        assert e.code == 2


def test_brt_train_forwards_jobs_to_ensemble(tmp_path, monkeypatch) -> None:
    captured = {}

    class DummyEnsemble:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ensemble_cli, "BRTEnsemble", DummyEnsemble)

    result = ensemble_cli.main(
        [
            "train",
            "--image",
            str(tmp_path / "image.tif"),
            "--labels",
            str(tmp_path / "labels.gpkg"),
            "--label-column",
            "presence",
            "--output",
            str(tmp_path / "ensemble"),
            "--n-models",
            "2",
            "--jobs",
            "3",
        ]
    )

    assert result == 0
    assert captured["n_models"] == 2
    assert captured["jobs"] == 3


def test_brt_predict_forwards_output_directory(tmp_path, monkeypatch) -> None:
    captured = {}

    class DummyEnsemble:
        def predict(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ensemble_cli, "BRTEnsemble", DummyEnsemble)

    result = ensemble_cli.main(
        [
            "predict",
            "--image",
            str(tmp_path / "image.tif"),
            "--ensemble-dir",
            str(tmp_path / "ensemble"),
            "--output-dir",
            str(tmp_path / "output"),
            "--feature-importance-out",
            str(tmp_path / "importance.csv"),
        ]
    )

    assert result == 0
    assert captured["output_dir"] == str(tmp_path / "output")
    assert captured["feature_importance_out"] == str(tmp_path / "importance.csv")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(0)
