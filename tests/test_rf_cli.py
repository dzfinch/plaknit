"""Tests for the RF CLI routing."""

from __future__ import annotations

import pytest

from plaknit.cli import main as cli_main_fn


@pytest.mark.parametrize("sub", ["train", "classify", "smooth"])
def test_rf_subcommand_help(sub: str, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main_fn(["rf", sub, "--help"])
    assert exc.value.code == 0
    assert f"plaknit rf {sub}" in capsys.readouterr().out


def test_old_classify_command_removed() -> None:
    assert cli_main_fn(["classify", "train"]) == 2


def test_old_predict_subcommand_removed() -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main_fn(["rf", "predict"])
    assert exc.value.code == 2
