import csv
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from plaknit.cli import compare_cli
from plaknit.processing.landcover_analysis import (
    InputSpec,
    _validate_specs,
    run_analysis,
)

CRS = "EPSG:32618"


def _write(path, values, res=10, top=None, nodata=255):
    values = np.asarray(values)
    if top is None:
        top = values.shape[0] * res
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=values.dtype,
        crs=CRS,
        transform=from_origin(0, top, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(values, 1)
    return str(path)


def _inputs(tmp_path):
    classes_a = (np.indices((24, 24)).sum(axis=0) % 2 + 1).astype("uint8")
    classes_b = np.roll(classes_a, 1, axis=1)
    paths = [
        _write(tmp_path / "classified_a.tif", classes_a),
        _write(tmp_path / "classified_b.tif", classes_b),
    ]
    return [
        InputSpec(
            label=label,
            classification=paths[i],
            date="2024-06-01",
        )
        for i, label in enumerate(("a", "b"))
    ]


@pytest.mark.parametrize("mode", ["spatial", "temporal"])
def test_cli_subcommands_emit_landscape_and_agreement_outputs(
    tmp_path, capsys, mode
):
    inputs = _inputs(tmp_path)
    if mode == "temporal":
        inputs[1].date = "2025-06-01"
    args = [mode]
    for item in inputs:
        args.extend(["--input", f"{item.label}={item.classification}"])
        if mode == "temporal":
            args.extend(["--date", f"{item.label}={item.date}"])
    args.extend(
        [
            "--out-dir",
            str(tmp_path / mode),
            "--width",
            "20",
            "--samples",
            "4",
            "--seed",
            "17",
        ]
    )
    assert compare_cli.main(args) == 0
    output = tmp_path / mode
    assert (output / "compare_class_metrics.csv").exists()
    assert (output / "compare_squares_metrics.csv").exists()
    assert (output / "compare_agreement_overview.csv").exists()
    assert (output / "compare_agreement_squares.gpkg").exists()
    assert not (output / "compare_landscape_metrics.csv").exists()
    assert not (output / "compare_agreement_patterns.csv").exists()
    assert not (output / "compare_disagreement_patches.gpkg").exists()
    metadata = json.loads((output / "compare_metadata.json").read_text())
    assert metadata["mode"] == mode
    assert metadata["inputs"][0]["label"] == "a"
    assert "not independently verified" in metadata["class_meaning_assumption"]
    if mode == "spatial":
        assert all("date" not in item for item in metadata["inputs"])
    assert "Selected" in capsys.readouterr().out


def test_temporal_validation_requires_distinct_dates():
    inputs = [
        InputSpec("a", "a.tif", "2024-01-01"),
        InputSpec("b", "b.tif", "2024-01-01"),
    ]
    with pytest.raises(ValueError, match="unique acquisition date"):
        _validate_specs("temporal", inputs, "date", 0)
    with pytest.raises(ValueError, match="requires a date"):
        _validate_specs(
            "temporal", [inputs[0], InputSpec("b", "b.tif", None)], "date", 0
        )


def test_nodata_common_footprint_and_all_agree_patch_output(tmp_path):
    a = np.ones((10, 10), dtype="uint8")
    b = np.ones((10, 10), dtype="uint8")
    a[5, 5] = 3
    b[5, 5] = 255
    path_a = _write(tmp_path / "nodata_a.tif", a)
    path_b = _write(tmp_path / "nodata_b.tif", b)
    inputs = [
        InputSpec("a", path_a, "2024-01-01"),
        InputSpec("b", path_b, "2024-01-01"),
    ]
    out = tmp_path / "nodata-result"
    run_analysis(
        "spatial",
        inputs,
        out,
        width_m=20,
        n_samples=2,
        seed=4,
    )
    metadata = json.loads((out / "compare_metadata.json").read_text())
    assert metadata["common_classification_footprint"][
        "valid_native_area_m2"
    ] == {
        "a": 9900.0,
        "b": 9900.0,
    }
    with open(out / "compare_class_metrics.csv", newline="") as stream:
        metrics = list(csv.DictReader(stream))
    assert {(row["input_label"], row["class_code"]) for row in metrics} == {
        ("a", "1"),
        ("b", "1"),
    }
    with open(out / "compare_agreement_overview.csv", newline="") as stream:
        overview = list(csv.DictReader(stream))
    assert len(overview) == 1
    assert float(overview[0]["full_agreement_fraction"]) == 1.0
    assert int(overview[0]["distinct_pattern_count"]) == 1
    assert int(overview[0]["disagreement_patch_count"]) == 0


def test_whole_and_square_metrics_retain_native_resolutions(tmp_path):
    fine = np.tile(np.array([[1, 2], [2, 1]], dtype="uint8"), (20, 20))
    coarse = np.tile(np.array([[1, 2], [2, 1]], dtype="uint8"), (8, 8))
    path_fine = _write(tmp_path / "fine.tif", fine, res=2)
    path_coarse = _write(tmp_path / "coarse.tif", coarse, res=5)
    inputs = [
        InputSpec("fine", path_fine, "2024-01-01"),
        InputSpec("coarse", path_coarse, "2024-01-01"),
    ]
    out = tmp_path / "native-resolution"
    run_analysis(
        "spatial",
        inputs,
        out,
        width_m=10,
        n_samples=2,
        seed=4,
    )
    with open(out / "compare_class_metrics.csv", newline="") as stream:
        class_rows = list(csv.DictReader(stream))
    assert {row["input_label"] for row in class_rows} == {"fine", "coarse"}
    assert len(class_rows) == 4
    with open(out / "compare_squares_metrics.csv", newline="") as stream:
        metrics = list(csv.DictReader(stream))
    assert {row["resolution_x_m"] for row in metrics} == {"2.0", "5.0"}


def test_pylandstats_units_and_boundary_rule(tmp_path):
    checker = np.array([[1, 2], [2, 1]], dtype="uint8")
    paths = [
        _write(tmp_path / "checker_a.tif", checker),
        _write(tmp_path / "checker_b.tif", checker),
    ]
    inputs = [
        InputSpec("a", paths[0], "2024-01-01"),
        InputSpec("b", paths[1], "2024-01-01"),
    ]
    out = tmp_path / "known-metrics"
    run_analysis("spatial", inputs, out, width_m=10, n_samples=1, seed=2)
    with open(out / "compare_class_metrics.csv", newline="") as stream:
        rows = list(csv.DictReader(stream))
    class_rows = [row for row in rows if row["input_label"] == "a"]
    assert len(class_rows) == 2
    assert all(
        float(row["proportion_pct"]) == pytest.approx(50)
        for row in class_rows
    )
    assert all(
        float(row["patch_density"]) == pytest.approx(50) for row in class_rows
    )
    assert all(int(row["patch_count"]) == 2 for row in class_rows)


def test_cli_rejects_removed_sensor_flag(tmp_path):
    with pytest.raises(SystemExit) as exc:
        compare_cli.main(
            [
                "spatial",
                "--input",
                "a=a.tif",
                "--sensor",
                "a=one",
                "--input",
                "b=b.tif",
                "--out-dir",
                str(tmp_path / "invalid"),
            ]
        )
    assert exc.value.code == 2
