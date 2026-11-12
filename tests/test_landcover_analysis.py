import csv
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from plaknit.cli import compare_cli
from plaknit.processing import landcover_compare as lc
from plaknit.processing.landcover_analysis import (
    InputSpec,
    _transition_metrics,
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
    assert not (output / "compare_agreement_overview.csv").exists()
    assert not (output / "compare_agreement_squares.csv").exists()
    assert not (output / "compare_agreement_squares.gpkg").exists()
    assert not (output / "compare_landscape_metrics.csv").exists()
    assert not (output / "compare_agreement_patterns.csv").exists()
    assert not (output / "compare_disagreement_patches.gpkg").exists()
    metadata = json.loads((output / "compare_metadata.json").read_text())
    assert metadata["mode"] == mode
    assert metadata["inputs"][0]["label"] == "a"
    assert metadata["analysis_settings"]["metric_engine_version"]
    assert "not independently verified" in metadata["class_meaning_assumption"]
    assert (
        "PyLandStats"
        in metadata["metric_definitions"]["square_shannon_diversity_index"]
    )
    assert "base 2" in metadata["metric_definitions"]["square_joint_entropy"]
    assert "square_complexity" not in metadata["metric_definitions"]
    if mode == "spatial":
        assert all("date" not in item for item in metadata["inputs"])
    assert "Selected" in capsys.readouterr().out


def test_temporal_validation_requires_distinct_dates():
    inputs = [
        InputSpec("a", "a.tif", "2024-01-01"),
        InputSpec("b", "b.tif", "2024-01-01"),
    ]
    with pytest.raises(ValueError, match="unique acquisition date"):
        _validate_specs("temporal", inputs, "date")
    with pytest.raises(ValueError, match="requires a date"):
        _validate_specs(
            "temporal", [inputs[0], InputSpec("b", "b.tif", None)], "date"
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
    with open(out / "compare_squares.csv", newline="") as stream:
        squares = list(csv.DictReader(stream))
    assert all(float(row["full_agreement_fraction"]) == 1.0 for row in squares)
    assert all(float(row["disagreement_fraction"]) == 0.0 for row in squares)


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
    metadata = json.loads((out / "compare_metadata.json").read_text())
    transition = metadata["square_class_transitions"]
    assert transition["reference_input_label"] == "coarse"
    assert transition["reference_resolution_m"] == [5.0, 5.0]
    assert transition["comparison_inputs"][0]["resolution_m"] == [2.0, 2.0]


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


@pytest.mark.parametrize("mode", ["spatial", "temporal"])
def test_squares_agreement_columns_match_geopackage(tmp_path, mode):
    import geopandas as gpd

    inputs = _inputs(tmp_path)
    if mode == "temporal":
        inputs[1].date = "2025-06-01"
    out = tmp_path / "result"
    result = run_analysis(mode, inputs, out, width_m=20, n_samples=3, seed=4)
    with open(out / "compare_squares.csv", newline="") as stream:
        rows = list(csv.DictReader(stream))
    geodata = gpd.read_file(out / "compare_squares.gpkg")
    assert len(rows) == len(geodata) == result.selection.found
    fine = lc.load_layers([item.classification for item in inputs])[0][0]
    with rasterio.open(out / "compare_agreement.tif") as src:
        agreement = src.read(1)
        for row, (_, geo) in zip(rows, geodata.iterrows()):
            for name in ("full_agreement_fraction", "disagreement_fraction"):
                assert float(row[name]) == pytest.approx(geo[name])
            r0, r1, c0, c1 = lc.centroid_window(
                fine,
                float(row["xmin"]),
                float(row["ymin"]),
                float(row["xmax"]),
                float(row["ymax"]),
            )
            data = agreement[r0 : r1 + 1, c0 : c1 + 1]
            valid = data[data > 0]
            assert float(row["full_agreement_fraction"]) == pytest.approx(
                np.count_nonzero(valid == 1) / valid.size
            )
            assert float(row["disagreement_fraction"]) == pytest.approx(
                1 - np.count_nonzero(valid == 1) / valid.size
            )
    assert all("agreement_overview" not in p.name for p in result.outputs)
    assert all("agreement_squares" not in p.name for p in result.outputs)
    with open(out / "compare_squares_metrics.csv", newline="") as stream:
        transitions = list(csv.DictReader(stream))
    assert len(transitions) == len(rows) * 2
    assert {row["reference_class_code"] for row in transitions} == {"1", "2"}
    assert {row["reference_input_label"] for row in transitions} == {"a"}
    assert "input_2_b_class_1_fraction" in transitions[0]


def _transition_fixture(tmp_path, arrays, resolutions, centers, width):
    paths = [
        _write(tmp_path / f"layer_{i}.tif", np.array(array, dtype="uint8"), res=res)
        for i, (array, res) in enumerate(zip(arrays, resolutions))
    ]
    layers, finest = lc.load_layers(
        paths, labels=[f"layer_{i}" for i in range(len(paths))]
    )
    tif = tmp_path / "agreement.tif"
    lc.write_agreement(layers, finest, tif)
    selection = lc.SelectionResult(centers, len(centers), 0, 1, 1)
    comparison = lc.CompareResult(
        selection,
        tmp_path / "x.gpkg",
        tmp_path / "x.csv",
        tif,
        tmp_path / "legend.csv",
        layers,
        finest,
    )
    return _transition_metrics(comparison, width)


def test_transition_values_and_five_class_rows(tmp_path):
    fine = [
        [1, 2, 1, 1],
        [1, 2, 4, 4],
        [2, 2, 3, 3],
        [2, 4, 3, 3],
    ]
    coarse = [[1, 1], [2, 3]]
    other = [
        [1, 1, 5, 5],
        [1, 1, 5, 5],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ]
    rows, fields, metadata = _transition_fixture(
        tmp_path, [fine, coarse, other], [1, 2, 1], [(2, 2)], 4
    )
    assert len(rows) == 5
    assert metadata["reference_input_order"] == 2
    assert metadata["class_codes"] == [1, 2, 3, 4, 5]
    assert len(fields) == 5 + 2 * 5
    assert all(row["reference_input_label"] == "layer_1" for row in rows)
    by_class = {row["reference_class_code"]: row for row in rows}
    one = by_class[1]
    assert one["reference_class_cell_count"] == 8
    assert one["reference_class_fraction"] == 0.5
    assert one["input_1_layer_0_class_1_fraction"] == 0.5
    assert one["input_1_layer_0_class_2_fraction"] == 0.25
    assert one["input_1_layer_0_class_4_fraction"] == 0.25
    assert one["input_3_layer_2_class_1_fraction"] == 0.5
    assert one["input_3_layer_2_class_5_fraction"] == 0.5
    assert by_class[2]["input_1_layer_0_class_2_fraction"] == 0.75
    assert by_class[2]["input_1_layer_0_class_4_fraction"] == 0.25
    assert by_class[3]["input_1_layer_0_class_3_fraction"] == 1.0
    for row in rows:
        for index in (1, 3):
            fractions = [
                row[f"input_{index}_layer_{index - 1}_class_{code}_fraction"]
                for code in range(1, 6)
            ]
            if row["reference_class_cell_count"]:
                assert sum(fractions) == pytest.approx(1.0)
            else:
                assert row["reference_class_fraction"] == 0
                assert all(value is None for value in fractions)


def test_five_classes_emit_five_csv_rows_per_square(tmp_path):
    a = np.tile(np.array([[1, 2, 3, 4, 5]], dtype="uint8"), (10, 2))
    b = np.roll(a, 1, axis=1)
    paths = [_write(tmp_path / "a.tif", a), _write(tmp_path / "b.tif", b)]
    out = tmp_path / "five-classes"
    result = run_analysis(
        "spatial",
        [InputSpec("a", paths[0]), InputSpec("b", paths[1])],
        out,
        width_m=20,
        n_samples=2,
    )
    with open(out / "compare_squares_metrics.csv", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5 * result.selection.found
    for square_id in range(1, result.selection.found + 1):
        assert {
            int(row["reference_class_code"])
            for row in rows
            if int(row["square_id"]) == square_id
        } == {1, 2, 3, 4, 5}
    metadata = json.loads((out / "compare_metadata.json").read_text())
    columns = metadata["square_class_transitions"]["transition_columns"]
    assert [(item["input_label"], item["class_code"]) for item in columns] == [
        ("b", code) for code in range(1, 6)
    ]
    assert [item["column"] for item in columns] == [
        f"input_2_b_class_{code}_fraction" for code in range(1, 6)
    ]


def test_equal_resolution_reference_tie_and_nodata(tmp_path):
    first = [[1, 1], [2, 2]]
    second = [[1, 2], [2, 255]]
    third = [[2, 1], [2, 1]]
    rows, _, metadata = _transition_fixture(
        tmp_path, [first, second, third], [1, 1, 1], [(1, 1)], 2
    )
    assert metadata["reference_input_order"] == 1
    assert [item["input_order"] for item in metadata["comparison_inputs"]] == [2, 3]
    by_class = {row["reference_class_code"]: row for row in rows}
    assert by_class[1]["reference_class_cell_count"] == 2
    assert by_class[1]["reference_class_fraction"] == pytest.approx(2 / 3)
    assert by_class[1]["input_2_layer_1_class_1_fraction"] == 0.5
    assert by_class[2]["reference_class_cell_count"] == 1
    assert by_class[2]["input_2_layer_1_class_2_fraction"] == 1.0


def test_square_metrics_labels_are_preserved_and_columns_are_distinct(tmp_path):
    arrays = np.tile(np.array([[1, 2]], dtype="uint8"), (12, 12))
    labels = ["reference, sensor", "sensor-a", "sensor_a", "sensor,b"]
    inputs = [
        InputSpec(label, _write(tmp_path / f"sensor_{i}.tif", arrays))
        for i, label in enumerate(labels)
    ]
    out = tmp_path / "labels"
    result = run_analysis("spatial", inputs, out, width_m=20, n_samples=1)
    assert result.selection.found == 1
    with open(out / "compare_squares_metrics.csv", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    assert len(fields) == len(set(fields))
    assert {row["reference_input_label"] for row in rows} == {"reference, sensor"}
    assert "input_2_sensor-a_class_1_fraction" in fields
    assert "input_3_sensor_a_class_1_fraction" in fields
    assert "input_4_sensor,b_class_1_fraction" in fields
    metadata = json.loads((out / "compare_metadata.json").read_text())
    columns = metadata["square_class_transitions"]["transition_columns"]
    for column in columns:
        assert column["column"] in fields
        assert labels[column["input_order"] - 1] == column["input_label"]


def test_transition_mapping_uses_largest_overlap_on_shifted_grid(tmp_path):
    fine = _write(
        tmp_path / "fine.tif",
        np.tile(np.array([[1, 1, 2, 2]], dtype="uint8"), (4, 1)),
        res=1,
    )
    coarse_path = tmp_path / "shifted.tif"
    with rasterio.open(
        coarse_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="uint8",
        crs=CRS,
        transform=from_origin(0.6, 4, 2, 2),
        nodata=255,
    ) as dst:
        dst.write(np.tile(np.array([[1, 2]], dtype="uint8"), (2, 1)), 1)
    layers, finest = lc.load_layers([fine, str(coarse_path)])
    agreement = tmp_path / "agreement.tif"
    lc.write_agreement(layers, finest, agreement)
    selection = lc.SelectionResult([(2, 2)], 1, 1, 1, 1)
    comparison = lc.CompareResult(
        selection,
        tmp_path / "x.gpkg",
        tmp_path / "x.csv",
        agreement,
        tmp_path / "legend.csv",
        layers,
        finest,
    )
    rows, _, _ = _transition_metrics(comparison, 4)
    by_class = {row["reference_class_code"]: row for row in rows}
    assert by_class[1]["reference_class_cell_count"] == 8
    assert by_class[1]["reference_class_fraction"] == pytest.approx(2 / 3)
    assert by_class[1]["input_1_r1_fine_class_1_fraction"] == 0.5
    assert by_class[1]["input_1_r1_fine_class_2_fraction"] == 0.5
    assert by_class[2]["reference_class_cell_count"] == 4
    assert by_class[2]["input_1_r1_fine_class_2_fraction"] == 1.0


def test_transitions_use_the_agreement_producers_finest_index(tmp_path):
    reference = _write(
        tmp_path / "reference.tif",
        np.tile(np.array([[1, 1, 2, 2]], dtype="uint8"), (4, 1)),
        res=1,
    )
    fine = _write(tmp_path / "fine.tif", np.ones((4, 4), dtype="uint8"), res=1)
    with rasterio.open(fine, "r+") as dataset:
        dataset.transform = from_origin(0.25, 4, 1, 1)
    layers, _ = lc.load_layers([reference, fine])
    agreement = tmp_path / "agreement.tif"
    lc.write_agreement(layers, 1, agreement)
    selection = lc.SelectionResult([(1.75, 2)], 1, 1, 1, 1)
    comparison = lc.CompareResult(
        selection,
        tmp_path / "x.gpkg",
        tmp_path / "x.csv",
        agreement,
        tmp_path / "legend.csv",
        layers,
        1,
    )
    rows, _, metadata = _transition_metrics(comparison, 2)
    by_class = {row["reference_class_code"]: row for row in rows}
    assert metadata["reference_input_order"] == 1
    assert by_class[1]["reference_class_cell_count"] == 4
    assert by_class[1]["reference_class_fraction"] == 1.0
    assert by_class[2]["reference_class_cell_count"] == 0


def test_empty_selection_writes_stable_headers(tmp_path):
    import geopandas as gpd

    inputs = _inputs(tmp_path)
    out = tmp_path / "empty"
    result = run_analysis("spatial", inputs, out, width_m=1000, n_samples=1)
    assert result.selection.found == 0
    with open(out / "compare_squares.csv", newline="") as stream:
        reader = csv.DictReader(stream)
        assert "full_agreement_fraction" in reader.fieldnames
        assert "a_shannon_diversity_index" in reader.fieldnames
        assert "a_joint_entropy" in reader.fieldnames
        assert "a_entropy" not in reader.fieldnames
        assert "a_complexity" not in reader.fieldnames
        assert list(reader) == []
    assert (
        "full_agreement_fraction" in gpd.read_file(out / "compare_squares.gpkg").columns
    )
    with open(out / "compare_squares_metrics.csv", newline="") as stream:
        reader = csv.DictReader(stream)
        assert "input_2_b_class_1_fraction" in reader.fieldnames
        assert "reference_input_label" in reader.fieldnames
        assert list(reader) == []


def test_existing_removed_outputs_are_not_deleted(tmp_path):
    inputs = _inputs(tmp_path)
    out = tmp_path / "existing"
    out.mkdir()
    obsolete = [
        out / "compare_agreement_overview.csv",
        out / "compare_agreement_squares.csv",
        out / "compare_agreement_squares.gpkg",
    ]
    for path in obsolete:
        path.write_text("existing output", encoding="utf-8")
    result = run_analysis("spatial", inputs, out, width_m=20, n_samples=1)
    for path in obsolete:
        assert path.read_text(encoding="utf-8") == "existing output"
        assert path not in result.outputs


def test_empty_agreement_raises(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path)
    original = lc.write_agreement

    def no_valid_counts(*args, **kwargs):
        original(*args, **kwargs)
        return {}

    monkeypatch.setattr(lc, "write_agreement", no_valid_counts)
    with pytest.raises(ValueError, match="agreement raster has no valid common cells"):
        run_analysis("spatial", inputs, tmp_path / "empty-agreement", width_m=20)


@pytest.mark.parametrize("mode", ["spatial", "temporal"])
def test_cli_rejects_removed_patch_option(tmp_path, mode):
    with pytest.raises(SystemExit) as exc:
        compare_cli.main(
            [
                mode,
                "--input",
                "a=a.tif",
                "--input",
                "b=b.tif",
                "--out-dir",
                str(tmp_path),
                "--patch-min-area-m2",
                "1",
            ]
        )
    assert exc.value.code == 2
