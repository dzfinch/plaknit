import csv
import json

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import box

from plaknit.cli import compare_cli
from plaknit.processing.reflectance_analysis import (
    SpatialUnit,
    _benjamini_hochberg,
    _estimate_block_size,
    load_spatial_units,
    run_reflectance,
)

TEST_CRS = CRS.from_epsg(32618)


def _write(path, values, resolution=10, top=None, nodata=None, scale=None, offset=0):
    values = np.asarray(values)
    top = values.shape[0] * resolution if top is None else top
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=values.dtype,
        crs=TEST_CRS,
        transform=from_origin(0, top, resolution, resolution),
        nodata=nodata,
    ) as dst:
        dst.write(values, 1)
        if scale is not None:
            dst.scales = (scale,)
            dst.offsets = (offset,)
    return str(path)


def _units():
    units = []
    for row in range(2):
        ymin = row * 100
        for col in range(5):
            xmin = col * 100
            units.append(
                SpatialUnit(
                    f"class1-{row}-{col}",
                    box(xmin, ymin, xmin + 100, ymin + 100),
                    TEST_CRS,
                )
            )
        for col in range(5, 10):
            xmin = col * 100
            units.append(
                SpatialUnit(
                    f"class2-{row}-{col}",
                    box(xmin, ymin, xmin + 100, ymin + 100),
                    TEST_CRS,
                )
            )
    return units


def _inputs(tmp_path, missing_unit=False, scaled=False):
    classes = np.ones((100, 100), dtype="uint8")
    classes[:, 50:] = 2
    _write(tmp_path / "class_a.tif", classes, nodata=255)
    _write(tmp_path / "class_b.tif", classes, nodata=255)
    base = (
        0.15
        + np.arange(100, dtype="float32")[None, :] / 1000
        + np.arange(100, dtype="float32")[:, None] / 10000
    )
    a = base.astype("float32")
    b = (base + 0.1).astype("float32")
    b_nodata = -9999.0
    if missing_unit:
        b[90:100, 0:10] = b_nodata
    a_path = _write(
        tmp_path / "ndvi_a.tif",
        (a * 10000).astype("uint16") if scaled else a,
        nodata=None,
        scale=0.0001 if scaled else None,
    )
    b_path = _write(
        tmp_path / "ndvi_b.tif",
        (b * 10000).astype("uint16") if scaled else b,
        nodata=65535 if scaled else b_nodata,
        scale=0.0001 if scaled else None,
    )
    if scaled:
        # Replace the missing value with the declared integer nodata code.
        with rasterio.open(b_path, "r+") as dst:
            values = dst.read(1)
            values[90:100, 0:10] = 65535
            dst.write(values, 1)
    return [
        {
            "label": "a",
            "classification": str(tmp_path / "class_a.tif"),
            "raster": a_path,
        },
        {
            "label": "b",
            "classification": str(tmp_path / "class_b.tif"),
            "raster": b_path,
        },
    ]


def _reflectance_inputs(records):
    from plaknit.processing.reflectance_analysis import ReflectanceInput

    return [
        ReflectanceInput(row["label"], row["classification"], {"NDVI": row["raster"]})
        for row in records
    ]


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_spatial_units_require_unique_ids_and_crs(tmp_path):
    units_path = tmp_path / "units.csv"
    units_path.write_text(
        "square_id,xmin,ymin,xmax,ymax\n1,0,0,100,100\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="spatial-units-crs"):
        load_spatial_units(units_path)
    units = load_spatial_units(units_path, "EPSG:32618")
    assert units[0].unit_id == "1"
    assert units[0].geometry.area == 10000

    units_path.write_text(
        "square_id,xmin,ymin,xmax,ymax\n1,0,0,100,100\n1,100,0,200,100\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        load_spatial_units(units_path, "EPSG:32618")


def test_spatial_comparisons_pair_unit_medians_and_preserve_native_values(tmp_path):
    inputs = _reflectance_inputs(_inputs(tmp_path))
    result = run_reflectance(
        inputs,
        tmp_path / "out",
        spatial_units=_units(),
        block_size_m=100,
        seed=17,
        bootstrap_replicates=200,
        min_independent_blocks=10,
    )
    class_rows = _read_csv(tmp_path / "out" / "reflectance_class_summary.csv")
    assert len(class_rows) == 4
    assert all(row["scaling_method"] == "native" for row in class_rows)
    assert 0.1 < float(class_rows[0]["median_of_unit_medians"]) < 0.4

    unit_rows = _read_csv(tmp_path / "out" / "reflectance_spatial_unit_summary.csv")
    assert len(unit_rows) == 80
    assert {row["unit_id"] for row in unit_rows} == {unit.unit_id for unit in _units()}

    comparisons = _read_csv(tmp_path / "out" / "reflectance_spatial_comparisons.csv")
    assert len(comparisons) == 2
    assert all(row["status"] == "tested" for row in comparisons)
    for row in comparisons:
        assert int(row["paired_unit_count"]) == 10
        assert int(row["independent_block_count"]) == 10
        assert float(row["median_difference_b_minus_a"]) == pytest.approx(0.1)
        estimate = float(row["median_difference_b_minus_a"])
        assert float(row["ci_lower"]) <= estimate <= float(row["ci_upper"])
        assert float(row["paired_rank_biserial_effect"]) == pytest.approx(1)
        assert 0 < float(row["p_value"]) <= 1
        assert 0 < float(row["p_value_bh"]) <= 1
    assert not (tmp_path / "out" / "reflectance_effect_estimates.png").exists()
    report = (tmp_path / "out" / "reflectance_methods_report.md").read_text()
    assert "cannot be attributed to sensor alone" in report
    metadata = json.loads((tmp_path / "out" / "reflectance_metadata.json").read_text())
    assert metadata["spatial_units"]["count"] == 20
    assert metadata["uncertainty"]["block_size_source"] == "user_override"
    assert metadata["value_scaling"].find("No independent min-max") >= 0
    assert {path.name for path in result.outputs} == {
        "reflectance_class_summary.csv",
        "reflectance_spatial_unit_summary.csv",
        "reflectance_spatial_comparisons.csv",
        "reflectance_methods_report.md",
        "reflectance_metadata.json",
    }
    run_reflectance(
        inputs,
        tmp_path / "repeat",
        spatial_units=_units(),
        block_size_m=100,
        seed=17,
        bootstrap_replicates=200,
        min_independent_blocks=10,
    )
    assert (tmp_path / "out" / "reflectance_spatial_comparisons.csv").read_text() == (
        tmp_path / "repeat" / "reflectance_spatial_comparisons.csv"
    ).read_text()


def test_declared_scale_offset_is_applied_instead_of_minmax(tmp_path):
    inputs = _reflectance_inputs(_inputs(tmp_path, scaled=True))
    run_reflectance(
        inputs,
        tmp_path / "out",
        spatial_units=_units(),
        block_size_m=100,
        bootstrap_replicates=100,
        min_independent_blocks=10,
    )
    rows = _read_csv(tmp_path / "out" / "reflectance_spatial_unit_summary.csv")
    a_rows = [row for row in rows if row["input_label"] == "a"]
    assert all(row["scaling_method"] == "metadata-scale-offset" for row in a_rows)
    assert float(a_rows[0]["median"]) == pytest.approx(0.164, abs=0.005)


def test_missing_values_reduce_paired_units_without_pixel_inference(tmp_path):
    inputs = _reflectance_inputs(_inputs(tmp_path, missing_unit=True))
    run_reflectance(
        inputs,
        tmp_path / "out",
        spatial_units=_units(),
        block_size_m=100,
        seed=4,
        min_independent_blocks=10,
        bootstrap_replicates=100,
    )
    rows = _read_csv(tmp_path / "out" / "reflectance_spatial_comparisons.csv")
    class1 = next(row for row in rows if row["class_code"] == "1")
    class2 = next(row for row in rows if row["class_code"] == "2")
    assert class1["paired_unit_count"] == "9"
    assert class1["independent_block_count"] == "9"
    assert class1["status"] == "insufficient_independent_blocks"
    assert class1["p_value"] == "" and class1["ci_lower"] == ""
    assert class2["status"] == "tested"


def test_class_filter_and_no_paired_units_are_explicit(tmp_path):
    inputs = _reflectance_inputs(_inputs(tmp_path))
    run_reflectance(
        inputs,
        tmp_path / "filtered",
        spatial_units=_units(),
        classes=[2],
        block_size_m=100,
        bootstrap_replicates=100,
        min_independent_blocks=10,
    )
    rows = _read_csv(tmp_path / "filtered" / "reflectance_spatial_comparisons.csv")
    assert len(rows) == 1 and rows[0]["class_code"] == "2"
    assert {
        row["class_code"]
        for row in _read_csv(
            tmp_path / "filtered" / "reflectance_spatial_unit_summary.csv"
        )
    } == {"2"}

    no_data = np.full((100, 100), -9999.0, dtype="float32")
    no_data_path = _write(tmp_path / "ndvi_no_data.tif", no_data, nodata=-9999.0)
    inputs[1].rasters = {"NDVI": no_data_path}
    run_reflectance(
        inputs,
        tmp_path / "empty",
        spatial_units=_units(),
        block_size_m=100,
        bootstrap_replicates=100,
    )
    empty_rows = _read_csv(tmp_path / "empty" / "reflectance_spatial_comparisons.csv")
    assert all(row["status"] == "no_paired_units" for row in empty_rows)
    assert all(row["p_value"] == "" for row in empty_rows)


def test_duplicate_units_and_invalid_block_width_fail(tmp_path):
    inputs = _reflectance_inputs(_inputs(tmp_path))
    units = _units()
    duplicate = units + [SpatialUnit(units[0].unit_id, units[0].geometry, TEST_CRS)]
    with pytest.raises(ValueError, match="unique"):
        run_reflectance(inputs, tmp_path / "duplicate", spatial_units=duplicate)
    with pytest.raises(ValueError, match="at least the largest spatial-unit width"):
        run_reflectance(
            inputs,
            tmp_path / "small-block",
            spatial_units=units,
            block_size_m=50,
        )


def test_unresolved_block_range_suppresses_inference(tmp_path):
    inputs = _reflectance_inputs(_inputs(tmp_path))
    result = run_reflectance(
        inputs,
        tmp_path / "out",
        spatial_units=_units(),
        bootstrap_replicates=100,
    )
    rows = _read_csv(tmp_path / "out" / "reflectance_spatial_comparisons.csv")
    assert all(row["status"] == "block_size_unresolved" for row in rows)
    assert all(row["ci_lower"] == "" and row["p_value"] == "" for row in rows)
    metadata = json.loads((tmp_path / "out" / "reflectance_metadata.json").read_text())
    assert metadata["uncertainty"]["range_estimate"]["status"] == "unresolved"
    report = (tmp_path / "out" / "reflectance_methods_report.md").read_text()
    assert "estimates above are descriptive only" in report
    assert len(result.outputs) == 5


def test_empirical_block_estimator_detects_correlated_spatial_differences():
    rng = np.random.default_rng(42)
    from scipy.ndimage import gaussian_filter

    x, y = np.meshgrid(np.arange(30) * 1000, np.arange(30) * 1000)
    coords = np.column_stack((x.ravel(), y.ravel())).astype(float)
    differences = gaussian_filter(rng.normal(size=(30, 30)), sigma=1.5).ravel()
    estimate = _estimate_block_size([(differences, coords)], minimum_size_m=100)
    assert estimate["status"] == "estimated"
    assert estimate["block_size_m"] >= 100
    assert estimate["lag_pair_counts"]


def test_benjamini_hochberg_preserves_positive_small_p_values():
    rows = [
        {"p_value": 1e-300},
        {"p_value": 1e-200},
        {"p_value": None},
    ]
    assert _benjamini_hochberg(rows) == 2
    assert rows[0]["p_value_bh"] > 0
    assert rows[0]["p_value_bh"] == pytest.approx(2e-300)
    assert rows[2]["p_value_bh"] is None


def test_cli_accepts_spatial_units_and_writes_spatial_outputs(tmp_path, capsys):
    records = _inputs(tmp_path)
    units_path = tmp_path / "units.csv"
    with open(units_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["square_id", "xmin", "ymin", "xmax", "ymax"])
        for unit in _units():
            xmin, ymin, xmax, ymax = unit.geometry.bounds
            writer.writerow([unit.unit_id, xmin, ymin, xmax, ymax])
    args = [
        "reflectance",
        "--input",
        f"a={records[0]['classification']}",
        "--raster",
        f"a:NDVI={records[0]['raster']}",
        "--input",
        f"b={records[1]['classification']}",
        "--raster",
        f"b:NDVI={records[1]['raster']}",
        "--spatial-units",
        str(units_path),
        "--spatial-units-crs",
        "EPSG:32618",
        "--block-size-m",
        "100",
        "--out-dir",
        str(tmp_path / "cli"),
    ]
    assert compare_cli.main(args) == 0
    assert (tmp_path / "cli" / "reflectance_spatial_comparisons.csv").exists()
    output = capsys.readouterr().out
    assert "reflectance_spatial_comparisons.csv" in output
    assert "reflectance_effect_estimates.png" not in output


def test_grid_mismatch_resamples_values_and_validity(tmp_path):
    classes = np.ones((100, 100), dtype="uint8")
    _write(tmp_path / "class_a.tif", classes, nodata=255)
    _write(tmp_path / "class_b.tif", classes, nodata=255)
    a = np.full((100, 100), 0.2, dtype="float32")
    b = np.full((50, 50), 0.4, dtype="float32")
    b[0, 0] = -9999
    a_path = _write(tmp_path / "a.tif", a, nodata=-9999)
    b_path = _write(tmp_path / "b_coarse.tif", b, resolution=20, top=1000, nodata=-9999)
    inputs = [
        {
            "label": "a",
            "classification": str(tmp_path / "class_a.tif"),
            "raster": a_path,
        },
        {
            "label": "b",
            "classification": str(tmp_path / "class_b.tif"),
            "raster": b_path,
        },
    ]
    result = run_reflectance(
        _reflectance_inputs(inputs),
        tmp_path / "out",
        spatial_units=_units(),
        block_size_m=100,
        bootstrap_replicates=100,
        min_independent_blocks=10,
    )
    rows = _read_csv(tmp_path / "out" / "reflectance_spatial_unit_summary.csv")
    b_rows = [row for row in rows if row["input_label"] == "b"]
    assert all(float(row["median"]) == pytest.approx(0.4) for row in b_rows)
    assert len(result.outputs) == 5
