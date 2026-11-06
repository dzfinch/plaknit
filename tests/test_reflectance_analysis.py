import csv
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from plaknit.cli import compare_cli
from plaknit.processing.reflectance_analysis import (
    ReflectanceInput,
    run_reflectance,
    scale_to_unit,
)

CRS = "EPSG:32618"


def _write(path, values, top=None, nodata=None, scale=None, offset=None):
    values = np.asarray(values)
    top = values.shape[0] * 10 if top is None else top
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=values.dtype,
        crs=CRS,
        transform=from_origin(0, top, 10, 10),
        nodata=nodata,
    ) as dst:
        dst.write(values, 1)
        if scale is not None:
            dst.scales = (scale,)
            dst.offsets = (offset or 0.0,)
    return str(path)


def _inputs(tmp_path, metadata=True):
    classes = (np.indices((24, 24)).sum(axis=0) % 2 + 1).astype("uint8")
    items = []
    for i, label in enumerate(("a", "b")):
        base = np.where(classes == 1, 1000 + 500 * i, 4000 + 500 * i)
        noise = np.random.default_rng(i).integers(0, 200, classes.shape)
        red = (base + noise).astype("uint16")
        items.append(
            ReflectanceInput(
                label,
                _write(tmp_path / f"class_{label}.tif", classes, nodata=255),
                {
                    "red": _write(
                        tmp_path / f"red_{label}.tif",
                        red,
                        scale=0.0001 if metadata else None,
                        offset=0.0,
                    )
                },
            )
        )
    return items


def test_metadata_scaling_is_used_when_it_maps_to_unit_range():
    values = np.array([[1000.0, 5000.0]])
    scaled, info = scale_to_unit(values, np.ones_like(values, bool), 1e-4, 0.0, "x")
    assert info["method"] == "metadata"
    assert scaled[0].tolist() == pytest.approx([0.1, 0.5])


def test_min_max_fallback_without_or_with_bad_metadata():
    values = np.array([[10.0, 20.0, 30.0]])
    valid = np.array([[True, True, True]])
    with pytest.warns(UserWarning, match="min/max"):
        scaled, info = scale_to_unit(values, valid, 1.0, 0.0, "x")
    assert info["method"] == "min-max"
    assert scaled[0].tolist() == pytest.approx([0.0, 0.5, 1.0])
    with pytest.warns(UserWarning, match="not 0-1"):
        _, info = scale_to_unit(values, valid, 1.0, 5.0, "x")
    assert info["method"] == "min-max"


def test_constant_raster_cannot_be_min_max_scaled():
    values = np.full((2, 2), 7.0)
    with pytest.raises(ValueError, match="constant"):
        scale_to_unit(values, np.ones_like(values, bool), 1.0, 0.0, "x")


def test_mann_whitney_outputs_and_reproducibility(tmp_path):
    inputs = _inputs(tmp_path)
    first = run_reflectance(inputs, tmp_path / "one", seed=5)
    run_reflectance(inputs, tmp_path / "two", seed=5)
    with open(tmp_path / "one" / "reflectance_class_summary.csv", newline="") as f:
        summary = list(csv.DictReader(f))
    assert len(summary) == 4
    assert all(
        int(r["sampled_pixel_count"]) == int(np.ceil(0.95 * int(r["eligible_pixel_count"])))
        for r in summary
    )
    assert all(r["scaling_method"] == "metadata" for r in summary)
    assert all(0 <= float(r["median"]) <= 1 for r in summary)
    with open(tmp_path / "one" / "reflectance_mann_whitney.csv", newline="") as f:
        tests = list(csv.DictReader(f))
    assert {r["class_code"] for r in tests} == {"1", "2"}
    assert all(r["status"] == "tested" for r in tests)
    assert all(float(r["p_value_bh"]) < 0.05 for r in tests)
    assert all(float(r["rank_biserial_a_vs_b"]) == pytest.approx(-1) for r in tests)
    assert (tmp_path / "one" / "reflectance_red_class_1.png").exists()
    meta = json.loads((tmp_path / "one" / "reflectance_metadata.json").read_text())
    assert meta["seed"] == 5 and meta["sample_fraction"] == 0.95
    assert first.scaling["a"]["red"]["method"] == "metadata"
    assert (tmp_path / "one" / "reflectance_class_summary.csv").read_text() == (
        tmp_path / "two" / "reflectance_class_summary.csv"
    ).read_text()


def test_min_max_fallback_is_recorded_without_metadata(tmp_path):
    inputs = _inputs(tmp_path, metadata=False)
    with pytest.warns(UserWarning, match="min/max"):
        result = run_reflectance(inputs, tmp_path / "out", seed=1)
    assert result.scaling["a"]["red"]["method"] == "min-max"


def test_raster_must_match_classification_grid(tmp_path):
    inputs = _inputs(tmp_path)
    bad = _write(tmp_path / "bad.tif", np.ones((24, 24), "float32") * 0.5, top=241)
    inputs[0].rasters = {"red": bad}
    with pytest.raises(ValueError, match="exactly match"):
        run_reflectance(inputs, tmp_path / "out")


def test_cli_reflectance_subcommand(tmp_path, capsys):
    inputs = _inputs(tmp_path)
    args = ["reflectance", "--out-dir", str(tmp_path / "cli"), "--seed", "2"]
    for item in inputs:
        args += ["--input", f"{item.label}={item.classification}"]
        args += ["--raster", f"{item.label}:red={item.rasters['red']}"]
    assert compare_cli.main(args) == 0
    assert (tmp_path / "cli" / "reflectance_mann_whitney.csv").exists()
    assert "Wrote" in capsys.readouterr().out
