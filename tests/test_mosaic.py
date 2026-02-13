"""Tests for mosaic projection/masking helpers."""

import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest
from affine import Affine
from rasterio.crs import CRS

from plaknit import mosaic


def _projection(path: str, epsg: int, lon: float, lat: float) -> mosaic.ProjectionInfo:
    crs = CRS.from_epsg(epsg)
    return mosaic.ProjectionInfo(
        path=Path(path),
        crs=crs,
        label=mosaic._normalize_crs_label(crs),
        center_lonlat=(lon, lat),
    )


def test_choose_target_projection_prefers_majority_crs():
    projections = [
        _projection("a.tif", 32631, 1.0, 1.0),
        _projection("b.tif", 32631, 1.1, 1.1),
        _projection("c.tif", 32731, 1.2, -0.1),
    ]

    target = mosaic._choose_target_projection(projections, requested_crs=None)

    assert target.to_epsg() == 32631


def test_choose_target_projection_breaks_tie_by_center_distance():
    projections = [
        _projection("n1.tif", 32631, 5.0, 5.0),
        _projection("n2.tif", 32631, 5.1, 5.1),
        _projection("s1.tif", 32731, 0.0, 0.0),
        _projection("s2.tif", 32731, 0.1, 1.0),
    ]

    target = mosaic._choose_target_projection(projections, requested_crs=None)

    assert target.to_epsg() == 32731


def test_choose_target_projection_respects_explicit_override():
    projections = [
        _projection("a.tif", 32631, 1.0, 1.0),
        _projection("b.tif", 32631, 1.2, 0.8),
    ]

    target = mosaic._choose_target_projection(projections, requested_crs="EPSG:32731")

    assert target.to_epsg() == 32731


def test_choose_target_projection_rejects_invalid_override():
    projections = [_projection("a.tif", 32631, 1.0, 1.0)]

    with pytest.raises(ValueError, match="Invalid target CRS"):
        mosaic._choose_target_projection(projections, requested_crs="EPSG:invalid")


def test_parse_args_accepts_target_crs():
    args = mosaic.parse_args(
        [
            "--inputs",
            "in.tif",
            "--udms",
            "udm.tif",
            "--output",
            "out.tif",
            "--target-crs",
            "EPSG:32731",
        ]
    )

    assert args.target_crs == "EPSG:32731"


def test_collect_projection_info_falls_back_to_native_center(monkeypatch, caplog):
    class _FakeDataset:
        def __init__(self):
            self.crs = CRS.from_epsg(32631)
            self.bounds = (10.0, 20.0, 30.0, 60.0)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(mosaic.rasterio, "open", lambda path: _FakeDataset())
    monkeypatch.setattr(
        mosaic,
        "transform_bounds",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad wkt")),
    )

    workflow = mosaic.MosaicWorkflow(
        mosaic.MosaicJob(inputs=["in.tif"], output="out.tif")
    )
    with caplog.at_level("WARNING", logger="plaknit.mosaic"):
        projections = workflow._collect_projection_info(["in.tif"])

    assert projections[0].center_lonlat == (20.0, 40.0)
    assert "using native bounds center for CRS tie-breaking" in caplog.text


def test_read_proj_layout_version_reads_major_minor():
    proj_db_path = Path.cwd() / f"test_proj_{uuid4().hex}.db"
    try:
        with sqlite3.connect(str(proj_db_path)) as conn:
            conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                ("DATABASE.LAYOUT.VERSION.MAJOR", "1"),
            )
            conn.execute(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                ("DATABASE.LAYOUT.VERSION.MINOR", "4"),
            )

        assert mosaic._read_proj_layout_version(proj_db_path) == (1, 4)
    finally:
        try:
            proj_db_path.unlink()
        except (FileNotFoundError, PermissionError):
            pass


def test_warn_on_proj_layout_mismatch_logs_remediation(monkeypatch, caplog):
    monkeypatch.setattr(
        mosaic,
        "_collect_proj_layout_versions",
        lambda: [
            (Path("/opt/proj-a"), (1, 2)),
            (Path("/opt/proj-b"), (1, 4)),
        ],
    )
    workflow = mosaic.MosaicWorkflow(
        mosaic.MosaicJob(inputs=["in.tif"], output="out.tif")
    )

    with caplog.at_level("WARNING", logger="plaknit.mosaic"):
        workflow._warn_on_proj_layout_mismatch()

    assert "Detected mixed PROJ database layouts" in caplog.text
    assert "PROJ_DATA and PROJ_LIB" in caplog.text
    assert "proj-a (layout 1.2)" in caplog.text
    assert "proj-b (layout 1.4)" in caplog.text


def test_warn_on_proj_layout_mismatch_silent_when_layouts_match(monkeypatch, caplog):
    monkeypatch.setattr(
        mosaic,
        "_collect_proj_layout_versions",
        lambda: [
            (Path("/opt/proj-a"), (1, 4)),
            (Path("/opt/proj-b"), (1, 4)),
        ],
    )
    workflow = mosaic.MosaicWorkflow(
        mosaic.MosaicJob(inputs=["in.tif"], output="out.tif")
    )

    with caplog.at_level("WARNING", logger="plaknit.mosaic"):
        workflow._warn_on_proj_layout_mismatch()

    assert "Detected mixed PROJ database layouts" not in caplog.text


def test_mask_single_strip_repairs_missing_georeferencing(monkeypatch, caplog):
    source_state = {
        "crs": CRS.from_epsg(32631),
        "transform": Affine(3.0, 0.0, 100.0, 0.0, -3.0, 200.0),
    }
    masked_state = {"crs": None, "transform": Affine.identity()}

    class _FakeDataset:
        def __init__(self, state):
            self._state = state

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @property
        def crs(self):
            return self._state["crs"]

        @crs.setter
        def crs(self, value):
            self._state["crs"] = value

        @property
        def transform(self):
            return self._state["transform"]

        @transform.setter
        def transform(self, value):
            self._state["transform"] = value

    def _fake_open(path, mode="r", **kwargs):
        path_str = str(path)
        if path_str.endswith("source.tif"):
            return _FakeDataset(source_state)
        if path_str.endswith("masked.tif"):
            return _FakeDataset(masked_state)
        raise AssertionError(f"Unexpected path: {path_str}")

    monkeypatch.setattr(mosaic.rasterio, "open", _fake_open)
    monkeypatch.setattr(mosaic.subprocess, "run", lambda *args, **kwargs: None)
    workflow = mosaic.MosaicWorkflow(
        mosaic.MosaicJob(inputs=["in.tif"], output="out.tif")
    )

    with caplog.at_level("WARNING", logger="plaknit.mosaic"):
        result = workflow._mask_single_strip(
            Path("source.tif"), Path("udm.tif"), Path("masked.tif")
        )

    assert result == Path("masked.tif")
    assert masked_state["crs"] == source_state["crs"]
    assert masked_state["transform"] == source_state["transform"]
    assert "repaired from" in caplog.text


def test_mask_single_strip_repairs_from_udm_when_source_missing(monkeypatch, caplog):
    source_state = {"crs": None, "transform": Affine.identity()}
    udm_state = {
        "crs": CRS.from_epsg(32637),
        "transform": Affine(3.0, 0.0, 500.0, 0.0, -3.0, 1000.0),
    }
    masked_state = {"crs": None, "transform": Affine.identity()}

    class _FakeDataset:
        def __init__(self, state):
            self._state = state

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @property
        def crs(self):
            return self._state["crs"]

        @crs.setter
        def crs(self, value):
            self._state["crs"] = value

        @property
        def transform(self):
            return self._state["transform"]

        @transform.setter
        def transform(self, value):
            self._state["transform"] = value

    def _fake_open(path, mode="r", **kwargs):
        path_str = str(path)
        if path_str.endswith("source.tif"):
            return _FakeDataset(source_state)
        if path_str.endswith("udm.tif"):
            return _FakeDataset(udm_state)
        if path_str.endswith("masked.tif"):
            return _FakeDataset(masked_state)
        raise AssertionError(f"Unexpected path: {path_str}")

    monkeypatch.setattr(mosaic.rasterio, "open", _fake_open)
    monkeypatch.setattr(mosaic.subprocess, "run", lambda *args, **kwargs: None)
    workflow = mosaic.MosaicWorkflow(
        mosaic.MosaicJob(inputs=["in.tif"], output="out.tif")
    )

    with caplog.at_level("WARNING", logger="plaknit.mosaic"):
        result = workflow._mask_single_strip(
            Path("source.tif"), Path("udm.tif"), Path("masked.tif")
        )

    assert result == Path("masked.tif")
    assert masked_state["crs"] == udm_state["crs"]
    assert masked_state["transform"] == udm_state["transform"]
    assert "udm.tif" in caplog.text.lower()
