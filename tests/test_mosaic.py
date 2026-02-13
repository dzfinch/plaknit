"""Tests for mosaic projection harmonization helpers."""

from pathlib import Path
from datetime import datetime

import pytest
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

    target = mosaic._choose_target_projection(
        projections, requested_crs="EPSG:32731"
    )

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


def test_parse_args_accepts_radiometric_harmonization_flags():
    args = mosaic.parse_args(
        [
            "--inputs",
            "in.tif",
            "--udms",
            "udm.tif",
            "--output",
            "out.tif",
            "--harmonize-radiometry",
            "--metadata-jsons",
            "meta_dir",
            "more_meta",
        ]
    )

    assert args.harmonize_radiometry is True
    assert args.metadata_jsons == ["meta_dir", "more_meta"]


def test_run_requires_metadata_jsons_only_when_harmonization_enabled():
    job = mosaic.MosaicJob(
        inputs=["in.tif"],
        output="out.tif",
        harmonize_radiometry=True,
        metadata_jsons=None,
    )

    workflow = mosaic.MosaicWorkflow(job)
    with pytest.raises(ValueError, match="Metadata JSONs are required"):
        workflow.run()


def test_construct_harmoni_graph_applies_overlap_and_time_filters():
    job = mosaic.MosaicJob(inputs=["in.tif"], output="out.tif")
    workflow = mosaic.MosaicWorkflow(job)
    scenes = [
        mosaic.HarmoniScene(
            raster_path=Path("a.tif"),
            metadata_path=Path("a.json"),
            scene_key="a",
            acquired=datetime(2024, 1, 1),
            bbox_wgs84=(0.0, 0.0, 1.0, 1.0),
            area_wgs84=1.0,
        ),
        mosaic.HarmoniScene(
            raster_path=Path("b.tif"),
            metadata_path=Path("b.json"),
            scene_key="b",
            acquired=datetime(2024, 1, 10),
            bbox_wgs84=(0.2, 0.2, 1.2, 1.2),
            area_wgs84=1.0,
        ),
        mosaic.HarmoniScene(
            raster_path=Path("c.tif"),
            metadata_path=Path("c.json"),
            scene_key="c",
            acquired=datetime(2024, 6, 1),
            bbox_wgs84=(2.0, 2.0, 3.0, 3.0),
            area_wgs84=1.0,
        ),
    ]

    graph = workflow._construct_harmoni_graph(scenes)

    assert 1 in graph[0]
    assert 0 in graph[1]
    assert graph[0][1] > 0.0
    assert graph[2] == {}
