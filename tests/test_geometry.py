"""Tests for geometry helpers."""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from plaknit import geometry as geometry_utils


def _polygon_with_vertices(vertex_count: int) -> Polygon:
    coords = []
    for idx in range(vertex_count):
        angle = 2 * math.pi * idx / vertex_count
        coords.append((math.cos(angle), math.sin(angle)))
    coords.append(coords[0])
    return Polygon(coords)


def test_geometry_vertex_count_and_simplification():
    polygon = _polygon_with_vertices(2000)
    original_vertices = geometry_utils.geometry_vertex_count(polygon)
    assert original_vertices >= 2000

    simplified = geometry_utils.simplify_geometry_to_vertex_limit(
        polygon, max_vertices=1500
    )
    simplified_vertices = geometry_utils.geometry_vertex_count(simplified)
    assert simplified_vertices <= 1500
    assert simplified_vertices < original_vertices


def test_distance_to_vector_uses_template_grid(tmp_path):
    template_path = tmp_path / "template.tif"
    vector_path = tmp_path / "target.geojson"
    output_path = tmp_path / "distance.tif"
    transform = from_origin(0, 50, 10, 10)

    with rasterio.open(
        template_path,
        "w",
        driver="GTiff",
        width=5,
        height=5,
        count=1,
        dtype="uint8",
        crs="EPSG:32633",
        transform=transform,
    ) as destination:
        destination.write(np.zeros((1, 5, 5), dtype="uint8"))

    gpd.GeoDataFrame(
        {"geometry": [Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])]},
        crs="EPSG:32633",
    ).to_file(vector_path, driver="GeoJSON")

    geometry_utils.distance_to_vector(template_path, vector_path, output_path)

    with rasterio.open(output_path) as source:
        distances = source.read(1)
        assert source.crs.to_epsg() == 32633
        assert distances[2, 2] == 0
        assert distances[2, 3] == 10
        assert distances[1, 1] == pytest.approx(np.sqrt(200))
