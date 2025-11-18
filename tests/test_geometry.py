"""Tests for geometry helpers."""

from __future__ import annotations

import math

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
