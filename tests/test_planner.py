"""Tests for the planning helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List

import pytest
from shapely.geometry import box, mapping

from plaknit import planner


class _FakeItem:
    def __init__(self, item_id: str, geom_dict: Dict, properties: Dict):
        self.id = item_id
        self.geometry = geom_dict
        self.properties = properties
        self.collection_id = "PSScene"


class _FakeSearch:
    def __init__(self, items: Iterable[_FakeItem]):
        self._items = list(items)

    def items(self) -> Iterable[_FakeItem]:
        return iter(self._items)


class _FakeClient:
    def __init__(self, response_map: Dict[str, List[_FakeItem]]):
        self.response_map = response_map

    def search(self, **kwargs):
        datetime_range = kwargs.get("datetime")
        items = self.response_map.get(datetime_range, [])
        return _FakeSearch(items)


def test_plan_monthly_composites_selects_scenes(monkeypatch):
    geom = box(0.0, 0.0, 0.01, 0.01)

    fake_items = [
        _FakeItem(
            "scene-1",
            mapping(geom),
            {"cloud_cover": 0.4, "sun_elevation": 50},
        ),
        _FakeItem(
            "scene-2",
            mapping(geom),
            {"cloud_cover": 0.3, "sun_elevation": 55},
        ),
    ]
    response_map = {"2024-01-01/2024-01-31": fake_items}
    fake_client = _FakeClient(response_map)

    monkeypatch.setenv("PL_API_KEY", "test-key")
    monkeypatch.setattr(planner, "load_aoi_geometry", lambda path: (geom, "EPSG:4326"))
    monkeypatch.setattr(planner, "_open_planet_stac_client", lambda key: fake_client)

    plan = planner.plan_monthly_composites(
        aoi_path="aoi.geojson",
        start_date="2024-01-01",
        end_date="2024-01-31",
        cloud_max=0.8,
        coverage_target=0.95,
        min_clear_obs=1.0,
        min_clear_fraction=0.5,
        tile_size_m=500,
    )

    assert "2024-01" in plan
    month_plan = plan["2024-01"]
    assert month_plan["candidate_count"] == 2
    assert month_plan["filtered_count"] == 2
    assert month_plan["selected_count"] >= 1
    assert month_plan["aoi_coverage"] >= 0.95
    assert month_plan["items"]
    assert month_plan["items"][0]["properties"]["sun_elevation"] >= 50


def test_plan_skips_scenes_missing_clear_metadata(monkeypatch):
    geom = box(0.0, 0.0, 0.01, 0.01)
    fake_items = [
        _FakeItem(
            "scene-missing",
            mapping(geom),
            {"sun_elevation": 55},
        )
    ]
    response_map = {"2024-01-01/2024-01-31": fake_items}
    fake_client = _FakeClient(response_map)

    monkeypatch.setenv("PL_API_KEY", "test-key")
    monkeypatch.setattr(planner, "load_aoi_geometry", lambda path: (geom, "EPSG:4326"))
    monkeypatch.setattr(planner, "_open_planet_stac_client", lambda key: fake_client)

    plan = planner.plan_monthly_composites(
        aoi_path="aoi.geojson",
        start_date="2024-01-01",
        end_date="2024-01-31",
        cloud_max=0.8,
        coverage_target=0.5,
        min_clear_obs=1.0,
        min_clear_fraction=0.5,
        tile_size_m=500,
    )

    month_plan = plan["2024-01"]
    assert month_plan["candidate_count"] == 1
    assert month_plan["filtered_count"] == 0
    assert month_plan["selected_count"] == 0


def test_lighting_similarity_matches_identical_conditions():
    tile_state = planner._TileState()
    planner._update_tile_lighting(tile_state, 110.0, 45.0, 0.7)

    similarity = planner._lighting_similarity(
        tile_state, 110.0, 45.0, azimuth_sigma=20.0, elevation_sigma=10.0
    )

    assert similarity == pytest.approx(1.0, rel=1e-6)


def test_lighting_similarity_penalizes_large_azimuth_difference():
    tile_state = planner._TileState()
    planner._update_tile_lighting(tile_state, 0.0, 45.0, 1.0)

    similarity = planner._lighting_similarity(
        tile_state, 90.0, 45.0, azimuth_sigma=20.0, elevation_sigma=10.0
    )

    assert similarity < 1e-3


def test_lighting_similarity_ignores_missing_metadata():
    tile_state = planner._TileState()
    planner._update_tile_lighting(tile_state, 200.0, 50.0, 0.5)

    similarity = planner._lighting_similarity(
        tile_state, None, None, azimuth_sigma=20.0, elevation_sigma=10.0
    )

    assert similarity == pytest.approx(1.0, rel=1e-6)
