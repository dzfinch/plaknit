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
        self.last_kwargs: Dict | None = None
        self.calls: List[Dict] = []

    def search(self, **kwargs):
        self.last_kwargs = kwargs
        self.calls.append(kwargs)
        datetime_range = kwargs.get("datetime")
        items = self.response_map.get(datetime_range, [])
        return _FakeSearch(items)


def test_plan_monthly_composites_selects_scenes(monkeypatch):
    geom = box(0.0, 0.0, 0.01, 0.01)

    fake_items = [
        _FakeItem(
            "scene-1",
            mapping(geom),
            {
                "eo:cloud_cover": 0.4,
                "view:sun_elevation": 50,
                "view:sun_azimuth": 100,
                "datetime": "2024-01-01T10:00:00Z",
                "pl:clear_confidence_percent": 85,
            },
        ),
        _FakeItem(
            "scene-2",
            mapping(geom),
            {
                "eo:cloud_cover": 0.3,
                "view:sun_elevation": 55,
                "view:sun_azimuth": 120,
                "datetime": "2024-01-02T10:00:00Z",
                "pl:visible_confidence_percent": 90,
            },
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
    assert month_plan["items"][0]["properties"]["view:sun_elevation"] >= 50
    assert month_plan["items"][0]["properties"]["datetime"]
    assert "visible_confidence_percent" not in month_plan["items"][0]["properties"]
    assert "clear_confidence_percent" not in month_plan["items"][0]["properties"]
    assert "shadow_percent" not in month_plan["items"][0]["properties"]
    assert "snow_ice_percent" not in month_plan["items"][0]["properties"]
    assert "heavy_haze_percent" not in month_plan["items"][0]["properties"]
    assert "anomalous_pixels" not in month_plan["items"][0]["properties"]
    assert fake_client.last_kwargs is not None
    assert fake_client.last_kwargs["query"]["view:sun_elevation"] == {"gte": 35.0}
    assert fake_client.last_kwargs["query"]["eo:cloud_cover"] == {"lte": 0.8}


def test_plan_skips_scenes_missing_clear_metadata(monkeypatch):
    geom = box(0.0, 0.0, 0.01, 0.01)
    fake_items = [
        _FakeItem(
            "scene-missing",
            mapping(geom),
            {"view:sun_elevation": 55},
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


def test_plan_filters_on_stable_quality_metadata(monkeypatch):
    geom = box(0.0, 0.0, 0.01, 0.01)
    fake_items = [
        _FakeItem(
            "scene-good",
            mapping(geom),
            {
                "eo:cloud_cover": 0.05,
                "view:sun_elevation": 50,
                "pl:shadow_percent": 1,
                "pl:ground_control": True,
                "pl:quality_category": "standard",
            },
        ),
        _FakeItem(
            "scene-bad-ground-control",
            mapping(geom),
            {
                "eo:cloud_cover": 0.05,
                "view:sun_elevation": 50,
                "pl:shadow_percent": 25,
                "pl:ground_control": False,
                "pl:quality_category": "standard",
            },
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
        coverage_target=0.5,
        min_clear_obs=1.0,
        min_clear_fraction=0.5,
        require_ground_control=True,
        quality_category="standard",
        tile_size_m=500,
    )

    month_plan = plan["2024-01"]
    assert month_plan["filtered_count"] == 1
    assert month_plan["items"][0]["id"] == "scene-good"


def test_score_candidate_prefers_higher_quality():
    tile_states = [planner._TileState()]
    high_quality = planner._Candidate(
        item_id="high",
        collection_id="PSScene",
        properties={},
        clear_fraction=0.9,
        tile_indexes=[0],
        quality_score=1.0,
    )
    low_quality = planner._Candidate(
        item_id="low",
        collection_id="PSScene",
        properties={},
        clear_fraction=0.9,
        tile_indexes=[0],
        quality_score=0.1,
    )

    high_score = planner._score_candidate(
        high_quality,
        tile_states,
        min_clear_obs=1.0,
        azimuth_sigma=20.0,
        elevation_sigma=10.0,
        quality_weight=0.5,
    )
    low_score = planner._score_candidate(
        low_quality,
        tile_states,
        min_clear_obs=1.0,
        azimuth_sigma=20.0,
        elevation_sigma=10.0,
        quality_weight=0.5,
    )

    assert high_score > low_score


def test_plan_supports_single_window_grouping(monkeypatch):
    geom = box(0.0, 0.0, 0.01, 0.01)
    fake_items = [
        _FakeItem(
            "scene-1",
            mapping(geom),
            {
                "pl:clear_percent": 95,
                "view:sun_elevation": 50,
                "eo:cloud_cover": 0.03,
            },
        )
    ]
    response_map = {"2024-01-10/2024-03-05": fake_items}
    fake_client = _FakeClient(response_map)

    monkeypatch.setenv("PL_API_KEY", "test-key")
    monkeypatch.setattr(planner, "load_aoi_geometry", lambda path: (geom, "EPSG:4326"))
    monkeypatch.setattr(planner, "_open_planet_stac_client", lambda key: fake_client)

    plan = planner.plan_monthly_composites(
        aoi_path="aoi.geojson",
        start_date="2024-01-10",
        end_date="2024-03-05",
        cloud_max=0.8,
        coverage_target=0.5,
        min_clear_obs=1.0,
        min_clear_fraction=0.5,
        month_grouping="single",
        tile_size_m=500,
    )

    assert list(plan.keys()) == ["2024-01-10_to_2024-03-05"]
    assert plan["2024-01-10_to_2024-03-05"]["candidate_count"] == 1


def test_plan_supports_fixed_window_days_grouping(monkeypatch):
    geom = box(0.0, 0.0, 0.01, 0.01)
    fake_items = [
        _FakeItem(
            "scene-1",
            mapping(geom),
            {
                "pl:clear_percent": 95,
                "view:sun_elevation": 50,
                "eo:cloud_cover": 0.03,
            },
        )
    ]
    response_map = {
        "2024-01-01/2024-01-01": fake_items,
        "2024-01-02/2024-01-02": fake_items,
        "2024-01-03/2024-01-03": fake_items,
    }
    fake_client = _FakeClient(response_map)

    monkeypatch.setenv("PL_API_KEY", "test-key")
    monkeypatch.setattr(planner, "load_aoi_geometry", lambda path: (geom, "EPSG:4326"))
    monkeypatch.setattr(planner, "_open_planet_stac_client", lambda key: fake_client)

    plan = planner.plan_monthly_composites(
        aoi_path="aoi.geojson",
        start_date="2024-01-01",
        end_date="2024-01-03",
        cloud_max=0.8,
        coverage_target=0.5,
        min_clear_obs=1.0,
        min_clear_fraction=0.5,
        month_grouping="fixed",
        window_days=1,
        tile_size_m=500,
    )

    assert list(plan.keys()) == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert len(fake_client.calls) == 3
    assert fake_client.calls[0]["datetime"] == "2024-01-01/2024-01-01"


def test_plan_filters_returned_items_by_instrument_type(monkeypatch):
    geom = box(0.0, 0.0, 0.01, 0.01)
    fake_items = [
        _FakeItem(
            "scene-psb",
            mapping(geom),
            {
                "pl:clear_percent": 99,
                "view:sun_elevation": 50,
                "eo:cloud_cover": 0.01,
                "instruments": ["PSB.SD"],
            },
        ),
        _FakeItem(
            "scene-ps2",
            mapping(geom),
            {
                "pl:clear_percent": 80,
                "view:sun_elevation": 50,
                "eo:cloud_cover": 0.01,
                "instruments": ["PS2.SD"],
            },
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
        coverage_target=0.5,
        min_clear_obs=1.0,
        min_clear_fraction=0.5,
        instrument_types=("PS2.SD",),
        tile_size_m=500,
    )

    month_plan = plan["2024-01"]
    assert month_plan["filtered_count"] == 1
    assert [item["id"] for item in month_plan["items"]] == ["scene-ps2"]
    assert fake_client.last_kwargs is not None
    assert fake_client.last_kwargs["query"]["instruments"] == {"in": ["PS2.SD"]}


def test_plan_filters_instruments_array_field(monkeypatch):
    geom = box(0.0, 0.0, 0.01, 0.01)
    fake_items = [
        _FakeItem(
            "scene-list-psb",
            mapping(geom),
            {
                "pl:clear_percent": 99,
                "view:sun_elevation": 50,
                "eo:cloud_cover": 0.01,
                "instruments": ["PSB.SD"],
            },
        ),
        _FakeItem(
            "scene-list-ps2",
            mapping(geom),
            {
                "pl:clear_percent": 95,
                "view:sun_elevation": 50,
                "eo:cloud_cover": 0.01,
                "instruments": ["PS2.SD"],
            },
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
        coverage_target=0.5,
        min_clear_obs=1.0,
        min_clear_fraction=0.5,
        instrument_types=("PS2.SD",),
        tile_size_m=500,
    )

    month_plan = plan["2024-01"]
    assert [item["id"] for item in month_plan["items"]] == ["scene-list-ps2"]
    assert month_plan["items"][0]["properties"]["instruments"] == ["PS2.SD"]


def test_plan_single_instrument_falls_back_when_metadata_missing(monkeypatch):
    geom = box(0.0, 0.0, 0.01, 0.01)
    fake_items = [
        _FakeItem(
            "scene-1",
            mapping(geom),
            {
                "pl:clear_percent": 95,
                "view:sun_elevation": 50,
                "eo:cloud_cover": 0.03,
            },
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
        instrument_types=("PS2.SD",),
        tile_size_m=500,
    )

    month_plan = plan["2024-01"]
    assert month_plan["filtered_count"] == 1
    assert month_plan["selected_count"] == 1


def test_plan_defaults_to_legacy_selection_policy(monkeypatch):
    geom = box(0.0, 0.0, 0.01, 0.01)
    fake_items = [
        _FakeItem(
            "scene-1",
            mapping(geom),
            {
                "pl:clear_percent": 95,
                "view:sun_elevation": 50,
                "eo:cloud_cover": 0.03,
                "datetime": "2024-01-01T10:00:00Z",
            },
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
        coverage_target=0.5,
        min_clear_obs=1.0,
        min_clear_fraction=0.5,
        tile_size_m=500,
    )
    month_plan = plan["2024-01"]
    assert month_plan["selection_policy"] == "legacy"
    assert month_plan["stopped_reason"] in {"coverage_target_met", "no_positive_gain"}


def test_parse_plan_args_rejects_removed_cohesive_flags():
    with pytest.raises(SystemExit):
        planner.parse_plan_args(
            [
                "--aoi",
                "aoi.geojson",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-31",
                "--instrument-type",
                "PSB.SD",
                "--cohesive",
            ]
        )


def test_parse_plan_args_requires_window_days_for_fixed_grouping():
    with pytest.raises(SystemExit):
        planner.parse_plan_args(
            [
                "--aoi",
                "aoi.geojson",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-31",
                "--instrument-type",
                "PSB.SD",
                "--grouping",
                "fixed",
            ]
        )


def test_parse_plan_args_rejects_window_days_without_fixed_grouping():
    with pytest.raises(SystemExit):
        planner.parse_plan_args(
            [
                "--aoi",
                "aoi.geojson",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-31",
                "--instrument-type",
                "PSB.SD",
                "--grouping",
                "calendar",
                "--window-days",
                "10",
            ]
        )
