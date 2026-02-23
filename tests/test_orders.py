"""Tests for the Planet orders helper."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from shapely.geometry import box, shape

from plaknit import orders


class _FakeOrdersClient:
    def __init__(self):
        self.requests: list[dict] = []

    async def create_order(self, request: dict) -> dict:
        self.requests.append(request)
        return {"id": "order-abc"}


class _AccessErrorOrdersClient:
    def __init__(self):
        self.requests: list[dict] = []
        self.first_attempt = True

    async def create_order(self, request: dict) -> dict:
        if self.first_attempt:
            self.first_attempt = False
            raise RuntimeError(
                json.dumps(
                    {
                        "field": {
                            "Details": [
                                {
                                    "message": "no access to assets: PSScene/item-2/[ortho_analytic_4b_sr]"
                                }
                            ]
                        }
                    }
                )
            )
        self.requests.append(request)
        return {"id": "order-success"}


class _FakeStacItem:
    def __init__(self, item_id: str, properties: dict):
        self.id = item_id
        self.properties = properties
        self.collection_id = "PSScene"


class _FakeStacSearch:
    def __init__(self, items):
        self._items = items

    def items(self):
        return iter(self._items)


class _FakeStacClient:
    def __init__(self, items):
        self._items = items
        self.last_kwargs: dict[str, Any] | None = None

    def search(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeStacSearch(self._items)


def test_submit_orders_for_plan_builds_correct_request(monkeypatch):
    plan = {
        "2024-01": {
            "items": [{"id": "item-1"}, {"id": "item-2"}],
            "selected_count": 2,
        },
        "2024-02": {
            "items": [],
            "selected_count": 0,
        },
    }

    fake_geom = box(0, 0, 1, 1)
    monkeypatch.setenv("PL_API_KEY", "test-key")
    monkeypatch.setattr(
        orders, "load_aoi_geometry", lambda path: (fake_geom, "EPSG:4326")
    )
    monkeypatch.setattr(orders, "reproject_geometry", lambda geom, src, dst: geom)
    monkeypatch.setattr(
        orders, "_open_planet_stac_client", lambda key: _FakeStacClient([])
    )

    fake_client = _FakeOrdersClient()

    @asynccontextmanager
    async def fake_context(api_key: str):
        yield fake_client

    monkeypatch.setattr(orders, "_orders_client_context", fake_context)

    result = orders.submit_orders_for_plan(
        plan=plan,
        aoi_path="aoi.geojson",
        sr_bands=8,
        harmonize_to="sentinel2",
        order_prefix="plaknit_plan",
        archive_type="zip",
        single_archive=True,
    )

    assert len(fake_client.requests) == 1
    request = fake_client.requests[0]
    assert request["name"] == "plaknit_plan_2024-01"
    assert request["products"][0]["product_bundle"] == "analytic_8b_sr_udm2"
    assert request["delivery"]["archive_type"] == "zip"
    assert request["delivery"]["single_archive"] is True
    assert request["delivery"]["archive_filename"] == "plaknit_plan_2024-01.zip"
    tools = request["tools"]
    assert any("clip" in tool for tool in tools)
    assert any(
        tool.get("harmonize", {}).get("target_sensor") == "Sentinel-2" for tool in tools
    )

    assert result["2024-01"]["order_id"] == "order-abc"
    assert result["2024-02"]["order_id"] is None


def test_submit_orders_drops_inaccessible_scenes(monkeypatch):
    plan = {
        "2024-01": {
            "items": [{"id": "item-1"}, {"id": "item-2"}],
            "selected_count": 2,
        }
    }

    fake_geom = box(0, 0, 1, 1)
    monkeypatch.setenv("PL_API_KEY", "test-key")
    monkeypatch.setattr(
        orders, "load_aoi_geometry", lambda path: (fake_geom, "EPSG:4326")
    )
    monkeypatch.setattr(orders, "reproject_geometry", lambda geom, src, dst: geom)
    monkeypatch.setattr(
        orders, "_open_planet_stac_client", lambda key: _FakeStacClient([])
    )

    fake_client = _AccessErrorOrdersClient()

    @asynccontextmanager
    async def fake_context(api_key: str):
        yield fake_client

    monkeypatch.setattr(orders, "_orders_client_context", fake_context)

    result = orders.submit_orders_for_plan(
        plan=plan,
        aoi_path="aoi.geojson",
        sr_bands=4,
        harmonize_to=None,
        order_prefix="demo",
        archive_type="zip",
        single_archive=True,
    )

    assert fake_client.requests, "Expected successful retry after dropping scenes."
    request = fake_client.requests[-1]
    assert request["products"][0]["item_ids"] == ["item-1"]
    assert result["2024-01"]["item_ids"] == ["item-1"]


def test_extract_inaccessible_item_ids_handles_null_field():
    payload = {"field": None, "general": [{"message": "AOI too complex"}]}

    result = orders._extract_inaccessible_item_ids(RuntimeError(json.dumps(payload)))

    assert result == []


def test_clip_geojson_simplifies_when_vertex_count_exceeds_limit(monkeypatch):
    detailed_geom = box(0, 0, 1, 1)
    simplified_geom = box(0, 0, 0.5, 0.5)

    monkeypatch.setattr(
        orders, "load_aoi_geometry", lambda path: (detailed_geom, "EPSG:4326")
    )
    monkeypatch.setattr(orders, "reproject_geometry", lambda geom, src, dst: geom)

    counts = iter([1601, 300])
    monkeypatch.setattr(orders, "geometry_vertex_count", lambda geom: next(counts))

    called: dict[str, Any] = {}

    def fake_simplify(geom, max_vertices):
        called["max_vertices"] = max_vertices
        return simplified_geom

    monkeypatch.setattr(orders, "simplify_geometry_to_vertex_limit", fake_simplify)

    geojson = orders._clip_geojson("aoi.geojson")

    assert called["max_vertices"] == 1500
    assert shape(geojson).equals(simplified_geom)


def test_order_cli_reads_plan_and_submits(monkeypatch):
    plan = {"2024-01": {"items": [{"id": "item-1"}]}}
    plan_path = Path.cwd() / f"test_plan_{uuid4().hex}.geojson"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    captured: dict[str, Any] = {}

    def fake_submit(**kwargs):
        captured.update(kwargs)
        return {"2024-01": {"order_id": "order-xyz", "item_ids": ["item-1"]}}

    monkeypatch.setattr(orders, "submit_orders_for_plan", fake_submit)

    try:
        exit_code = orders.main(
            [
                "--plan",
                str(plan_path),
                "--aoi",
                "aoi.geojson",
                "--sr-bands",
                "8",
                "--harmonize-to",
                "none",
                "--order-prefix",
                "demo",
                "--archive-type",
                "tar",
                "--no-single-archive",
            ]
        )
    finally:
        try:
            plan_path.unlink()
        except FileNotFoundError:
            pass

    assert exit_code == 0
    assert captured["plan"] == plan
    assert captured["aoi_path"] == "aoi.geojson"
    assert captured["sr_bands"] == 8
    assert captured["harmonize_to"] is None
    assert captured["order_prefix"] == "demo"
    assert captured["archive_type"] == "tar"
    assert captured["single_archive"] is False


def test_find_replacement_items_ignores_udm_filters_and_ranks_by_clear_fraction():
    plan_entry = {
        "filters": {
            "item_type": "PSScene",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "cloud_max": 0.1,
            "sun_elevation_min": 35.0,
            "min_clear_fraction": 0.5,
            "require_ground_control": True,
            "max_shadow_fraction": 0.05,
            "max_view_angle": 10,
            "quality_weight": 0.5,
        }
    }
    items = [
        _FakeStacItem(
            "lower-clear",
            {
                "pl:clear_percent": 96,
                "eo:cloud_cover": 0.04,
                "pl:ground_control": True,
                "pl:shadow_percent": 1,
                "view:off_nadir": 5.0,
                "view:sun_elevation": 47.0,
                "view:sun_azimuth": 130.0,
                "datetime": "2024-01-10T10:00:00Z",
            },
        ),
        _FakeStacItem(
            "higher-clear-high-shadow",
            {
                "pl:clear_percent": 99,
                "eo:cloud_cover": 0.01,
                "pl:ground_control": True,
                "pl:shadow_percent": 40,
                "view:off_nadir": 2.0,
                "view:sun_elevation": 48.0,
                "view:sun_azimuth": 131.0,
                "datetime": "2024-01-11T10:00:00Z",
            },
        ),
        _FakeStacItem(
            "highest-clear-no-ground-control",
            {
                "pl:clear_percent": 100,
                "eo:cloud_cover": 0.0,
                "pl:ground_control": False,
                "view:off_nadir": 1.0,
                "view:sun_elevation": 49.0,
            },
        ),
    ]
    stac_client = _FakeStacClient(items)

    replacements = orders._find_replacement_items(
        stac_client=stac_client,
        plan_entry=plan_entry,
        month="2024-01",
        aoi_geojson={"type": "Polygon", "coordinates": []},
        desired_count=3,
        exclude_ids=set(),
    )

    # require_ground_control still applies; max_shadow_fraction is ignored.
    assert [item["id"] for item in replacements] == [
        "higher-clear-high-shadow",
        "lower-clear",
    ]
    assert stac_client.last_kwargs is not None
    assert stac_client.last_kwargs["query"]["view:sun_elevation"] == {"gte": 35.0}
    assert stac_client.last_kwargs["query"]["eo:cloud_cover"] == {"lte": 0.1}
    assert replacements[0]["properties"]["view:sun_elevation"] == 48.0
    assert replacements[0]["properties"]["view:sun_azimuth"] == 131.0
    assert replacements[0]["properties"]["datetime"] == "2024-01-11T10:00:00Z"
    assert replacements[0]["properties"]["view:off_nadir"] == 2.0
    assert "visible_confidence_percent" not in replacements[0]["properties"]
    assert "clear_confidence_percent" not in replacements[0]["properties"]
    assert "shadow_percent" not in replacements[0]["properties"]
    assert "snow_ice_percent" not in replacements[0]["properties"]
    assert "heavy_haze_percent" not in replacements[0]["properties"]
    assert "anomalous_pixels" not in replacements[0]["properties"]


def test_find_replacement_items_honors_instrument_types_filter():
    plan_entry = {
        "filters": {
            "item_type": "PSScene",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "cloud_max": 0.1,
            "sun_elevation_min": 35.0,
            "min_clear_fraction": 0.5,
            "instrument_types": ["PS2.SD"],
        }
    }
    items = [
        _FakeStacItem(
            "psb-scene",
            {
                "pl:clear_percent": 99,
                "eo:cloud_cover": 0.01,
                "instruments": ["PSB.SD"],
                "view:sun_elevation": 47.0,
            },
        ),
        _FakeStacItem(
            "ps2-scene",
            {
                "pl:clear_percent": 90,
                "eo:cloud_cover": 0.02,
                "instruments": ["PS2.SD"],
                "view:sun_elevation": 47.0,
            },
        ),
    ]
    stac_client = _FakeStacClient(items)

    replacements = orders._find_replacement_items(
        stac_client=stac_client,
        plan_entry=plan_entry,
        month="2024-01",
        aoi_geojson={"type": "Polygon", "coordinates": []},
        desired_count=2,
        exclude_ids=set(),
    )

    assert [item["id"] for item in replacements] == ["ps2-scene"]
    assert stac_client.last_kwargs is not None
    assert stac_client.last_kwargs["query"]["instruments"] == {"in": ["PS2.SD"]}


def test_find_replacement_items_honors_instruments_array_field():
    plan_entry = {
        "filters": {
            "item_type": "PSScene",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "cloud_max": 0.1,
            "sun_elevation_min": 35.0,
            "min_clear_fraction": 0.5,
            "instrument_types": ["PS2.SD"],
        }
    }
    items = [
        _FakeStacItem(
            "psb-scene",
            {
                "pl:clear_percent": 99,
                "eo:cloud_cover": 0.01,
                "instruments": ["PSB.SD"],
                "view:sun_elevation": 47.0,
            },
        ),
        _FakeStacItem(
            "ps2-scene",
            {
                "pl:clear_percent": 90,
                "eo:cloud_cover": 0.02,
                "instruments": ["PS2.SD"],
                "view:sun_elevation": 47.0,
            },
        ),
    ]
    stac_client = _FakeStacClient(items)

    replacements = orders._find_replacement_items(
        stac_client=stac_client,
        plan_entry=plan_entry,
        month="2024-01",
        aoi_geojson={"type": "Polygon", "coordinates": []},
        desired_count=2,
        exclude_ids=set(),
    )

    assert [item["id"] for item in replacements] == ["ps2-scene"]
    assert replacements[0]["properties"]["instruments"] == ["PS2.SD"]


def test_find_replacement_items_single_instrument_falls_back_when_metadata_missing():
    plan_entry = {
        "filters": {
            "item_type": "PSScene",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "cloud_max": 0.1,
            "sun_elevation_min": 35.0,
            "min_clear_fraction": 0.5,
            "instrument_types": ["PS2.SD"],
        }
    }
    items = [
        _FakeStacItem(
            "scene-no-inst",
            {
                "pl:clear_percent": 92,
                "eo:cloud_cover": 0.02,
                "view:sun_elevation": 47.0,
            },
        )
    ]
    stac_client = _FakeStacClient(items)

    replacements = orders._find_replacement_items(
        stac_client=stac_client,
        plan_entry=plan_entry,
        month="2024-01",
        aoi_geojson={"type": "Polygon", "coordinates": []},
        desired_count=1,
        exclude_ids=set(),
    )

    assert [item["id"] for item in replacements] == ["scene-no-inst"]


def test_month_start_end_accepts_legacy_filter_keys():
    plan_entry = {
        "filters": {
            "month_start": "2024-01-02",
            "month_end": "2024-01-30",
        }
    }
    start, end = orders._month_start_end("2024-01", plan_entry)
    assert start.isoformat() == "2024-01-02"
    assert end.isoformat() == "2024-01-30"
