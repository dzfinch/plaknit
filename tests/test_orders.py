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

    def search(self, **kwargs):
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


def test_orders_clear_fraction_treats_clear_percent_one_as_one_percent():
    clear_fraction = orders._clear_fraction({"clear_percent": 1})
    assert clear_fraction == 0.01


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


def test_find_replacement_items_applies_quality_filters():
    plan_entry = {
        "filters": {
            "item_type": "PSScene",
            "month_start": "2024-01-01",
            "month_end": "2024-01-31",
            "min_clear_fraction": 0.5,
            "require_ground_control": True,
            "max_shadow_fraction": 0.05,
            "quality_weight": 0.5,
        }
    }
    items = [
        _FakeStacItem(
            "good",
            {
                "clear_percent": 96,
                "cloud_cover": 0.04,
                "ground_control": True,
                "shadow_percent": 1,
            },
        ),
        _FakeStacItem(
            "bad-shadow",
            {
                "clear_percent": 99,
                "cloud_cover": 0.01,
                "ground_control": True,
                "shadow_percent": 40,
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

    assert [item["id"] for item in replacements] == ["good"]


def test_find_replacement_items_reads_stac_extension_aliases():
    plan_entry = {
        "filters": {
            "item_type": "PSScene",
            "month_start": "2024-01-01",
            "month_end": "2024-01-31",
            "min_clear_fraction": 0.5,
            "require_ground_control": True,
            "quality_category": "standard",
            "publishing_stage": "finalized",
            "max_shadow_fraction": 0.05,
            "max_view_angle": 10.0,
            "min_visible_confidence": 0.8,
            "min_clear_confidence": 0.8,
        }
    }
    items = [
        _FakeStacItem(
            "alias-good",
            {
                "pl:clear_percent": 97,
                "eo:cloud_cover": 0.03,
                "view:sun_elevation": 52,
                "view:sun_azimuth": 140,
                "datetime": "2024-01-11T00:00:00Z",
                "view:off_nadir": 3.8,
                "pl:visible_confidence_percent": 92,
                "pl:clear_confidence_percent": 93,
                "pl:ground_control": True,
                "pl:quality_category": "standard",
                "pl:publishing_stage": "finalized",
                "pl:shadow_percent": 1,
                "pl:snow_ice_percent": 0,
                "pl:heavy_haze_percent": 1,
                "pl:anomalous_pixels": 0,
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

    assert [item["id"] for item in replacements] == ["alias-good"]
    props = replacements[0]["properties"]
    assert props["cloud_cover"] == 0.03
    assert props["sun_elevation"] == 52
    assert props["sun_azimuth"] == 140
    assert props["view_angle"] == 3.8
    assert props["acquired"] == "2024-01-11T00:00:00Z"
