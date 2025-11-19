"""Tests for the Planet orders helper."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

from shapely.geometry import box

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


def test_order_cli_reads_plan_and_submits(monkeypatch, tmp_path):
    plan = {"2024-01": {"items": [{"id": "item-1"}]}}
    plan_path = tmp_path / "plan.geojson"
    plan_path.write_text(json.dumps(plan))

    captured: dict[str, Any] = {}

    def fake_submit(**kwargs):
        captured.update(kwargs)
        return {"2024-01": {"order_id": "order-xyz", "item_ids": ["item-1"]}}

    monkeypatch.setattr(orders, "submit_orders_for_plan", fake_submit)

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

    assert exit_code == 0
    assert captured["plan"] == plan
    assert captured["aoi_path"] == "aoi.geojson"
    assert captured["sr_bands"] == 8
    assert captured["harmonize_to"] is None
    assert captured["order_prefix"] == "demo"
    assert captured["archive_type"] == "tar"
    assert captured["single_archive"] is False
