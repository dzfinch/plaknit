"""Tests for the Planet orders helper."""

from __future__ import annotations

from unittest import mock

from shapely.geometry import box

from plaknit import orders


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
    monkeypatch.setattr(orders, "load_aoi_geometry", lambda path: (fake_geom, "EPSG:4326"))
    monkeypatch.setattr(orders, "reproject_geometry", lambda geom, src, dst: geom)

    orders_client = mock.MagicMock()
    orders_client.create_order.return_value = {"id": "order-abc"}
    monkeypatch.setattr(orders, "_get_orders_client", lambda key: orders_client)

    result = orders.submit_orders_for_plan(
        plan=plan,
        aoi_path="aoi.geojson",
        sr_bands=8,
        harmonize_to="sentinel2",
        order_prefix="plaknit_plan",
        archive_type="zip",
    )

    assert orders_client.create_order.call_count == 1
    request = orders_client.create_order.call_args[0][0]
    assert request["name"] == "plaknit_plan_2024-01"
    assert request["products"][0]["product_bundle"] == "analytic_8b_sr_udm2"
    assert request["delivery"]["archive_type"] == "zip"
    tools = request["tools"]
    assert any("clip" in tool for tool in tools)
    assert any(tool.get("harmonize", {}).get("target_sensor") == "Sentinel-2" for tool in tools)

    assert result["2024-01"]["order_id"] == "order-abc"
    assert result["2024-02"]["order_id"] is None

