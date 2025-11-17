"""Planet Orders API helpers for plaknit."""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Dict

from shapely.geometry import mapping

from .geometry import load_aoi_geometry, reproject_geometry

ORDER_LOGGER_NAME = "plaknit.plan"


def _get_logger() -> logging.Logger:
    return logging.getLogger(ORDER_LOGGER_NAME)


def _require_api_key() -> str:
    api_key = os.environ.get("PL_API_KEY")
    if not api_key:
        raise EnvironmentError("PL_API_KEY environment variable is required for orders.")
    return api_key


def _get_orders_client(api_key: str):
    from planet import OrdersClient, api

    auth = api.Auth.from_key(api_key)
    return OrdersClient(auth=auth)


def _bundle_for_sr_bands(sr_bands: int) -> str:
    if sr_bands == 4:
        return "analytic_sr_udm2"
    if sr_bands == 8:
        return "analytic_8b_sr_udm2"
    raise ValueError("sr_bands must be 4 or 8.")


def _clip_geojson(aoi_path: str) -> Dict[str, Any]:
    geometry, crs = load_aoi_geometry(aoi_path)
    geom_wgs84 = reproject_geometry(geometry, crs, "EPSG:4326")
    return mapping(geom_wgs84)


def submit_orders_for_plan(
    plan: dict,
    aoi_path: str,
    sr_bands: int = 4,
    harmonize_to: str | None = "sentinel2",
    order_prefix: str = "plaknit_plan",
    archive_type: str = "zip",
) -> dict:
    """
    Submit Planet Orders API requests for each month in the plan.
    """

    api_key = _require_api_key()
    client = _get_orders_client(api_key)
    logger = _get_logger()

    clip_geojson = _clip_geojson(aoi_path)
    bundle = _bundle_for_sr_bands(sr_bands)
    harmonize_normalized = harmonize_to.lower() if harmonize_to else None

    tools = [{"clip": {"aoi": clip_geojson}}]
    if harmonize_normalized == "sentinel2":
        tools.append({"harmonize": {"target_sensor": "Sentinel-2"}})

    results: dict[str, dict[str, Any]] = {}
    for month in sorted(plan.keys()):
        entry = plan[month]
        items = entry.get("items", [])
        if not items:
            logger.info("Skipping order for %s: no selected items.", month)
            results[month] = {"order_id": None, "item_ids": []}
            continue

        item_ids = [item["id"] for item in items if item.get("id")]
        if not item_ids:
            logger.info("Skipping order for %s: missing item IDs.", month)
            results[month] = {"order_id": None, "item_ids": []}
            continue

        order_tools = copy.deepcopy(tools)
        order_request = {
            "name": f"{order_prefix}_{month}",
            "products": [
                {
                    "item_ids": item_ids,
                    "item_type": "PSScene",
                    "product_bundle": bundle,
                }
            ],
            "tools": order_tools,
            "delivery": {
                "archive_type": archive_type,
            },
        }

        try:
            order = client.create_order(order_request)
        except Exception as exc:  # pragma: no cover - exercised via mocks
            logger.error("Failed to submit order for %s: %s", month, exc)
            results[month] = {"order_id": None, "item_ids": item_ids}
            continue

        order_id = None
        if isinstance(order, dict):
            order_id = order.get("id")
        else:
            order_id = getattr(order, "id", None)

        logger.info("Submitted order for %s: %s", month, order_id)
        results[month] = {"order_id": order_id, "item_ids": item_ids}

    return results


__all__ = ["submit_orders_for_plan"]
