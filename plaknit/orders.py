"""Planet Orders API helpers for plaknit."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Sequence

from shapely.geometry import mapping

from .geometry import load_aoi_geometry, reproject_geometry

ORDER_LOGGER_NAME = "plaknit.plan"


def _get_logger() -> logging.Logger:
    return logging.getLogger(ORDER_LOGGER_NAME)


def _require_api_key() -> str:
    api_key = os.environ.get("PL_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "PL_API_KEY environment variable is required for orders."
        )
    return api_key


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


def _configure_order_logger(verbosity: int) -> logging.Logger:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(message)s")
    else:
        root.setLevel(level)

    logger = _get_logger()
    logger.setLevel(level)
    return logger


def _load_plan_from_path(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as src:
        data = json.load(src)
    if not isinstance(data, dict):
        raise ValueError("Plan file must contain a JSON object.")
    return data


def _print_order_summary(results: Dict[str, Dict[str, Any]]) -> None:
    if not results:
        print("No orders submitted.")
        return
    header = "Month     Items  Order ID"
    divider = "-" * len(header)
    print(header)
    print(divider)
    for month in sorted(results.keys()):
        entry = results[month]
        item_count = len(entry.get("item_ids", []) or [])
        order_id = entry.get("order_id") or "-"
        print(f"{month:8}  {item_count:5d}  {order_id}")


def _parse_error_payload(error: Exception) -> Optional[Dict[str, Any]]:
    raw = str(error)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _extract_inaccessible_item_ids(error: Exception) -> List[str]:
    payload = _parse_error_payload(error)
    if not payload:
        return []

    field = payload.get("field", {})
    details = field.get("Details") or field.get("details") or []
    inaccessible: List[str] = []
    for detail in details:
        message = detail.get("message")
        if not message or "no access to assets" not in message:
            continue
        if "PSScene/" not in message:
            continue
        start = message.find("PSScene/") + len("PSScene/")
        end = message.find("/", start)
        item_id = message[start:end] if end != -1 else message[start:]
        item_id = item_id.strip()
        if item_id and item_id not in inaccessible:
            inaccessible.append(item_id)
    return inaccessible


@asynccontextmanager
async def _orders_client_context(api_key: str):
    from planet import Auth, Session

    auth = Auth.from_key(api_key)
    async with Session(auth=auth) as session:
        yield session.client("orders")


async def _submit_orders_async(
    plan: dict,
    aoi_path: str,
    sr_bands: int,
    harmonize_to: str | None,
    order_prefix: str,
    archive_type: str,
    api_key: str,
) -> dict:
    logger = _get_logger()
    clip_geojson = _clip_geojson(aoi_path)
    bundle = _bundle_for_sr_bands(sr_bands)
    harmonize_normalized = harmonize_to.lower() if harmonize_to else None

    tools = [{"clip": {"aoi": clip_geojson}}]
    if harmonize_normalized == "sentinel2":
        tools.append({"harmonize": {"target_sensor": "Sentinel-2"}})

    results: dict[str, dict[str, Any]] = {}
    async with _orders_client_context(api_key) as client:
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

            remaining_items = [item for item in items if item.get("id")]
            order_result: Optional[Any] = None

            while remaining_items:
                submit_item_ids = [item["id"] for item in remaining_items]
                order_tools = copy.deepcopy(tools)
                order_request = {
                    "name": f"{order_prefix}_{month}",
                    "products": [
                        {
                            "item_ids": submit_item_ids,
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
                    order_result = await client.create_order(order_request)
                except Exception as exc:  # pragma: no cover - exercised via mocks
                    inaccessible_ids = _extract_inaccessible_item_ids(exc)
                    if not inaccessible_ids:
                        logger.error("Failed to submit order for %s: %s", month, exc)
                        results[month] = {"order_id": None, "item_ids": submit_item_ids}
                        break

                    logger.warning(
                        "Removing %d inaccessible scene(s) for %s: %s",
                        len(inaccessible_ids),
                        month,
                        ", ".join(inaccessible_ids),
                    )
                    remaining_items = [
                        item
                        for item in remaining_items
                        if item["id"] not in inaccessible_ids
                    ]
                    if not remaining_items:
                        logger.error(
                            "Skipping order for %s: no accessible scenes remain.", month
                        )
                        results[month] = {"order_id": None, "item_ids": []}
                        break
                    continue
                else:
                    break

            if order_result is None:
                continue

            order_id = None
            if isinstance(order_result, dict):
                order_id = order_result.get("id")
            else:
                order_id = getattr(order_result, "id", None)

            logger.info("Submitted order for %s: %s", month, order_id)
            results[month] = {"order_id": order_id, "item_ids": submit_item_ids}

    return results


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
    return asyncio.run(
        _submit_orders_async(
            plan=plan,
            aoi_path=aoi_path,
            sr_bands=sr_bands,
            harmonize_to=harmonize_to,
            order_prefix=order_prefix,
            archive_type=archive_type,
            api_key=api_key,
        )
    )


def build_order_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plaknit order",
        description="Submit Planet orders for an existing plan JSON/GeoJSON file.",
    )
    parser.add_argument(
        "--plan", "-p", required=True, help="Path to a saved plan JSON/GeoJSON file."
    )
    parser.add_argument(
        "--aoi",
        "-a",
        required=True,
        help="AOI file used to clip orders (.geojson/.json/.shp/.gpkg).",
    )
    parser.add_argument(
        "--sr-bands",
        type=int,
        choices=(4, 8),
        default=4,
        help="Surface reflectance bundle: 4-band or 8-band (default: 4).",
    )
    parser.add_argument(
        "--harmonize-to",
        choices=("sentinel2", "none"),
        default="sentinel2",
        help="Harmonize target sensor (sentinel2) or disable (none).",
    )
    parser.add_argument(
        "--order-prefix",
        default="plaknit_plan",
        help="Prefix for Planet order names (default: plaknit_plan).",
    )
    parser.add_argument(
        "--archive-type",
        default="zip",
        help="Delivery archive type for orders (default: zip).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v info, -vv debug).",
    )
    return parser


def parse_order_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_order_parser()
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_order_args(argv)
    _configure_order_logger(args.verbose)
    plan = _load_plan_from_path(args.plan)
    harmonize = None if args.harmonize_to == "none" else args.harmonize_to

    results = submit_orders_for_plan(
        plan=plan,
        aoi_path=args.aoi,
        sr_bands=args.sr_bands,
        harmonize_to=harmonize,
        order_prefix=args.order_prefix,
        archive_type=args.archive_type,
    )
    _print_order_summary(results)
    return 0


__all__ = [
    "submit_orders_for_plan",
    "build_order_parser",
    "parse_order_args",
    "main",
]
