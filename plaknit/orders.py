"""Planet Orders API helpers for plaknit."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import warnings
from base64 import b64encode
from calendar import monthrange
from datetime import date
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Sequence

from pystac_client import Client
from shapely.geometry import mapping

from .geometry import load_aoi_geometry, reproject_geometry

ORDER_LOGGER_NAME = "plaknit.plan"
PLANET_STAC_URL = "https://api.planet.com/x/data/"
MAX_ITEMS_PER_ORDER = 100


def _get_logger() -> logging.Logger:
    return logging.getLogger(ORDER_LOGGER_NAME)


def _require_api_key() -> str:
    api_key = os.environ.get("PL_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "PL_API_KEY environment variable is required for orders."
        )
    return api_key


def _open_planet_stac_client(api_key: str) -> Client:
    warnings.filterwarnings(
        "ignore", message=".*Server does not conform to QUERY.*", category=UserWarning
    )
    token = b64encode(f"{api_key}:".encode("utf-8")).decode("ascii")
    headers = {"Authorization": f"Basic {token}"}
    return Client.open(PLANET_STAC_URL, headers=headers)


def _month_start_end(month: str, plan_entry: Dict[str, Any]) -> tuple[date, date]:
    filters = plan_entry.get("filters", {}) or {}
    try:
        start_str = filters.get("month_start")
        end_str = filters.get("month_end")
        if start_str and end_str:
            start = date.fromisoformat(start_str)
            end = date.fromisoformat(end_str)
            return start, end
    except Exception:
        pass
    year, month_num = month.split("-")
    last_day = monthrange(int(year), int(month_num))[1]
    start = date(int(year), int(month_num), 1)
    end = date(int(year), int(month_num), last_day)
    return start, end


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


def _clear_fraction(properties: Dict[str, Any]) -> Optional[float]:
    clear_value = properties.get("clear_percent") or properties.get("pl:clear_percent")
    if clear_value is None:
        clear_value = properties.get("clear_fraction") or properties.get(
            "pl:clear_fraction"
        )
    if clear_value is not None:
        try:
            clear_float = float(clear_value)
            if clear_float > 1:
                clear_float /= 100.0
            return max(0.0, min(1.0, clear_float))
        except (ValueError, TypeError):
            pass

    cloud_value = properties.get("cloud_cover") or properties.get("pl:cloud_cover")
    if cloud_value is None:
        cloud_value = properties.get("cloud_percent") or properties.get(
            "pl:cloud_percent"
        )
    if cloud_value is not None:
        try:
            cloud_fraction = float(cloud_value)
            if cloud_fraction > 1:
                cloud_fraction /= 100.0
            return max(0.0, min(1.0, 1.0 - cloud_fraction))
        except (ValueError, TypeError):
            pass

    return None


def _find_replacement_items(
    *,
    stac_client: Client,
    plan_entry: Dict[str, Any],
    month: str,
    aoi_geojson: Dict[str, Any],
    desired_count: int,
    exclude_ids: set[str],
) -> List[Dict[str, Any]]:
    if desired_count <= 0:
        return []

    filters = plan_entry.get("filters", {}) or {}
    item_type = filters.get("item_type") or "PSScene"
    collection = filters.get("collection")
    imagery_type = filters.get("imagery_type")
    instrument_types = filters.get("instrument_types")
    cloud_max = filters.get("cloud_max")
    sun_elevation_min = filters.get("sun_elevation_min")
    min_clear_fraction = filters.get("min_clear_fraction", 0.0) or 0.0
    limit = filters.get("limit")

    item_collections = [collection] if collection else [item_type]
    query: Dict[str, Any] = {}
    if sun_elevation_min is not None:
        query["sun_elevation"] = {"gte": sun_elevation_min}
    if cloud_max is not None:
        query["cloud_cover"] = {"lte": cloud_max}
    if imagery_type:
        query["pl:imagery_type"] = {"eq": imagery_type}
    if instrument_types:
        unique_instruments = []
        for inst in instrument_types:
            if inst not in unique_instruments:
                unique_instruments.append(inst)
        if len(unique_instruments) == 1:
            query["pl:instrument"] = {"eq": unique_instruments[0]}
        else:
            query["pl:instrument"] = {"in": unique_instruments}

    month_start, month_end = _month_start_end(month, plan_entry)
    datetime_range = f"{month_start.isoformat()}/{month_end.isoformat()}"

    search = stac_client.search(
        collections=item_collections,
        datetime=datetime_range,
        intersects=aoi_geojson,
        query=query,
        max_items=limit,
    )
    candidates: List[tuple[float, Any]] = []
    for item in search.items():
        if item.id in exclude_ids:
            continue
        properties = dict(item.properties)
        properties["id"] = item.id
        clear_fraction = _clear_fraction(properties)
        if clear_fraction is None or clear_fraction < min_clear_fraction:
            continue
        candidates.append((clear_fraction, item))

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    replacements: List[Dict[str, Any]] = []
    for clear_fraction, item in candidates[:desired_count]:
        replacements.append(
            {
                "id": item.id,
                "collection": item.collection_id or collection or "PSScene",
                "clear_fraction": clear_fraction,
                "properties": {
                    "cloud_cover": item.properties.get("cloud_cover"),
                    "clear_percent": item.properties.get("clear_percent"),
                    "sun_elevation": item.properties.get("sun_elevation"),
                    "sun_azimuth": item.properties.get("sun_azimuth"),
                    "acquired": item.properties.get("acquired"),
                },
            }
        )

    return replacements


async def _submit_orders_async(
    plan: dict,
    aoi_path: str,
    sr_bands: int,
    harmonize_to: str | None,
    order_prefix: str,
    archive_type: str,
    single_archive: bool,
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
    stac_client = _open_planet_stac_client(api_key)
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
            dropped_ids: set[str] = set()
            order_index = 1
            results.setdefault(
                month, {"order_id": None, "order_ids": [], "item_ids": []}
            )

            while remaining_items:
                batch = remaining_items[:MAX_ITEMS_PER_ORDER]
                remaining_items = remaining_items[MAX_ITEMS_PER_ORDER:]
                working_batch = batch
                order_result: Optional[Any] = None
                order_name = (
                    f"{order_prefix}_{month}"
                    if order_index == 1
                    else f"{order_prefix}_{month}_{order_index}"
                )

                while working_batch:
                    submit_item_ids = [item["id"] for item in working_batch]
                    order_tools = copy.deepcopy(tools)
                    delivery: Dict[str, Any] = {
                        "archive_type": archive_type,
                        "archive_filename": order_name,
                    }
                    if single_archive:
                        delivery["single_archive"] = True

                    order_request = {
                        "name": order_name,
                        "products": [
                            {
                                "item_ids": submit_item_ids,
                                "item_type": "PSScene",
                                "product_bundle": bundle,
                            }
                        ],
                        "tools": order_tools,
                        "delivery": delivery,
                    }

                    try:
                        order_result = await client.create_order(order_request)
                    except Exception as exc:  # pragma: no cover - exercised via mocks
                        inaccessible_ids = _extract_inaccessible_item_ids(exc)
                        if not inaccessible_ids:
                            logger.error(
                                "Failed to submit order for %s: %s", month, exc
                            )
                            results[month] = {
                                "order_id": None,
                                "order_ids": [],
                                "item_ids": submit_item_ids,
                            }
                            working_batch = []
                            remaining_items = []
                            break

                        logger.warning(
                            "Removing %d inaccessible scene(s) for %s: %s",
                            len(inaccessible_ids),
                            month,
                            ", ".join(inaccessible_ids),
                        )
                        dropped_ids.update(inaccessible_ids)
                        working_batch = [
                            item
                            for item in working_batch
                            if item["id"] not in inaccessible_ids
                        ]
                        desired_replacements = len(inaccessible_ids)
                        replacements = _find_replacement_items(
                            stac_client=stac_client,
                            plan_entry=entry,
                            month=month,
                            aoi_geojson=clip_geojson,
                            desired_count=desired_replacements,
                            exclude_ids=set(submit_item_ids) | dropped_ids,
                        )
                        if replacements:
                            logger.info(
                                "Adding %d replacement scene(s) for %s.",
                                len(replacements),
                                month,
                            )
                            working_batch.extend(replacements)
                            if len(working_batch) > MAX_ITEMS_PER_ORDER:
                                # keep batch size within limit
                                overflow = working_batch[MAX_ITEMS_PER_ORDER:]
                                working_batch = working_batch[:MAX_ITEMS_PER_ORDER]
                                remaining_items = overflow + remaining_items
                            continue
                        if not working_batch:
                            logger.error(
                                "Skipping order for %s: no accessible scenes remain.",
                                month,
                            )
                            break
                        logger.warning(
                            "Proceeding with %d scene(s) after removals.",
                            len(working_batch),
                        )
                        continue
                    else:
                        order_id = None
                        if isinstance(order_result, dict):
                            order_id = order_result.get("id")
                        else:
                            order_id = getattr(order_result, "id", None)
                        logger.info(
                            "Submitted order for %s (%d scenes): %s",
                            order_name,
                            len(submit_item_ids),
                            order_id,
                        )
                        results[month]["order_ids"].append(order_id)
                        if results[month]["order_id"] is None:
                            results[month]["order_id"] = order_id
                        results[month]["item_ids"].extend(submit_item_ids)
                        working_batch = []
                        order_index += 1

            if not results.get(month):
                results[month] = {"order_id": None, "order_ids": [], "item_ids": []}

    return results


def submit_orders_for_plan(
    plan: dict,
    aoi_path: str,
    sr_bands: int = 4,
    harmonize_to: str | None = "sentinel2",
    order_prefix: str = "plaknit_plan",
    archive_type: str = "zip",
    single_archive: bool = True,
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
            single_archive=single_archive,
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
        "--single-archive",
        dest="single_archive",
        action="store_true",
        default=True,
        help="Deliver each order as a single archive file (default: enabled).",
    )
    parser.add_argument(
        "--no-single-archive",
        dest="single_archive",
        action="store_false",
        help="Deliver separate files per scene instead of one archive.",
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
        single_archive=args.single_archive,
    )
    _print_order_summary(results)
    return 0


__all__ = [
    "submit_orders_for_plan",
    "build_order_parser",
    "parse_order_args",
    "main",
]
