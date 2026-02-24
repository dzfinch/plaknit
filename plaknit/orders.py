"""Planet Orders API helpers for plaknit."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import math
import os
import warnings
from base64 import b64encode
from calendar import monthrange
from datetime import date
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Sequence

from pystac_client import Client
from shapely.geometry import mapping

from .geometry import (
    geometry_vertex_count,
    load_aoi_geometry,
    reproject_geometry,
    simplify_geometry_to_vertex_limit,
)

ORDER_LOGGER_NAME = "plaknit.plan"
PLANET_STAC_URL = "https://api.planet.com/x/data/"
MAX_ITEMS_PER_ORDER = 500
CLEAR_FRACTION_KEYS = (
    "clear_percent",
    "pl:clear_percent",
    "pl_clear_percent",
    "clear_fraction",
    "pl:clear_fraction",
    "pl_clear_fraction",
)
CLEAR_PERCENT_KEYS = ("clear_percent", "pl:clear_percent", "pl_clear_percent")
CLEAR_FRACTION_VALUE_KEYS = (
    "clear_fraction",
    "pl:clear_fraction",
    "pl_clear_fraction",
)
CLOUD_COVER_KEYS = (
    "cloud_cover",
    "eo:cloud_cover",
    "eo_cloud_cover",
    "pl:cloud_cover",
    "pl_cloud_cover",
    "cloud_percent",
    "pl:cloud_percent",
    "pl_cloud_percent",
)
SUN_ELEVATION_KEYS = (
    "sun_elevation",
    "view:sun_elevation",
    "view_sun_elevation",
)
SUN_AZIMUTH_KEYS = ("sun_azimuth", "view:sun_azimuth", "view_sun_azimuth")
ACQUIRED_KEYS = (
    "acquired",
    "datetime",
    "pl:acquired",
    "pl_acquired",
    "pl:acquired_datetime",
    "pl_acquired_datetime",
)
VISIBLE_CONFIDENCE_KEYS = (
    "visible_confidence_percent",
    "pl:visible_confidence_percent",
    "pl_visible_confidence_percent",
)
CLEAR_CONFIDENCE_KEYS = (
    "clear_confidence_percent",
    "pl:clear_confidence_percent",
    "pl_clear_confidence_percent",
)
SHADOW_PERCENT_KEYS = ("shadow_percent", "pl:shadow_percent", "pl_shadow_percent")
SNOW_ICE_PERCENT_KEYS = (
    "snow_ice_percent",
    "pl:snow_ice_percent",
    "pl_snow_ice_percent",
)
HEAVY_HAZE_PERCENT_KEYS = (
    "heavy_haze_percent",
    "pl:heavy_haze_percent",
    "pl_heavy_haze_percent",
)
VIEW_ANGLE_KEYS = ("view_angle", "view:off_nadir", "view_off_nadir")
GROUND_CONTROL_KEYS = ("ground_control", "pl:ground_control", "pl_ground_control")
QUALITY_CATEGORY_KEYS = (
    "quality_category",
    "pl:quality_category",
    "pl_quality_category",
)
PUBLISHING_STAGE_KEYS = (
    "publishing_stage",
    "pl:publishing_stage",
    "pl_publishing_stage",
)
ANOMALOUS_PIXELS_KEYS = (
    "anomalous_pixels",
    "pl:anomalous_pixels",
    "pl_anomalous_pixels",
)
GSD_KEYS = ("gsd", "pixel_resolution", "pl:pixel_resolution", "pl_pixel_resolution")
INSTRUMENT_KEYS = ("instruments", "instrument", "pl:instrument", "pl_instrument")
CONSTELLATION_KEYS = ("constellation", "eo:constellation", "pl:constellation")
FOUR_BAND_ONLY_INSTRUMENTS = {"ps2", "ps2.sd"}


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
    vertex_limit = 1500
    current_vertices = geometry_vertex_count(geom_wgs84)
    if current_vertices > vertex_limit:
        geom_wgs84 = simplify_geometry_to_vertex_limit(geom_wgs84, vertex_limit)
        simplified_vertices = geometry_vertex_count(geom_wgs84)
        _get_logger().info(
            "Simplified AOI vertices from %d to %d.",
            current_vertices,
            simplified_vertices,
        )
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
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def _extract_inaccessible_item_ids(error: Exception) -> List[str]:
    payload = _parse_error_payload(error)
    if not payload:
        return []

    field = payload.get("field")
    details: List[Dict[str, Any]] = []
    if isinstance(field, dict):
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
    clear_fraction = _property_percent_fraction(properties, CLEAR_PERCENT_KEYS)
    if clear_fraction is None:
        clear_fraction = _property_fraction(properties, CLEAR_FRACTION_VALUE_KEYS)
    if clear_fraction is not None:
        return clear_fraction

    cloud_fraction = _property_fraction(properties, CLOUD_COVER_KEYS)
    if cloud_fraction is not None:
        return max(0.0, min(1.0, 1.0 - cloud_fraction))

    return None


def _get_property(properties: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in properties and properties[key] not in (None, ""):
            return properties[key]
    return None


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return None


def _matches_instrument_filter(
    properties: Dict[str, Any], instrument_types: Sequence[str]
) -> bool:
    normalized_filters = {
        instrument.strip().lower()
        for instrument in instrument_types
        if isinstance(instrument, str) and instrument.strip()
    }
    if not normalized_filters:
        return True

    instrument_value = _get_property(properties, INSTRUMENT_KEYS)
    if instrument_value is None:
        return True

    if isinstance(instrument_value, str):
        candidates = [instrument_value]
    elif isinstance(instrument_value, Sequence) and not isinstance(
        instrument_value, (str, bytes)
    ):
        candidates = [str(value) for value in instrument_value]
    else:
        candidates = [str(instrument_value)]

    for candidate in candidates:
        if candidate.strip().lower() in normalized_filters:
            return True
    return False


def _normalized_property_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [str(v) for v in value]
    else:
        values = [str(value)]
    normalized: List[str] = []
    for candidate in values:
        cleaned = candidate.strip().lower()
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def _normalized_instrument_values(value: Any) -> List[str]:
    return _normalized_property_values(value)


def _required_asset_keys_for_sr_bands(sr_bands: int) -> tuple[str, ...]:
    # Planet STAC sometimes exposes SR assets with/without the `ortho_` prefix.
    if sr_bands == 4:
        return ("ortho_analytic_4b_sr", "analytic_4b_sr", "analytic_sr")
    if sr_bands == 8:
        return ("ortho_analytic_8b_sr", "analytic_8b_sr")
    return ()


def _asset_keys_from_item(item: Any) -> set[str]:
    assets = getattr(item, "assets", None)
    if isinstance(assets, dict):
        return {str(key) for key in assets.keys()}
    return set()


def _lookup_item_metadata(
    *,
    stac_client: Client,
    item_ids: Sequence[str],
    collection: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    if not item_ids:
        return {}
    unique_ids = list(dict.fromkeys(item_ids))
    search_kwargs: Dict[str, Any] = {"ids": unique_ids, "max_items": len(unique_ids)}
    if collection:
        search_kwargs["collections"] = [collection]
    try:
        search = stac_client.search(**search_kwargs)
    except Exception as exc:
        _get_logger().warning(
            "Unable to query STAC metadata for preflight checks: %s", exc
        )
        return {}

    metadata: Dict[str, Dict[str, Any]] = {}
    for item in search.items():
        metadata[item.id] = {
            "properties": dict(getattr(item, "properties", {}) or {}),
            "asset_keys": _asset_keys_from_item(item),
        }
    return metadata


def _normalized_instrument_filter(filters: Dict[str, Any]) -> tuple[str, ...]:
    instrument_types = filters.get("instrument_types")
    if not instrument_types:
        return ()
    if isinstance(instrument_types, str):
        instrument_types = [instrument_types]
    normalized: List[str] = []
    for value in instrument_types:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if cleaned and cleaned.lower() != "none" and cleaned not in normalized:
            normalized.append(cleaned)
    return tuple(normalized)


def _preflight_order_items(
    *,
    stac_client: Client,
    plan_entry: Dict[str, Any],
    items: List[Dict[str, Any]],
    sr_bands: int,
    metadata_cache: Dict[str, Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
    filters = plan_entry.get("filters", {}) or {}
    instrument_types = _normalized_instrument_filter(filters)
    if not instrument_types:
        return items, {}

    required_asset_keys = _required_asset_keys_for_sr_bands(sr_bands)
    collection = filters.get("collection") or filters.get("item_type") or "PSScene"
    ids_to_lookup = [item["id"] for item in items if item["id"] not in metadata_cache]
    fetched_metadata = _lookup_item_metadata(
        stac_client=stac_client,
        item_ids=ids_to_lookup,
        collection=collection,
    )
    metadata_cache.update(fetched_metadata)

    resolved: Dict[str, tuple[Dict[str, Any], set[str]]] = {}
    expected_constellations: set[str] = set()
    for item in items:
        item_id = item["id"]
        properties = dict(item.get("properties", {}) or {})
        asset_keys: set[str] = set()
        direct_assets = item.get("assets")
        if isinstance(direct_assets, dict):
            asset_keys.update(str(key) for key in direct_assets.keys())

        cached = metadata_cache.get(item_id, {})
        cached_properties = cached.get("properties")
        if isinstance(cached_properties, dict):
            for key, value in cached_properties.items():
                if key not in properties or properties[key] in (None, ""):
                    properties[key] = value
        cached_assets = cached.get("asset_keys")
        if isinstance(cached_assets, set):
            asset_keys.update(str(key) for key in cached_assets)

        resolved[item_id] = (properties, asset_keys)

        instrument_value = _get_property(properties, INSTRUMENT_KEYS)
        if instrument_value is None:
            continue
        if not _matches_instrument_filter(properties, instrument_types):
            continue
        expected_constellations.update(
            _normalized_property_values(_get_property(properties, CONSTELLATION_KEYS))
        )

    kept: List[Dict[str, Any]] = []
    dropped: Dict[str, str] = {}
    for item in items:
        item_id = item["id"]
        properties, asset_keys = resolved.get(item_id, ({}, set()))

        instrument_value = _get_property(properties, INSTRUMENT_KEYS)
        if instrument_value is None:
            constellation_values = _normalized_property_values(
                _get_property(properties, CONSTELLATION_KEYS)
            )
            if (
                expected_constellations
                and constellation_values
                and expected_constellations.intersection(constellation_values)
            ):
                pass
            else:
                dropped[item_id] = "missing instrument metadata"
                continue
        if not _matches_instrument_filter(properties, instrument_types):
            dropped[item_id] = f"instrument not in requested filter: {instrument_types}"
            continue

        normalized_instruments = _normalized_instrument_values(instrument_value)
        if sr_bands == 8 and any(
            instrument in FOUR_BAND_ONLY_INSTRUMENTS
            for instrument in normalized_instruments
        ):
            dropped[item_id] = "4-band-only instrument cannot satisfy 8-band order"
            continue

        if required_asset_keys and asset_keys and asset_keys.isdisjoint(
            required_asset_keys
        ):
            expected = ", ".join(required_asset_keys)
            observed = ", ".join(sorted(asset_keys)[:6])
            if len(asset_keys) > 6:
                observed += ", ..."
            _get_logger().debug(
                "Asset preflight mismatch for %s (expected one of [%s], observed [%s]); allowing submit.",
                item_id,
                expected,
                observed,
            )

        kept.append(item)

    return kept, dropped


def _normalized_fraction(value: Any) -> Optional[float]:
    parsed = _float_or_none(value)
    if parsed is None:
        return None
    if parsed > 1:
        parsed /= 100.0
    return max(0.0, min(1.0, parsed))


def _property_fraction(
    properties: Dict[str, Any], keys: Sequence[str]
) -> Optional[float]:
    return _normalized_fraction(_get_property(properties, keys))


def _normalized_percent(value: Any) -> Optional[float]:
    parsed = _float_or_none(value)
    if parsed is None:
        return None
    if parsed >= 1:
        parsed /= 100.0
    return max(0.0, min(1.0, parsed))


def _property_percent_fraction(
    properties: Dict[str, Any], keys: Sequence[str]
) -> Optional[float]:
    return _normalized_percent(_get_property(properties, keys))


def _passes_quality_filters(
    properties: Dict[str, Any],
    *,
    require_ground_control: bool,
    quality_category: Optional[str],
    publishing_stage: Optional[str],
    max_anomalous_pixels: Optional[float],
    max_shadow_fraction: Optional[float],
    max_snow_ice_fraction: Optional[float],
    max_heavy_haze_fraction: Optional[float],
    min_visible_confidence: Optional[float],
    min_clear_confidence: Optional[float],
    max_view_angle: Optional[float],
) -> bool:
    if require_ground_control:
        ground_control = _bool_or_none(_get_property(properties, GROUND_CONTROL_KEYS))
        if ground_control is not True:
            return False

    if quality_category:
        value = _get_property(properties, QUALITY_CATEGORY_KEYS)
        if (
            not isinstance(value, str)
            or value.strip().lower() != quality_category.lower()
        ):
            return False

    if publishing_stage:
        value = _get_property(properties, PUBLISHING_STAGE_KEYS)
        if (
            not isinstance(value, str)
            or value.strip().lower() != publishing_stage.lower()
        ):
            return False

    if max_anomalous_pixels is not None:
        anomalous = _float_or_none(_get_property(properties, ANOMALOUS_PIXELS_KEYS))
        if anomalous is None or anomalous > max_anomalous_pixels:
            return False

    if max_view_angle is not None:
        view_angle = _float_or_none(_get_property(properties, VIEW_ANGLE_KEYS))
        if view_angle is None or abs(view_angle) > max_view_angle:
            return False

    shadow_fraction = _property_percent_fraction(properties, SHADOW_PERCENT_KEYS)
    if max_shadow_fraction is not None and (
        shadow_fraction is None or shadow_fraction > max_shadow_fraction
    ):
        return False

    snow_ice_fraction = _property_percent_fraction(properties, SNOW_ICE_PERCENT_KEYS)
    if max_snow_ice_fraction is not None and (
        snow_ice_fraction is None or snow_ice_fraction > max_snow_ice_fraction
    ):
        return False

    heavy_haze_fraction = _property_percent_fraction(
        properties, HEAVY_HAZE_PERCENT_KEYS
    )
    if max_heavy_haze_fraction is not None and (
        heavy_haze_fraction is None or heavy_haze_fraction > max_heavy_haze_fraction
    ):
        return False

    visible_confidence = _property_percent_fraction(properties, VISIBLE_CONFIDENCE_KEYS)
    if min_visible_confidence is not None and (
        visible_confidence is None or visible_confidence < min_visible_confidence
    ):
        return False

    clear_confidence = _property_percent_fraction(properties, CLEAR_CONFIDENCE_KEYS)
    if min_clear_confidence is not None and (
        clear_confidence is None or clear_confidence < min_clear_confidence
    ):
        return False

    return True


def _quality_score(properties: Dict[str, Any]) -> float:
    metrics: List[tuple[float, float]] = []

    visible_confidence = _property_percent_fraction(properties, VISIBLE_CONFIDENCE_KEYS)
    if visible_confidence is not None:
        metrics.append((0.2, visible_confidence))

    clear_confidence = _property_percent_fraction(properties, CLEAR_CONFIDENCE_KEYS)
    if clear_confidence is not None:
        metrics.append((0.2, clear_confidence))

    shadow_fraction = _property_percent_fraction(properties, SHADOW_PERCENT_KEYS)
    if shadow_fraction is not None:
        metrics.append((0.15, max(0.0, min(1.0, 1.0 - shadow_fraction))))

    snow_ice_fraction = _property_percent_fraction(properties, SNOW_ICE_PERCENT_KEYS)
    if snow_ice_fraction is not None:
        metrics.append((0.15, max(0.0, min(1.0, 1.0 - snow_ice_fraction))))

    heavy_haze_fraction = _property_percent_fraction(
        properties, HEAVY_HAZE_PERCENT_KEYS
    )
    if heavy_haze_fraction is not None:
        metrics.append((0.1, max(0.0, min(1.0, 1.0 - heavy_haze_fraction))))

    view_angle = _float_or_none(_get_property(properties, VIEW_ANGLE_KEYS))
    if view_angle is not None:
        view_score = math.exp(-((abs(view_angle) / 15.0) ** 2))
        metrics.append((0.1, view_score))

    gsd = _float_or_none(_get_property(properties, GSD_KEYS))
    if gsd is not None and gsd > 0:
        gsd_score = math.exp(-((max(0.0, gsd - 3.0) / 1.5) ** 2))
        metrics.append((0.1, gsd_score))

    if not metrics:
        return 0.5
    total_weight = sum(weight for weight, _ in metrics)
    return sum(weight * value for weight, value in metrics) / total_weight


def _scene_property_snapshot(properties: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "instrument": _get_property(properties, INSTRUMENT_KEYS),
        "constellation": _get_property(properties, CONSTELLATION_KEYS),
        "cloud_cover": _get_property(properties, CLOUD_COVER_KEYS),
        "clear_percent": _get_property(properties, CLEAR_FRACTION_KEYS),
        "sun_elevation": _get_property(properties, SUN_ELEVATION_KEYS),
        "sun_azimuth": _get_property(properties, SUN_AZIMUTH_KEYS),
        "acquired": _get_property(properties, ACQUIRED_KEYS),
        "visible_confidence_percent": _get_property(
            properties, VISIBLE_CONFIDENCE_KEYS
        ),
        "clear_confidence_percent": _get_property(properties, CLEAR_CONFIDENCE_KEYS),
        "shadow_percent": _get_property(properties, SHADOW_PERCENT_KEYS),
        "snow_ice_percent": _get_property(properties, SNOW_ICE_PERCENT_KEYS),
        "heavy_haze_percent": _get_property(properties, HEAVY_HAZE_PERCENT_KEYS),
        "view_angle": _get_property(properties, VIEW_ANGLE_KEYS),
        "ground_control": _get_property(properties, GROUND_CONTROL_KEYS),
        "quality_category": _get_property(properties, QUALITY_CATEGORY_KEYS),
        "publishing_stage": _get_property(properties, PUBLISHING_STAGE_KEYS),
        "anomalous_pixels": _get_property(properties, ANOMALOUS_PIXELS_KEYS),
        "gsd": _get_property(properties, GSD_KEYS),
    }


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
    require_ground_control = bool(filters.get("require_ground_control", False))
    quality_category = filters.get("quality_category")
    publishing_stage = filters.get("publishing_stage")
    max_anomalous_pixels = filters.get("max_anomalous_pixels")
    max_shadow_fraction = _normalized_fraction(filters.get("max_shadow_fraction"))
    max_snow_ice_fraction = _normalized_fraction(filters.get("max_snow_ice_fraction"))
    max_heavy_haze_fraction = _normalized_fraction(
        filters.get("max_heavy_haze_fraction")
    )
    min_visible_confidence = _normalized_fraction(filters.get("min_visible_confidence"))
    min_clear_confidence = _normalized_fraction(filters.get("min_clear_confidence"))
    max_view_angle = _float_or_none(filters.get("max_view_angle"))
    quality_weight = _float_or_none(filters.get("quality_weight"))
    if quality_weight is None:
        quality_weight = 0.35
    quality_weight = max(0.0, min(1.0, quality_weight))
    limit = filters.get("limit")
    if isinstance(quality_category, str) and quality_category.lower() == "none":
        quality_category = None
    if isinstance(publishing_stage, str) and publishing_stage.lower() == "none":
        publishing_stage = None

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
    candidates: List[tuple[float, float, Any]] = []
    for item in search.items():
        if item.id in exclude_ids:
            continue
        properties = dict(item.properties)
        properties["id"] = item.id
        if instrument_types:
            if _get_property(properties, INSTRUMENT_KEYS) is None:
                continue
            if not _matches_instrument_filter(properties, instrument_types):
                continue
        clear_fraction = _clear_fraction(properties)
        if clear_fraction is None or clear_fraction < min_clear_fraction:
            continue
        if not _passes_quality_filters(
            properties,
            require_ground_control=require_ground_control,
            quality_category=quality_category,
            publishing_stage=publishing_stage,
            max_anomalous_pixels=_float_or_none(max_anomalous_pixels),
            max_shadow_fraction=max_shadow_fraction,
            max_snow_ice_fraction=max_snow_ice_fraction,
            max_heavy_haze_fraction=max_heavy_haze_fraction,
            min_visible_confidence=min_visible_confidence,
            min_clear_confidence=min_clear_confidence,
            max_view_angle=max_view_angle,
        ):
            continue
        quality_score = _quality_score(properties)
        quality_multiplier = max(
            0.0, 1.0 + quality_weight * (quality_score - 0.5) * 2.0
        )
        combined_score = clear_fraction * quality_multiplier
        candidates.append((combined_score, clear_fraction, item))

    candidates.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)
    replacements: List[Dict[str, Any]] = []
    for _, clear_fraction, item in candidates[:desired_count]:
        replacements.append(
            {
                "id": item.id,
                "collection": item.collection_id or collection or "PSScene",
                "clear_fraction": clear_fraction,
                "properties": _scene_property_snapshot(dict(item.properties)),
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
            ordered_ids: set[str] = set()
            order_index = 1
            results.setdefault(
                month, {"order_id": None, "order_ids": [], "item_ids": []}
            )

            while remaining_items:
                remaining_items = [
                    item
                    for item in remaining_items
                    if item["id"] not in ordered_ids and item["id"] not in dropped_ids
                ]
                if not remaining_items:
                    break
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
                    deduped_batch = []
                    seen_ids: set[str] = set()
                    for item in working_batch:
                        item_id = item["id"]
                        if (
                            item_id in seen_ids
                            or item_id in ordered_ids
                            or item_id in dropped_ids
                        ):
                            continue
                        seen_ids.add(item_id)
                        deduped_batch.append(item)
                    working_batch = deduped_batch
                    if not working_batch:
                        break
                    submit_item_ids = [item["id"] for item in working_batch]
                    order_tools = copy.deepcopy(tools)
                    archive_type_normalized = (
                        archive_type.lower() if archive_type else None
                    )
                    delivery: Dict[str, Any] = {}
                    if archive_type_normalized:
                        delivery["archive_type"] = archive_type_normalized
                        delivery["single_archive"] = bool(single_archive)
                        if archive_type_normalized == "zip":
                            archive_filename = order_name
                            if not order_name.lower().endswith(".zip"):
                                archive_filename = f"{order_name}.zip"
                            delivery["archive_filename"] = archive_filename

                    order_request = {
                        "name": order_name,
                        "source_type": "scenes",
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
                        remaining_ids = {item["id"] for item in remaining_items}
                        replacements = _find_replacement_items(
                            stac_client=stac_client,
                            plan_entry=entry,
                            month=month,
                            aoi_geojson=clip_geojson,
                            desired_count=desired_replacements,
                            exclude_ids=set(submit_item_ids)
                            | dropped_ids
                            | ordered_ids
                            | remaining_ids,
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
                        ordered_ids.update(submit_item_ids)
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
