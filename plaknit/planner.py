"""Planning utilities and CLI for PlanetScope composites."""

from __future__ import annotations

import argparse
import json
import logging
import os
from base64 import b64encode
from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pystac_client import Client
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

from .geometry import load_aoi_geometry, reproject_geometry
from .orders import submit_orders_for_plan

PLANET_STAC_URL = "https://api.planet.com/x/data/"
PLAN_LOGGER_NAME = "plaknit.plan"
TILE_PROJECTION = "EPSG:6933"
DEPTH_TARGET_FRACTION = 0.95


@dataclass
class _TileState:
    covered: bool = False
    clear_obs: float = 0.0


@dataclass
class _Candidate:
    item_id: str
    collection_id: Optional[str]
    properties: Dict[str, Any]
    clear_fraction: float
    tile_indexes: List[int]
    selected: bool = False


def _get_logger() -> logging.Logger:
    return logging.getLogger(PLAN_LOGGER_NAME)


def _require_api_key() -> str:
    api_key = os.environ.get("PL_API_KEY")
    if not api_key:
        raise EnvironmentError("PL_API_KEY environment variable is required.")
    return api_key


def _open_planet_stac_client(api_key: str) -> Client:
    token = b64encode(f"{api_key}:".encode("utf-8")).decode("ascii")
    headers = {"Authorization": f"Basic {token}"}
    return Client.open(PLANET_STAC_URL, headers=headers)


def _geometry_to_geojson(geometry: BaseGeometry) -> Dict[str, Any]:
    return mapping(geometry)


def _iterate_months(start: date, end: date) -> Iterable[tuple[str, date, date]]:
    current = date(start.year, start.month, 1)
    while current <= end:
        last_day = monthrange(current.year, current.month)[1]
        month_end = date(current.year, current.month, last_day)
        month_start = current
        yield (
            current.strftime("%Y-%m"),
            max(month_start, start),
            min(month_end, end),
        )
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def _generate_tiles(geometry: BaseGeometry, tile_size: int) -> List[BaseGeometry]:
    minx, miny, maxx, maxy = geometry.bounds
    tiles: List[BaseGeometry] = []
    x = minx
    while x < maxx:
        x_max = min(x + tile_size, maxx)
        y = miny
        while y < maxy:
            y_max = min(y + tile_size, maxy)
            tile = box(x, y, x_max, y_max)
            if tile.intersects(geometry):
                tiles.append(tile)
            y = y_max
        x = x_max

    if not tiles:  # ensure at least one tile for tiny AOIs
        tiles.append(geometry.envelope)

    return tiles


def _get_property(properties: Dict[str, Any], keys: Sequence[str]) -> Any:
    """Return the first non-null property value for any key in keys."""
    for key in keys:
        if key in properties and properties[key] not in (None, ""):
            return properties[key]
    return None


def _clear_fraction(properties: Dict[str, Any]) -> float:
    clear_value = _get_property(
        properties,
        [
            "clear_percent",
            "pl:clear_percent",
            "pl_clear_percent",
            "clear_fraction",
            "pl:clear_fraction",
        ],
    )
    if clear_value is not None:
        try:
            clear_float = float(clear_value)
            if clear_float > 1:
                clear_float /= 100.0
            return max(0.0, min(1.0, clear_float))
        except (ValueError, TypeError):
            pass

    cloud_value = _get_property(
        properties,
        [
            "cloud_cover",
            "pl:cloud_cover",
            "pl_cloud_cover",
            "cloud_percent",
            "pl:cloud_percent",
            "pl_cloud_percent",
        ],
    )
    if cloud_value is not None:
        try:
            cloud_fraction = float(cloud_value)
            if cloud_fraction > 1:
                cloud_fraction /= 100.0
            return max(0.0, min(1.0, 1.0 - cloud_fraction))
        except (ValueError, TypeError):
            pass

    logger = _get_logger()
    logger.warning(
        "Scene %s is missing clear/cloud metadata; assuming fully clear.",
        properties.get("id", "unknown"),
    )
    return 1.0


def _tiles_for_scene(
    scene_geom: BaseGeometry, prepared_tiles: List[Any]
) -> List[int]:  # type: ignore[type-arg]
    indexes: List[int] = []
    for idx, tile in enumerate(prepared_tiles):
        if tile.intersects(scene_geom):
            indexes.append(idx)
    return indexes


def _score_candidate(
    candidate: _Candidate, tile_states: List[_TileState], min_clear_obs: float
) -> float:
    score = 0.0
    for idx in candidate.tile_indexes:
        tile_state = tile_states[idx]
        needs_coverage = not tile_state.covered
        needs_depth = tile_state.clear_obs < min_clear_obs
        if needs_coverage or needs_depth:
            deficit = max(0.0, min_clear_obs - tile_state.clear_obs)
            weight = 1.0 + deficit
            score += weight * candidate.clear_fraction
    return score


def _apply_candidate(candidate: _Candidate, tile_states: List[_TileState]) -> None:
    for idx in candidate.tile_indexes:
        tile_state = tile_states[idx]
        tile_state.covered = True
        tile_state.clear_obs += candidate.clear_fraction


def _coverage_fraction(tile_states: List[_TileState]) -> float:
    if not tile_states:
        return 1.0
    covered = sum(1 for tile in tile_states if tile.covered)
    return covered / len(tile_states)


def _depth_fraction(tile_states: List[_TileState], min_clear_obs: float) -> float:
    if not tile_states:
        return 1.0
    sufficient = sum(1 for tile in tile_states if tile.clear_obs >= min_clear_obs)
    return sufficient / len(tile_states)


def plan_monthly_composites(
    aoi_path: str,
    start_date: str,
    end_date: str,
    item_type: str = "PSScene",
    collection: str | None = None,
    cloud_max: float = 0.1,
    sun_elevation_min: float = 35.0,
    coverage_target: float = 0.98,
    min_clear_fraction: float = 0.8,
    min_clear_obs: float = 3.0,
    month_grouping: str = "calendar",
    limit: int | None = None,
    tile_size_m: int = 1000,
) -> dict:
    """
    Plan monthly PlanetScope composites over an AOI.
    """

    if month_grouping != "calendar":
        raise ValueError("Only 'calendar' month grouping is supported.")
    if tile_size_m <= 0:
        raise ValueError("tile_size_m must be positive.")

    logger = _get_logger()
    api_key = _require_api_key()
    client = _open_planet_stac_client(api_key)

    aoi_geom, aoi_crs = load_aoi_geometry(aoi_path)
    aoi_crs = aoi_crs or "EPSG:4326"
    aoi_wgs84 = reproject_geometry(aoi_geom, aoi_crs, "EPSG:4326")
    aoi_projected = reproject_geometry(aoi_geom, aoi_crs, TILE_PROJECTION)
    tiles_projected = _generate_tiles(aoi_projected, tile_size_m)
    prepared_tiles = [prep(tile) for tile in tiles_projected]
    logger.info(
        "AOI tiling: %d tiles at %d m resolution (%s).",
        len(tiles_projected),
        tile_size_m,
        TILE_PROJECTION,
    )

    collections_param = [collection] if collection else [item_type]
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except ValueError as exc:  # pragma: no cover - validated inputs
        raise ValueError("Dates must be formatted as YYYY-MM-DD.") from exc
    if start > end:
        raise ValueError("start_date must be before or equal to end_date.")

    plan: dict[str, dict[str, Any]] = {}
    for month_id, month_start, month_end in _iterate_months(start, end):
        month_plan = _plan_single_month(
            month_id=month_id,
            month_start=month_start,
            month_end=month_end,
            client=client,
            aoi_wgs84=aoi_wgs84,
            tiles_projected=tiles_projected,
            prepared_tiles=prepared_tiles,
            collections=collections_param,
            cloud_max=cloud_max,
            sun_elevation_min=sun_elevation_min,
            coverage_target=coverage_target,
            min_clear_fraction=min_clear_fraction,
            min_clear_obs=min_clear_obs,
            limit=limit,
        )
        month_plan["tile_size_m"] = tile_size_m
        month_plan["coverage_target"] = coverage_target
        month_plan["min_clear_obs"] = min_clear_obs
        plan[month_id] = month_plan

    return plan


def _plan_single_month(
    *,
    month_id: str,
    month_start: date,
    month_end: date,
    client: Client,
    aoi_wgs84: BaseGeometry,
    tiles_projected: List[BaseGeometry],
    prepared_tiles: List[Any],  # type: ignore[type-arg]
    collections: List[str],
    cloud_max: float,
    sun_elevation_min: float,
    coverage_target: float,
    min_clear_fraction: float,
    min_clear_obs: float,
    limit: int | None,
) -> Dict[str, Any]:
    logger = _get_logger()
    datetime_range = f"{month_start.isoformat()}/{month_end.isoformat()}"
    query: Dict[str, Any] = {
        "sun_elevation": {"gte": sun_elevation_min},
    }
    if cloud_max is not None:
        query["cloud_cover"] = {"lte": cloud_max}

    logger.info("Searching Planet STAC for %s (%s).", month_id, datetime_range)
    search = client.search(
        collections=collections,
        datetime=datetime_range,
        intersects=_geometry_to_geojson(aoi_wgs84),
        query=query,
        max_items=limit,
    )
    items = list(search.items())
    candidate_count = len(items)

    tile_states = [_TileState() for _ in tiles_projected]
    candidates: List[_Candidate] = []

    for item in items:
        properties = dict(item.properties)
        properties["id"] = item.id
        cloud_value = _get_property(
            properties,
            [
                "cloud_cover",
                "pl:cloud_cover",
                "pl_cloud_cover",
                "cloud_percent",
                "pl:cloud_percent",
                "pl_cloud_percent",
            ],
        )
        if cloud_value is not None and cloud_max is not None:
            try:
                cloud_fraction = float(cloud_value)
                if cloud_fraction > 1:
                    cloud_fraction /= 100.0
                if cloud_fraction > cloud_max:
                    continue
            except (ValueError, TypeError):
                pass

        sun_value = properties.get("sun_elevation")
        if sun_value is not None:
            try:
                if float(sun_value) < sun_elevation_min:
                    continue
            except (ValueError, TypeError):
                pass

        scene_geom = shape(item.geometry)
        if scene_geom.is_empty:
            continue

        scene_geom_projected = reproject_geometry(scene_geom, "EPSG:4326", TILE_PROJECTION)
        tile_indexes = _tiles_for_scene(scene_geom_projected, prepared_tiles)
        if not tile_indexes:
            continue

        clear_fraction = _clear_fraction(properties)
        if clear_fraction < min_clear_fraction:
            continue

        candidates.append(
            _Candidate(
                item_id=item.id,
                collection_id=item.collection_id,
                properties=properties,
                clear_fraction=clear_fraction,
                tile_indexes=tile_indexes,
            )
        )

    filtered_count = len(candidates)
    logger.info(
        "Month %s: %d candidates (%d after filters).",
        month_id,
        candidate_count,
        filtered_count,
    )

    selected: List[_Candidate] = []
    while True:
        coverage = _coverage_fraction(tile_states)
        depth_fraction = _depth_fraction(tile_states, min_clear_obs)
        if coverage >= coverage_target and depth_fraction >= DEPTH_TARGET_FRACTION:
            break

        best_candidate: Optional[_Candidate] = None
        best_score = 0.0
        for candidate in candidates:
            if candidate.selected:
                continue
            score = _score_candidate(candidate, tile_states, min_clear_obs)
            if score > best_score:
                best_candidate = candidate
                best_score = score

        if best_candidate is None or best_score <= 0:
            break

        best_candidate.selected = True
        _apply_candidate(best_candidate, tile_states)
        selected.append(best_candidate)

    coverage = _coverage_fraction(tile_states)
    depth_fraction = _depth_fraction(tile_states, min_clear_obs)
    if coverage < coverage_target:
        logger.warning(
            "Coverage target (%.2f) not met for %s (achieved %.3f).",
            coverage_target,
            month_id,
            coverage,
        )
    if depth_fraction < DEPTH_TARGET_FRACTION:
        logger.warning(
            "Clear observation depth target not met for %s (achieved %.3f).",
            month_id,
            depth_fraction,
        )

    item_entries = [
        {
            "id": candidate.item_id,
            "collection": candidate.collection_id,
            "clear_fraction": candidate.clear_fraction,
            "properties": {
                "cloud_cover": _get_property(
                    candidate.properties,
                    [
                        "cloud_cover",
                        "pl:cloud_cover",
                        "pl_cloud_cover",
                        "cloud_percent",
                        "pl:cloud_percent",
                        "pl_cloud_percent",
                    ],
                ),
                "clear_percent": _get_property(
                    candidate.properties,
                    [
                        "clear_percent",
                        "pl:clear_percent",
                        "pl_clear_percent",
                        "clear_fraction",
                        "pl:clear_fraction",
                    ],
                ),
                "sun_elevation": candidate.properties.get("sun_elevation"),
                "sun_azimuth": candidate.properties.get("sun_azimuth"),
                "acquired": candidate.properties.get("acquired"),
            },
        }
        for candidate in selected
    ]

    return {
        "items": item_entries,
        "aoi_coverage": coverage,
        "candidate_count": candidate_count,
        "filtered_count": filtered_count,
        "selected_count": len(selected),
        "tile_count": len(tile_states),
        "clear_depth_fraction": depth_fraction,
    }


def write_plan(plan: dict, path: str) -> None:
    """Write plan dict to JSON file (pretty-printed)."""
    with open(path, "w", encoding="utf-8") as dst:
        json.dump(plan, dst, indent=2)


def configure_planning_logger(verbosity: int) -> logging.Logger:
    """Configure logging similar to mosaic CLI."""
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


def build_plan_parser() -> argparse.ArgumentParser:
    """Create an argparse parser for the plan command."""
    parser = argparse.ArgumentParser(
        prog="plaknit plan",
        description="Plan monthly PlanetScope composites and optionally submit orders.",
    )
    parser.add_argument("--aoi", "-a", required=True, help="AOI file (.geojson/.json/.shp/.gpkg).")
    parser.add_argument("--start", "-s", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", "-e", required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument("--item-type", default="PSScene", help="Planet item type (default: PSScene).")
    parser.add_argument("--collection", help="Optional collection ID for the STAC search.")
    parser.add_argument("--cloud-max", type=float, default=0.1, help="Maximum cloud fraction (0-1).")
    parser.add_argument(
        "--sun-elev-min",
        type=float,
        default=35.0,
        help="Minimum sun elevation in degrees (default: 35).",
    )
    parser.add_argument(
        "--coverage-target",
        type=float,
        default=0.98,
        help="Target AOI coverage fraction (default: 0.98).",
    )
    parser.add_argument(
        "--min-clear-fraction",
        type=float,
        default=0.8,
        help="Minimum clear fraction per scene (default: 0.8).",
    )
    parser.add_argument(
        "--min-clear-obs",
        type=float,
        default=3.0,
        help="Target expected clear observations per tile (default: 3).",
    )
    parser.add_argument(
        "--tile-size-m",
        type=int,
        default=1000,
        help="Tile size in meters for the AOI grid (default: 1000).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of STAC items per month (passes through to the STAC search).",
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
        default="none",
        help="Harmonize target sensor (sentinel2) or disable (none).",
    )
    parser.add_argument("--order", action="store_true", help="Submit Planet orders using the plan.")
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
        "--out",
        help="Optional path to write the plan JSON.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v info, -vv debug).",
    )
    return parser


def parse_plan_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_plan_parser()
    return parser.parse_args(argv)


def _harmonize_display(value: str) -> str:
    if value == "sentinel2":
        return "Sentinel-2"
    return "-"


def _order_id_for_month(order_results: Dict[str, Dict[str, Any]], month: str) -> str:
    if month not in order_results:
        return "-"
    order_id = order_results[month].get("order_id")
    return order_id or "-"


def _print_summary(
    plan: Dict[str, Dict[str, Any]],
    order_results: Dict[str, Dict[str, Any]],
    sr_bands: int,
    harmonize: str,
) -> None:
    header = (
        "Month     Candidates  Filtered  Selected  Coverage  MinClearObs  SR-bands  Harmonize     Order ID"
    )
    divider = "-" * len(header)
    print(header)
    print(divider)
    for month in sorted(plan.keys()):
        entry = plan[month]
        coverage = entry.get("aoi_coverage", 0.0)
        min_clear_obs = entry.get("min_clear_obs", 0.0)
        candidate_count = entry.get("candidate_count", 0)
        filtered_count = entry.get("filtered_count", 0)
        selected_count = entry.get("selected_count", 0)
        print(
            f"{month:8}  {candidate_count:10d}  {filtered_count:8d}  {selected_count:8d}  "
            f"{coverage:8.3f}  {min_clear_obs:11.1f}  {sr_bands:8d}  "
            f"{_harmonize_display(harmonize):11}  {_order_id_for_month(order_results, month)}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_plan_args(argv)
    logger = configure_planning_logger(args.verbose)
    _require_api_key()  # fail fast if missing
    harmonize = None if args.harmonize_to == "none" else args.harmonize_to

    plan = plan_monthly_composites(
        aoi_path=args.aoi,
        start_date=args.start,
        end_date=args.end,
        item_type=args.item_type,
        collection=args.collection,
        cloud_max=args.cloud_max,
        sun_elevation_min=args.sun_elev_min,
        coverage_target=args.coverage_target,
        min_clear_fraction=args.min_clear_fraction,
        min_clear_obs=args.min_clear_obs,
        month_grouping="calendar",
        limit=args.limit,
        tile_size_m=args.tile_size_m,
    )

    order_results: Dict[str, Dict[str, Any]] = {}
    if args.order:
        logger.info("Submitting Planet orders for %d months.", len(plan))
        order_results = submit_orders_for_plan(
            plan=plan,
            aoi_path=args.aoi,
            sr_bands=args.sr_bands,
            harmonize_to=harmonize,
            order_prefix=args.order_prefix,
            archive_type=args.archive_type,
        )

    if args.out:
        write_plan(plan, args.out)
        logger.info("Plan written to %s", args.out)

    _print_summary(plan, order_results, args.sr_bands, args.harmonize_to)
    return 0


__all__ = [
    "plan_monthly_composites",
    "write_plan",
    "configure_planning_logger",
    "build_plan_parser",
    "parse_plan_args",
    "main",
]
