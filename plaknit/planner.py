"""Planning utilities and CLI for PlanetScope composites."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import warnings
from base64 import b64encode
from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except Exception:  # pragma: no cover - optional dependency
    Progress = None  # type: ignore

from pystac_client import Client
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

from .geometry import (
    geometry_vertex_count,
    load_aoi_geometry,
    reproject_geometry,
    simplify_geometry_to_vertex_limit,
)
from .orders import submit_orders_for_plan

PLANET_STAC_URL = "https://api.planet.com/x/data/"
PLAN_LOGGER_NAME = "plaknit.plan"
TILE_PROJECTION = "EPSG:6933"
DEPTH_TARGET_FRACTION = 0.95
PLANET_MAX_ROI_VERTICES = 1500
PLANETSCOPE_IMAGERY_TYPE = "planetscope"
PLANETSCOPE_INSTRUMENT_IDS = ("PS2", "PS2.SD", "PSB.SD")


class _ProgressManager:
    """Thin wrapper around rich Progress that degrades gracefully when unavailable."""

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled and Progress is not None)
        self._progress: Optional[Progress] = None
        self._totals: dict[int, float] = {}

    def __enter__(self) -> "_ProgressManager":
        if self.enabled and Progress is not None:
            self._progress = Progress(
                BarColumn(),
                TextColumn("{task.description}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                refresh_per_second=5,
            )
            self._progress.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._progress:
            self._progress.stop()
        self._progress = None
        self._totals.clear()

    def add(self, description: str, total: Optional[int] = None) -> Optional[int]:
        if not self._progress:
            return None
        task_id = self._progress.add_task(description, total=total)
        if task_id is not None and total is not None:
            self._totals[int(task_id)] = float(total)
        return task_id

    def add_total(self, task_id: Optional[int], delta: int) -> None:
        if self._progress and task_id is not None and delta:
            current_total = self._totals.get(int(task_id), 0.0)
            new_total = current_total + delta
            self._totals[int(task_id)] = new_total
            self._progress.update(task_id, total=new_total)

    def advance(self, task_id: Optional[int], advance: float = 1.0) -> None:
        if self._progress and task_id is not None:
            self._progress.advance(task_id, advance)

    def complete(self, task_id: Optional[int]) -> None:
        if self._progress and task_id is not None:
            total = self._totals.get(int(task_id))
            if total is None:
                self._progress.update(task_id, completed=1)
            else:
                self._progress.update(task_id, total=total, completed=total)


@dataclass
class _TileState:
    covered: bool = False
    clear_obs: float = 0.0
    sun_elevation_sum: float = 0.0
    sun_elevation_weight: float = 0.0
    sun_azimuth_x: float = 0.0
    sun_azimuth_y: float = 0.0
    sun_azimuth_weight: float = 0.0


@dataclass
class _Candidate:
    item_id: str
    collection_id: Optional[str]
    properties: Dict[str, Any]
    clear_fraction: float
    tile_indexes: List[int]
    sun_azimuth: Optional[float] = None
    sun_elevation: Optional[float] = None
    selected: bool = False


def _get_logger() -> logging.Logger:
    return logging.getLogger(PLAN_LOGGER_NAME)


def _require_api_key() -> str:
    api_key = os.environ.get("PL_API_KEY")
    if not api_key:
        raise EnvironmentError("PL_API_KEY environment variable is required.")
    return api_key


def _open_planet_stac_client(api_key: str) -> Client:
    warnings.filterwarnings(
        "ignore", message=".*Server does not conform to QUERY.*", category=UserWarning
    )
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


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clear_fraction(properties: Dict[str, Any]) -> Optional[float]:
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
        "Scene %s is missing clear/cloud metadata; skipping.",
        properties.get("id", "unknown"),
    )
    return None


def _tiles_for_scene(
    scene_geom: BaseGeometry, prepared_tiles: List[Any]
) -> List[int]:  # type: ignore[type-arg]
    indexes: List[int] = []
    for idx, tile in enumerate(prepared_tiles):
        if tile.intersects(scene_geom):
            indexes.append(idx)
    return indexes


def _circular_difference_deg(value: float, reference: float) -> float:
    diff = (value - reference + 180.0) % 360.0 - 180.0
    return abs(diff)


def _mean_sun_elevation(tile_state: _TileState) -> Optional[float]:
    if tile_state.sun_elevation_weight <= 0:
        return None
    return tile_state.sun_elevation_sum / tile_state.sun_elevation_weight


def _mean_sun_azimuth(tile_state: _TileState) -> Optional[float]:
    if tile_state.sun_azimuth_weight <= 0:
        return None
    angle = math.degrees(math.atan2(tile_state.sun_azimuth_y, tile_state.sun_azimuth_x))
    if angle < 0:
        angle += 360.0
    return angle


def _lighting_similarity(
    tile_state: _TileState,
    sun_azimuth: Optional[float],
    sun_elevation: Optional[float],
    azimuth_sigma: float,
    elevation_sigma: float,
) -> float:
    similarity = 1.0
    mean_azimuth = _mean_sun_azimuth(tile_state)
    if (
        sun_azimuth is not None
        and mean_azimuth is not None
        and azimuth_sigma is not None
        and azimuth_sigma > 0
    ):
        diff = _circular_difference_deg(sun_azimuth, mean_azimuth)
        similarity *= math.exp(-((diff / azimuth_sigma) ** 2))
    mean_elevation = _mean_sun_elevation(tile_state)
    if (
        sun_elevation is not None
        and mean_elevation is not None
        and elevation_sigma is not None
        and elevation_sigma > 0
    ):
        diff = abs(sun_elevation - mean_elevation)
        similarity *= math.exp(-((diff / elevation_sigma) ** 2))
    return similarity


def _update_tile_lighting(
    tile_state: _TileState,
    sun_azimuth: Optional[float],
    sun_elevation: Optional[float],
    weight: float,
) -> None:
    if weight <= 0:
        return
    if sun_elevation is not None:
        tile_state.sun_elevation_sum += sun_elevation * weight
        tile_state.sun_elevation_weight += weight
    if sun_azimuth is not None:
        radians = math.radians(sun_azimuth % 360.0)
        tile_state.sun_azimuth_x += math.cos(radians) * weight
        tile_state.sun_azimuth_y += math.sin(radians) * weight
        tile_state.sun_azimuth_weight += weight


def _score_candidate(
    candidate: _Candidate,
    tile_states: List[_TileState],
    min_clear_obs: float,
    azimuth_sigma: float,
    elevation_sigma: float,
) -> float:
    score = 0.0
    for idx in candidate.tile_indexes:
        tile_state = tile_states[idx]
        needs_coverage = not tile_state.covered
        needs_depth = tile_state.clear_obs < min_clear_obs
        if needs_coverage or needs_depth:
            deficit = max(0.0, min_clear_obs - tile_state.clear_obs)
            weight = 1.0 + deficit
            similarity = _lighting_similarity(
                tile_state,
                candidate.sun_azimuth,
                candidate.sun_elevation,
                azimuth_sigma,
                elevation_sigma,
            )
            score += weight * candidate.clear_fraction * similarity
    return score


def _apply_candidate(candidate: _Candidate, tile_states: List[_TileState]) -> None:
    for idx in candidate.tile_indexes:
        tile_state = tile_states[idx]
        tile_state.covered = True
        tile_state.clear_obs += candidate.clear_fraction
        _update_tile_lighting(
            tile_state,
            candidate.sun_azimuth,
            candidate.sun_elevation,
            candidate.clear_fraction,
        )


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
    imagery_type: str | None = PLANETSCOPE_IMAGERY_TYPE,
    instrument_types: Sequence[str] | None = PLANETSCOPE_INSTRUMENT_IDS,
    cloud_max: float = 0.1,
    sun_elevation_min: float = 35.0,
    coverage_target: float = 0.98,
    min_clear_fraction: float = 0.8,
    min_clear_obs: float = 3.0,
    azimuth_sigma: float = 20.0,
    elevation_sigma: float = 10.0,
    month_grouping: str = "calendar",
    limit: int | None = None,
    tile_size_m: int = 1000,
    progress: Optional[_ProgressManager] = None,
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
    aoi_wgs84_vertices = geometry_vertex_count(aoi_wgs84)
    aoi_wgs84_query = simplify_geometry_to_vertex_limit(
        aoi_wgs84, PLANET_MAX_ROI_VERTICES
    )
    simplified_vertices = geometry_vertex_count(aoi_wgs84_query)
    if simplified_vertices < aoi_wgs84_vertices:
        logger.info(
            "AOI vertices reduced from %d to %d to comply with Planet ROI limits (<= %d).",
            aoi_wgs84_vertices,
            simplified_vertices,
            PLANET_MAX_ROI_VERTICES,
        )
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

    month_definitions = list(_iterate_months(start, end))
    progress_log = progress or _ProgressManager(
        enabled=logging.getLogger().isEnabledFor(logging.INFO)
    )
    plan: dict[str, dict[str, Any]] = {}
    try:
        progress_log.__enter__()
        filtering_task = (
            progress_log.add("Filtering scenes", total=0) if progress_log else None
        )
        optimize_task = (
            progress_log.add("Optimizing tiles", total=0) if progress_log else None
        )
        for month_id, month_start, month_end in month_definitions:
            month_plan = _plan_single_month(
                month_id=month_id,
                month_start=month_start,
                month_end=month_end,
                client=client,
                aoi_wgs84=aoi_wgs84_query,
                tiles_projected=tiles_projected,
                prepared_tiles=prepared_tiles,
                collections=collections_param,
                imagery_type=imagery_type,
                instrument_types=instrument_types,
                cloud_max=cloud_max,
                sun_elevation_min=sun_elevation_min,
                coverage_target=coverage_target,
                min_clear_fraction=min_clear_fraction,
                min_clear_obs=min_clear_obs,
                azimuth_sigma=azimuth_sigma,
                elevation_sigma=elevation_sigma,
                limit=limit,
                filtering_task=filtering_task,
                optimize_task=optimize_task,
                progress=progress_log,
            )
            month_plan["tile_size_m"] = tile_size_m
            month_plan["coverage_target"] = coverage_target
            month_plan["min_clear_obs"] = min_clear_obs
            month_plan["filters"] = {
                "item_type": item_type,
                "collection": collection,
                "imagery_type": imagery_type,
                "instrument_types": (
                    list(instrument_types) if instrument_types else None
                ),
                "cloud_max": cloud_max,
                "sun_elevation_min": sun_elevation_min,
                "min_clear_fraction": min_clear_fraction,
                "month_start": month_start.isoformat(),
                "month_end": month_end.isoformat(),
                "azimuth_sigma": azimuth_sigma,
                "elevation_sigma": elevation_sigma,
                "limit": limit,
            }
            plan[month_id] = month_plan
        if progress_log:
            progress_log.complete(filtering_task)
            progress_log.complete(optimize_task)
    finally:
        progress_log.__exit__(None, None, None)

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
    imagery_type: str | None,
    instrument_types: Sequence[str] | None,
    cloud_max: float,
    sun_elevation_min: float,
    coverage_target: float,
    min_clear_fraction: float,
    min_clear_obs: float,
    azimuth_sigma: float,
    elevation_sigma: float,
    limit: int | None,
    filtering_task: Optional[int],
    optimize_task: Optional[int],
    progress: Optional[_ProgressManager] = None,
) -> Dict[str, Any]:
    logger = _get_logger()
    datetime_range = f"{month_start.isoformat()}/{month_end.isoformat()}"
    query: Dict[str, Any] = {
        "sun_elevation": {"gte": sun_elevation_min},
    }
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

    logger.debug("Searching Planet STAC for %s (%s).", month_id, datetime_range)
    search = client.search(
        collections=collections,
        datetime=datetime_range,
        intersects=_geometry_to_geojson(aoi_wgs84),
        query=query,
        max_items=limit,
    )
    items = list(search.items())
    candidate_count = len(items)
    if progress:
        progress.add_total(filtering_task, candidate_count)

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

        scene_geom_projected = reproject_geometry(
            scene_geom, "EPSG:4326", TILE_PROJECTION
        )
        tile_indexes = _tiles_for_scene(scene_geom_projected, prepared_tiles)
        if not tile_indexes:
            continue

        clear_fraction = _clear_fraction(properties)
        if clear_fraction is None:
            continue
        if clear_fraction < min_clear_fraction:
            continue
        sun_azimuth = _float_or_none(properties.get("sun_azimuth"))
        sun_elevation = _float_or_none(properties.get("sun_elevation"))

        candidates.append(
            _Candidate(
                item_id=item.id,
                collection_id=item.collection_id,
                properties=properties,
                clear_fraction=clear_fraction,
                tile_indexes=tile_indexes,
                sun_azimuth=sun_azimuth,
                sun_elevation=sun_elevation,
            )
        )
        if progress:
            progress.advance(filtering_task)

    filtered_count = len(candidates)
    logger.debug(
        "Month %s: %d candidates (%d after filters).",
        month_id,
        candidate_count,
        filtered_count,
    )

    selected: List[_Candidate] = []
    if progress:
        progress.add_total(optimize_task, len(candidates))
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
            score = _score_candidate(
                candidate,
                tile_states,
                min_clear_obs,
                azimuth_sigma,
                elevation_sigma,
            )
            if score > best_score:
                best_candidate = candidate
                best_score = score

        if best_candidate is None or best_score <= 0:
            break

        best_candidate.selected = True
        _apply_candidate(best_candidate, tile_states)
        selected.append(best_candidate)
        if progress:
            progress.advance(optimize_task)

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
    parser.add_argument(
        "--aoi", "-a", required=True, help="AOI file (.geojson/.json/.shp/.gpkg)."
    )
    parser.add_argument("--start", "-s", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", "-e", required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--item-type", default="PSScene", help="Planet item type (default: PSScene)."
    )
    parser.add_argument(
        "--collection", help="Optional collection ID for the STAC search."
    )
    parser.add_argument(
        "--imagery-type",
        default=PLANETSCOPE_IMAGERY_TYPE,
        help="PlanetScope imagery type filter (default: planetscope; set to 'none' to disable).",
    )
    parser.add_argument(
        "--instrument-type",
        dest="instrument_types",
        action="append",
        help=(
            "Repeatable Dove instrument filter "
            "(required when multiple PlanetScope instruments exist; set to 'none' to disable)."
        ),
    )
    parser.add_argument(
        "--cloud-max", type=float, default=0.1, help="Maximum cloud fraction (0-1)."
    )
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
        "--azimuth-sigma",
        type=float,
        default=20.0,
        help="Gaussian sigma in degrees for sun azimuth similarity penalty (default: 20).",
    )
    parser.add_argument(
        "--elevation-sigma",
        type=float,
        default=10.0,
        help="Gaussian sigma in degrees for sun elevation similarity penalty (default: 10).",
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
    parser.add_argument(
        "--order", action="store_true", help="Submit Planet orders using the plan."
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
        help="Deliver each submitted order as a single archive (default: enabled).",
    )
    parser.add_argument(
        "--no-single-archive",
        dest="single_archive",
        action="store_false",
        help="Disable single-archive delivery and keep per-scene files.",
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


def _is_planetscope_request(args: argparse.Namespace) -> bool:
    item_type = (getattr(args, "item_type", "") or "").lower()
    imagery = (
        getattr(args, "imagery_type", PLANETSCOPE_IMAGERY_TYPE)
        or PLANETSCOPE_IMAGERY_TYPE
    )
    return item_type == "psscene" and imagery.lower() != "none"


def parse_plan_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_plan_parser()
    args = parser.parse_args(argv)
    if (
        _is_planetscope_request(args)
        and len(PLANETSCOPE_INSTRUMENT_IDS) > 1
        and not args.instrument_types
    ):
        parser.error(
            "Multiple PlanetScope instrument IDs are available "
            f"({', '.join(PLANETSCOPE_INSTRUMENT_IDS)}). Use --instrument-type to select one."
        )
    return args


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
    header = "Month     Candidates  Filtered  Selected  Coverage  MinClearObs  SR-bands  Harmonize     Order ID"
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
    imagery_filter = (
        None
        if args.imagery_type is None or args.imagery_type.lower() == "none"
        else args.imagery_type
    )
    if args.instrument_types:
        instrument_filters: Optional[Sequence[str]] = tuple(
            inst for inst in args.instrument_types if inst and inst.lower() != "none"
        )
        if not instrument_filters:
            instrument_filters = None
    elif _is_planetscope_request(args) and len(PLANETSCOPE_INSTRUMENT_IDS) == 1:
        instrument_filters = PLANETSCOPE_INSTRUMENT_IDS
    else:
        instrument_filters = None

    plan = plan_monthly_composites(
        aoi_path=args.aoi,
        start_date=args.start,
        end_date=args.end,
        item_type=args.item_type,
        collection=args.collection,
        imagery_type=imagery_filter,
        instrument_types=instrument_filters,
        cloud_max=args.cloud_max,
        sun_elevation_min=args.sun_elev_min,
        coverage_target=args.coverage_target,
        min_clear_fraction=args.min_clear_fraction,
        min_clear_obs=args.min_clear_obs,
        azimuth_sigma=args.azimuth_sigma,
        elevation_sigma=args.elevation_sigma,
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
            single_archive=args.single_archive,
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
