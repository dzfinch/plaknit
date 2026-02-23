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
from datetime import date, datetime, timedelta
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
CLOUD_COVER_KEYS = (
    "eo:cloud_cover",
)
CLEAR_FRACTION_KEYS = (
    "pl:clear_percent",
)
SUN_ELEVATION_KEYS = ("view:sun_elevation",)
SUN_AZIMUTH_KEYS = ("view:sun_azimuth",)
ACQUIRED_KEYS = ("datetime",)
VIEW_ANGLE_KEYS = ("view:off_nadir",)
GROUND_CONTROL_KEYS = ("pl:ground_control",)
QUALITY_CATEGORY_KEYS = ("pl:quality_category",)
PUBLISHING_STAGE_KEYS = ("pl:publishing_stage",)
GSD_KEYS = ("gsd",)
INSTRUMENT_KEYS = ("instruments",)
STRIP_ID_KEYS = ("pl:strip_id",)

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
    off_nadir: Optional[float] = None
    acquired_dt: Optional[datetime] = None
    strip_id: Optional[str] = None
    quality_score: float = 1.0
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


def _window_key(window_start: date, window_end: date) -> str:
    if window_start == window_end:
        return window_start.isoformat()
    return f"{window_start.isoformat()}_to_{window_end.isoformat()}"


def _iterate_windows(
    start: date,
    end: date,
    *,
    grouping: str,
    window_days: Optional[int] = None,
) -> Iterable[tuple[str, date, date]]:
    if grouping == "calendar":
        yield from _iterate_months(start, end)
        return
    if grouping == "single":
        yield (_window_key(start, end), start, end)
        return
    if grouping == "fixed":
        if window_days is None or window_days <= 0:
            raise ValueError(
                "window_days must be a positive integer for fixed grouping."
            )
        current = start
        while current <= end:
            window_end = min(end, current + timedelta(days=window_days - 1))
            yield (_window_key(current, window_end), current, window_end)
            current = window_end + timedelta(days=1)
        return
    raise ValueError("grouping must be one of: calendar, single, fixed.")


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


def _normalize_instrument_filters(
    instrument_types: Sequence[str] | None,
) -> List[str]:
    if not instrument_types:
        return []
    unique: List[str] = []
    for instrument in instrument_types:
        if not isinstance(instrument, str):
            continue
        normalized = instrument.strip()
        if not normalized or normalized.lower() == "none":
            continue
        if normalized not in unique:
            unique.append(normalized)
    return unique


def _scene_instruments(properties: Dict[str, Any]) -> List[str]:
    value = _get_property(properties, INSTRUMENT_KEYS)
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, (list, tuple)):
        instruments: List[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            normalized = item.strip()
            if normalized and normalized not in instruments:
                instruments.append(normalized)
        return instruments
    return []


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _normalized_strip_id(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    strip = value.strip()
    if not strip:
        return None
    return strip


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


def _normalize_fraction_threshold(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return _normalized_fraction(value)


def _clear_fraction(properties: Dict[str, Any]) -> Optional[float]:
    clear_fraction = _property_fraction(properties, CLEAR_FRACTION_KEYS)
    if clear_fraction is not None:
        return clear_fraction

    cloud_fraction = _property_fraction(properties, CLOUD_COVER_KEYS)
    if cloud_fraction is not None:
        return max(0.0, min(1.0, 1.0 - cloud_fraction))

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
    quality_weight: float,
) -> float:
    quality_multiplier = max(
        0.0, 1.0 + quality_weight * (candidate.quality_score - 0.5) * 2.0
    )
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
            score += weight * candidate.clear_fraction * similarity * quality_multiplier
    return score


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

    if max_view_angle is not None:
        view_angle = _float_or_none(_get_property(properties, VIEW_ANGLE_KEYS))
        if view_angle is None or abs(view_angle) > max_view_angle:
            return False

    # Planet STAC does not reliably expose UDM-derived confidence/haze/shadow
    # and anomalous fields, so these filters are intentionally ignored.

    return True


def _quality_score(properties: Dict[str, Any]) -> float:
    metrics: List[tuple[float, float]] = []

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


def _select_legacy(
    *,
    candidates: List[_Candidate],
    tile_states: List[_TileState],
    coverage_target: float,
    min_clear_obs: float,
    azimuth_sigma: float,
    elevation_sigma: float,
    quality_weight: float,
    optimize_task: Optional[int],
    progress: Optional[_ProgressManager],
) -> tuple[List[_Candidate], str]:
    selected: List[_Candidate] = []
    while True:
        coverage = _coverage_fraction(tile_states)
        depth_fraction = _depth_fraction(tile_states, min_clear_obs)
        if coverage >= coverage_target and depth_fraction >= DEPTH_TARGET_FRACTION:
            return selected, "coverage_target_met"

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
                quality_weight,
            )
            if score > best_score:
                best_candidate = candidate
                best_score = score

        if best_candidate is None or best_score <= 0:
            return selected, "no_positive_gain"

        best_candidate.selected = True
        _apply_candidate(best_candidate, tile_states)
        selected.append(best_candidate)
        if progress:
            progress.advance(optimize_task)


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
    require_ground_control: bool = False,
    quality_category: Optional[str] = None,
    publishing_stage: Optional[str] = None,
    max_anomalous_pixels: Optional[float] = None,
    max_shadow_fraction: Optional[float] = None,
    max_snow_ice_fraction: Optional[float] = None,
    max_heavy_haze_fraction: Optional[float] = None,
    min_visible_confidence: Optional[float] = None,
    min_clear_confidence: Optional[float] = None,
    max_view_angle: Optional[float] = None,
    quality_weight: float = 0.35,
    month_grouping: str = "calendar",
    window_days: Optional[int] = None,
    limit: int | None = None,
    tile_size_m: int = 1000,
    progress: Optional[_ProgressManager] = None,
) -> dict:
    """
    Plan PlanetScope composites over an AOI with configurable time windows.
    """

    month_grouping = (month_grouping or "calendar").lower()
    if tile_size_m <= 0:
        raise ValueError("tile_size_m must be positive.")
    quality_weight = max(0.0, min(1.0, float(quality_weight)))
    max_shadow_fraction = _normalize_fraction_threshold(max_shadow_fraction)
    max_snow_ice_fraction = _normalize_fraction_threshold(max_snow_ice_fraction)
    max_heavy_haze_fraction = _normalize_fraction_threshold(max_heavy_haze_fraction)
    min_visible_confidence = _normalize_fraction_threshold(min_visible_confidence)
    min_clear_confidence = _normalize_fraction_threshold(min_clear_confidence)
    if quality_category is not None and quality_category.lower() == "none":
        quality_category = None
    if publishing_stage is not None and publishing_stage.lower() == "none":
        publishing_stage = None
    instrument_types = _normalize_instrument_filters(instrument_types)

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

    window_definitions = list(
        _iterate_windows(
            start,
            end,
            grouping=month_grouping,
            window_days=window_days,
        )
    )
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
        for month_id, month_start, month_end in window_definitions:
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
                require_ground_control=require_ground_control,
                quality_category=quality_category,
                publishing_stage=publishing_stage,
                max_anomalous_pixels=max_anomalous_pixels,
                max_shadow_fraction=max_shadow_fraction,
                max_snow_ice_fraction=max_snow_ice_fraction,
                max_heavy_haze_fraction=max_heavy_haze_fraction,
                min_visible_confidence=min_visible_confidence,
                min_clear_confidence=min_clear_confidence,
                max_view_angle=max_view_angle,
                quality_weight=quality_weight,
                limit=limit,
                filtering_task=filtering_task,
                optimize_task=optimize_task,
                progress=progress_log,
            )
            month_plan["tile_size_m"] = tile_size_m
            month_plan["coverage_target"] = coverage_target
            month_plan["min_clear_obs"] = min_clear_obs
            filters: Dict[str, Any] = {
                "item_type": item_type,
                "cloud_max": cloud_max,
                "sun_elevation_min": sun_elevation_min,
                "min_clear_fraction": min_clear_fraction,
                "start_date": month_start.isoformat(),
                "end_date": month_end.isoformat(),
            }
            if collection:
                filters["collection"] = collection
            if imagery_type:
                filters["imagery_type"] = imagery_type
            if instrument_types:
                filters["instrument_types"] = list(instrument_types)
            if require_ground_control:
                filters["require_ground_control"] = True
            if quality_category:
                filters["quality_category"] = quality_category
            if publishing_stage:
                filters["publishing_stage"] = publishing_stage
            if max_view_angle is not None:
                filters["max_view_angle"] = max_view_angle
            if limit is not None:
                filters["limit"] = limit
            month_plan["filters"] = filters
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
    quality_weight: float,
    limit: int | None,
    filtering_task: Optional[int],
    optimize_task: Optional[int],
    progress: Optional[_ProgressManager] = None,
) -> Dict[str, Any]:
    logger = _get_logger()
    datetime_range = f"{month_start.isoformat()}/{month_end.isoformat()}"
    requested_instruments = _normalize_instrument_filters(instrument_types)
    requested_instrument_set = {instrument.lower() for instrument in requested_instruments}
    query: Dict[str, Any] = {
        "view:sun_elevation": {"gte": sun_elevation_min},
    }
    if cloud_max is not None:
        query["eo:cloud_cover"] = {"lte": cloud_max}
    if imagery_type:
        query["pl:imagery_type"] = {"eq": imagery_type}
    if requested_instruments:
        if len(requested_instruments) == 1:
            query["instruments"] = {"in": requested_instruments}
        else:
            query["instruments"] = {"in": requested_instruments}

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

    has_instrument_metadata = False
    if requested_instrument_set:
        for item in items:
            if _scene_instruments(dict(item.properties)):
                has_instrument_metadata = True
                break
        if len(requested_instrument_set) == 1 and not has_instrument_metadata:
            logger.warning(
                "No instrument metadata found in STAC items for %s; "
                "falling back to STAC query-only instrument filtering.",
                month_id,
            )
    require_instrument_metadata = (
        len(requested_instrument_set) == 1 and has_instrument_metadata
    )

    for item in items:
        properties = dict(item.properties)
        properties["id"] = item.id
        item_instruments = _scene_instruments(properties)
        if requested_instrument_set:
            if item_instruments:
                if not any(
                    inst.lower() in requested_instrument_set
                    for inst in item_instruments
                ):
                    continue
            elif require_instrument_metadata:
                continue
        cloud_value = _get_property(properties, CLOUD_COVER_KEYS)
        if cloud_value is not None and cloud_max is not None:
            try:
                cloud_fraction = float(cloud_value)
                if cloud_fraction > 1:
                    cloud_fraction /= 100.0
                if cloud_fraction > cloud_max:
                    continue
            except (ValueError, TypeError):
                pass

        sun_value = _get_property(properties, SUN_ELEVATION_KEYS)
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
        if not _passes_quality_filters(
            properties,
            require_ground_control=require_ground_control,
            quality_category=quality_category,
            publishing_stage=publishing_stage,
            max_anomalous_pixels=max_anomalous_pixels,
            max_shadow_fraction=max_shadow_fraction,
            max_snow_ice_fraction=max_snow_ice_fraction,
            max_heavy_haze_fraction=max_heavy_haze_fraction,
            min_visible_confidence=min_visible_confidence,
            min_clear_confidence=min_clear_confidence,
            max_view_angle=max_view_angle,
        ):
            continue
        sun_azimuth = _float_or_none(_get_property(properties, SUN_AZIMUTH_KEYS))
        sun_elevation = _float_or_none(_get_property(properties, SUN_ELEVATION_KEYS))
        quality_score = _quality_score(properties)

        candidates.append(
            _Candidate(
                item_id=item.id,
                collection_id=item.collection_id,
                properties=properties,
                clear_fraction=clear_fraction,
                tile_indexes=tile_indexes,
                sun_azimuth=sun_azimuth,
                sun_elevation=sun_elevation,
                off_nadir=_float_or_none(_get_property(properties, VIEW_ANGLE_KEYS)),
                acquired_dt=_parse_datetime(_get_property(properties, ACQUIRED_KEYS)),
                strip_id=_normalized_strip_id(_get_property(properties, STRIP_ID_KEYS)),
                quality_score=quality_score,
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
    selected, stopped_reason = _select_legacy(
        candidates=candidates,
        tile_states=tile_states,
        coverage_target=coverage_target,
        min_clear_obs=min_clear_obs,
        azimuth_sigma=azimuth_sigma,
        elevation_sigma=elevation_sigma,
        quality_weight=quality_weight,
        optimize_task=optimize_task,
        progress=progress,
    )
    selection_diagnostics: Dict[str, Any] = {
        "selection_policy": "legacy",
        "policy_version": "legacy_v1",
        "stopped_reason": stopped_reason,
    }

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
                "eo:cloud_cover": _get_property(candidate.properties, CLOUD_COVER_KEYS),
                "pl:clear_percent": _get_property(
                    candidate.properties, CLEAR_FRACTION_KEYS
                ),
                "view:sun_elevation": _get_property(
                    candidate.properties, SUN_ELEVATION_KEYS
                ),
                "view:sun_azimuth": _get_property(
                    candidate.properties, SUN_AZIMUTH_KEYS
                ),
                "datetime": _get_property(candidate.properties, ACQUIRED_KEYS),
                "view:off_nadir": _get_property(candidate.properties, VIEW_ANGLE_KEYS),
                "pl:ground_control": _get_property(
                    candidate.properties, GROUND_CONTROL_KEYS
                ),
                "instruments": _get_property(candidate.properties, INSTRUMENT_KEYS),
                "pl:quality_category": _get_property(
                    candidate.properties, QUALITY_CATEGORY_KEYS
                ),
                "pl:publishing_stage": _get_property(
                    candidate.properties, PUBLISHING_STAGE_KEYS
                ),
                "gsd": _get_property(candidate.properties, GSD_KEYS),
            },
        }
        for candidate in selected
    ]

    output = {
        "items": item_entries,
        "aoi_coverage": coverage,
        "candidate_count": candidate_count,
        "filtered_count": filtered_count,
        "selected_count": len(selected),
        "tile_count": len(tile_states),
        "clear_depth_fraction": depth_fraction,
    }
    output.update(selection_diagnostics)
    return output


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
        description="Plan PlanetScope composites over flexible date windows and optionally submit orders.",
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
        "--require-ground-control",
        action="store_true",
        help="Require ground_control=true metadata for candidate scenes.",
    )
    parser.add_argument(
        "--quality-category",
        default="none",
        help="Require a specific quality_category value (default: none).",
    )
    parser.add_argument(
        "--publishing-stage",
        default="none",
        help="Require a specific publishing_stage value (default: none).",
    )
    parser.add_argument(
        "--max-anomalous-pixels",
        type=float,
        help="Maximum anomalous_pixels allowed per scene (for example 0).",
    )
    parser.add_argument(
        "--max-shadow",
        type=float,
        help="Maximum shadow fraction (0-1 or 0-100).",
    )
    parser.add_argument(
        "--max-snow-ice",
        type=float,
        help="Maximum snow/ice fraction (0-1 or 0-100).",
    )
    parser.add_argument(
        "--max-heavy-haze",
        type=float,
        help="Maximum heavy haze fraction (0-1 or 0-100).",
    )
    parser.add_argument(
        "--min-visible-confidence",
        type=float,
        help="Minimum visible confidence fraction (0-1 or 0-100).",
    )
    parser.add_argument(
        "--min-clear-confidence",
        type=float,
        help="Minimum clear confidence fraction (0-1 or 0-100).",
    )
    parser.add_argument(
        "--max-view-angle",
        type=float,
        help="Maximum absolute view angle in degrees.",
    )
    parser.add_argument(
        "--quality-weight",
        type=float,
        default=0.35,
        help="Weight [0-1] for quality metadata in candidate ranking (default: 0.35).",
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
        help="Maximum number of STAC items per planning window (passes through to STAC search).",
    )
    parser.add_argument(
        "--grouping",
        choices=("calendar", "single", "fixed"),
        default="calendar",
        help=(
            "Planning window grouping: calendar month buckets, single full range, "
            "or fixed-size windows (use with --window-days)."
        ),
    )
    parser.add_argument(
        "--window-days",
        type=int,
        help="Window size in days when --grouping fixed (for example 1=daily, 90=seasonal).",
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
    if args.grouping == "fixed":
        if args.window_days is None or args.window_days <= 0:
            parser.error(
                "--window-days must be a positive integer when --grouping fixed."
            )
    elif args.window_days is not None:
        parser.error("--window-days is only valid when --grouping fixed.")
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


def _order_id_for_window(
    order_results: Dict[str, Dict[str, Any]], window_key: str
) -> str:
    if window_key not in order_results:
        return "-"
    order_id = order_results[window_key].get("order_id")
    return order_id or "-"


def _print_summary(
    plan: Dict[str, Dict[str, Any]],
    order_results: Dict[str, Dict[str, Any]],
    sr_bands: int,
    harmonize: str,
) -> None:
    header = "Window                     Candidates  Filtered  Selected  Coverage  MinClearObs  SR-bands  Harmonize     Order ID"
    divider = "-" * len(header)
    print(header)
    print(divider)
    for window_key in sorted(plan.keys()):
        entry = plan[window_key]
        coverage = entry.get("aoi_coverage", 0.0)
        min_clear_obs = entry.get("min_clear_obs", 0.0)
        candidate_count = entry.get("candidate_count", 0)
        filtered_count = entry.get("filtered_count", 0)
        selected_count = entry.get("selected_count", 0)
        print(
            f"{window_key:24}  {candidate_count:10d}  {filtered_count:8d}  {selected_count:8d}  "
            f"{coverage:8.3f}  {min_clear_obs:11.1f}  {sr_bands:8d}  "
            f"{_harmonize_display(harmonize):11}  {_order_id_for_window(order_results, window_key)}"
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

    quality_category = args.quality_category
    if quality_category and quality_category.lower() == "none":
        quality_category = None
    publishing_stage = args.publishing_stage
    if publishing_stage and publishing_stage.lower() == "none":
        publishing_stage = None

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
        require_ground_control=args.require_ground_control,
        quality_category=quality_category,
        publishing_stage=publishing_stage,
        max_anomalous_pixels=args.max_anomalous_pixels,
        max_shadow_fraction=args.max_shadow,
        max_snow_ice_fraction=args.max_snow_ice,
        max_heavy_haze_fraction=args.max_heavy_haze,
        min_visible_confidence=args.min_visible_confidence,
        min_clear_confidence=args.min_clear_confidence,
        max_view_angle=args.max_view_angle,
        quality_weight=args.quality_weight,
        month_grouping=args.grouping,
        window_days=args.window_days,
        limit=args.limit,
        tile_size_m=args.tile_size_m,
    )

    order_results: Dict[str, Dict[str, Any]] = {}
    if args.order:
        logger.info("Submitting Planet orders for %d windows.", len(plan))
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
