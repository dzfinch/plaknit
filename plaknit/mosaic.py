"""Planet mosaic workflow orchestration and CLI."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
import logging
import math
import os
import sqlite3
import shutil
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from sklearn.linear_model import HuberRegressor

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

try:
    import otbApplication  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
    otbApplication = None  # type: ignore
    _OTB_IMPORT_ERROR = exc
else:  # pragma: no cover - import success depends on environment
    _OTB_IMPORT_ERROR = None

PathLike = Union[str, Path]
_LOGGING_CONFIGURED = False


@dataclass(frozen=True)
class MosaicJob:
    """Configuration required to run the Planet mosaic workflow."""

    inputs: Sequence[PathLike]
    output: PathLike
    udms: Optional[Sequence[PathLike]] = None
    workdir: Optional[PathLike] = None
    tmpdir: Optional[PathLike] = None
    ram: int = 131072
    jobs: int = 4
    skip_masking: bool = False
    sr_bands: int = 4
    add_ndvi: bool = False
    target_crs: Optional[str] = None
    harmonize_radiometry: bool = False
    metadata_jsons: Optional[Sequence[PathLike]] = None


@dataclass(frozen=True)
class ProjectionInfo:
    """Projection metadata extracted from one raster."""

    path: Path
    crs: CRS
    label: str
    center_lonlat: Tuple[float, float]


@dataclass(frozen=True)
class HarmoniScene:
    """Metadata used for graph-based radiometric harmonization."""

    raster_path: Path
    metadata_path: Path
    scene_key: str
    acquired: datetime
    bbox_wgs84: Tuple[float, float, float, float]
    area_wgs84: float


def _normalize_crs_label(crs: CRS) -> str:
    """Return a stable label for grouping equivalent CRSs."""
    epsg = crs.to_epsg()
    if epsg is not None:
        return f"EPSG:{epsg}"
    return crs.to_string()


def _canonicalize_crs(crs: CRS) -> CRS:
    """Prefer official EPSG definitions when possible."""
    epsg = crs.to_epsg()
    if epsg is None:
        return crs
    try:
        return CRS.from_epsg(epsg)
    except Exception:
        return crs


def _choose_target_projection(
    projections: Sequence[ProjectionInfo], requested_crs: Optional[str]
) -> CRS:
    """Pick target CRS using explicit override, then majority, then center tie-break."""
    if not projections:
        raise ValueError("No projection metadata available for mosaicking.")

    if requested_crs:
        try:
            return CRS.from_user_input(requested_crs)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid target CRS '{requested_crs}'.") from exc

    counts = Counter(info.label for info in projections)
    max_count = max(counts.values())
    winners = {label for label, count in counts.items() if count == max_count}
    if len(winners) == 1:
        winner = next(iter(winners))
        return next(info.crs for info in projections if info.label == winner)

    avg_lon = float(np.mean([info.center_lonlat[0] for info in projections]))
    avg_lat = float(np.mean([info.center_lonlat[1] for info in projections]))

    winner_info = min(
        (info for info in projections if info.label in winners),
        key=lambda info: (
            (info.center_lonlat[0] - avg_lon) ** 2
            + (info.center_lonlat[1] - avg_lat) ** 2,
            info.label,
        ),
    )
    return winner_info.crs


def _scene_key(path: Path) -> str:
    """Normalize scene names to match imagery and metadata files."""
    stem = path.stem.lower()
    for token in (
        "_3b_analyticms",
        "_analyticms",
        "_udm2",
        "_metadata",
        "_manifest",
    ):
        token_pos = stem.find(token)
        if token_pos > 0:
            stem = stem[:token_pos]
    return stem.rstrip("_-")


def _parse_iso_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    left, bottom, right, top = bbox
    return max(0.0, right - left) * max(0.0, top - bottom)


def _bbox_overlap_ratio(
    bbox_a: Tuple[float, float, float, float],
    bbox_b: Tuple[float, float, float, float],
    area_a: float,
    area_b: float,
) -> float:
    left = max(bbox_a[0], bbox_b[0])
    bottom = max(bbox_a[1], bbox_b[1])
    right = min(bbox_a[2], bbox_b[2])
    top = min(bbox_a[3], bbox_b[3])
    overlap = _bbox_area((left, bottom, right, top))
    denom = min(area_a, area_b)
    if overlap <= 0.0 or denom <= 0.0:
        return 0.0
    return overlap / denom


def _split_env_paths(raw: Optional[str]) -> List[Path]:
    if not raw:
        return []
    paths: List[Path] = []
    for entry in raw.split(os.pathsep):
        entry = entry.strip()
        if not entry:
            continue
        path = Path(entry).expanduser()
        if path.name == "proj.db":
            path = path.parent
        paths.append(path)
    return paths


def _discover_proj_directories() -> List[Path]:
    candidates: List[Path] = []
    seen: Set[str] = set()

    for env_var in ("PROJ_DATA", "PROJ_LIB"):
        for path in _split_env_paths(os.environ.get(env_var)):
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(path)

    try:
        from rasterio.env import PROJDataFinder

        discovered = PROJDataFinder().search()
    except Exception:
        discovered = None
    if discovered:
        path = Path(discovered).expanduser()
        key = str(path)
        if key not in seen:
            seen.add(key)
            candidates.append(path)

    otb_proj_dir = Path("/app/otb/share/proj")
    if otb_proj_dir.exists():
        key = str(otb_proj_dir)
        if key not in seen:
            seen.add(key)
            candidates.append(otb_proj_dir)

    return candidates


def _read_proj_layout_version(proj_db_path: Path) -> Optional[Tuple[int, int]]:
    if not proj_db_path.exists():
        return None
    try:
        with sqlite3.connect(str(proj_db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value FROM metadata WHERE key='DATABASE.LAYOUT.VERSION.MAJOR'"
            )
            major_row = cursor.fetchone()
            cursor.execute(
                "SELECT value FROM metadata WHERE key='DATABASE.LAYOUT.VERSION.MINOR'"
            )
            minor_row = cursor.fetchone()
    except Exception:
        return None

    if not major_row or not minor_row:
        return None
    try:
        return (int(major_row[0]), int(minor_row[0]))
    except (TypeError, ValueError):
        return None


def _collect_proj_layout_versions() -> List[Tuple[Path, Tuple[int, int]]]:
    collected: List[Tuple[Path, Tuple[int, int]]] = []
    for directory in _discover_proj_directories():
        proj_db_path = directory / "proj.db"
        version = _read_proj_layout_version(proj_db_path)
        if version is None:
            continue
        try:
            display_dir = directory.resolve()
        except Exception:
            display_dir = directory
        collected.append((display_dir, version))
    return collected


def _extract_bbox_from_coordinates(
    coordinates: Any,
) -> Optional[Tuple[float, float, float, float]]:
    points: List[Tuple[float, float]] = []

    def _walk(node: Any) -> None:
        if isinstance(node, (list, tuple)):
            if len(node) >= 2 and all(
                isinstance(value, (int, float)) for value in node[:2]
            ):
                points.append((float(node[0]), float(node[1])))
                return
            for child in node:
                _walk(child)

    _walk(coordinates)
    if not points:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _extract_bbox_from_metadata(
    metadata: Dict[str, Any],
) -> Optional[Tuple[float, float, float, float]]:
    bbox = metadata.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

    geometry = metadata.get("geometry")
    if isinstance(geometry, dict):
        coordinates = geometry.get("coordinates")
        bbox_from_geom = _extract_bbox_from_coordinates(coordinates)
        if bbox_from_geom is not None:
            return bbox_from_geom

    return None


def configure_logging(verbosity: int) -> logging.Logger:
    """Configure and return a module-level logger."""
    global _LOGGING_CONFIGURED
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    root = logging.getLogger()
    if not _LOGGING_CONFIGURED or not root.handlers:
        logging.basicConfig(level=level, format="%(message)s")
        _LOGGING_CONFIGURED = True
    else:
        root.setLevel(level)

    logger = logging.getLogger("plaknit.mosaic")
    logger.setLevel(level)
    return logger


class MosaicWorkflow:
    """Coordinate masking and OTB mosaicking."""

    _HARMONI_MAX_TIME_GAP_DAYS = 40
    _HARMONI_MIN_OVERLAP_RATIO = 0.2
    _HARMONI_MIN_SAMPLES = 1000
    _HARMONI_MAX_SAMPLES = 200000
    _HARMONI_SLOPE_BOUNDS = (0.25, 4.0)
    _HARMONI_INTERCEPT_ABS_MAX = 10000.0

    def __init__(self, job: MosaicJob, logger: Optional[logging.Logger] = None):
        self.job = job
        self.log = logger or logging.getLogger("plaknit.mosaic")
        self._tmpdir_created: Optional[Path] = None
        self._workdir_created: Optional[Path] = None
        self._progress_lock = threading.Lock()

    @contextmanager
    def _progress(self, enabled: bool = True):
        if not enabled or Progress is None:
            yield None
            return
        progress = Progress(
            BarColumn(),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=5,
        )
        progress.start()
        try:
            yield progress
        finally:
            progress.stop()

    def run(self) -> Path:
        """Execute the workflow and return the output path."""
        job = self.job
        if not job.inputs:
            raise ValueError("At least one input strip must be provided.")
        if job.harmonize_radiometry and not job.metadata_jsons:
            raise ValueError(
                "Metadata JSONs are required when --harmonize-radiometry is enabled."
            )
        if job.ram <= 0:
            raise ValueError("RAM must be a positive integer.")
        if job.add_ndvi and job.sr_bands not in (4, 8):
            raise ValueError("sr_bands must be 4 or 8 when --ndvi is requested.")

        self._configure_environment()
        self._warn_on_proj_layout_mismatch()
        inputs = self._expand(job.inputs, label="inputs")
        tmpdir = self._prepare_tmpdir()
        with self._progress(enabled=self.log.isEnabledFor(logging.INFO)) as progress:
            harmonization_task = (
                progress.add_task("Radiometry", total=len(inputs))
                if progress and job.harmonize_radiometry
                else None
            )
            mask_task = (
                progress.add_task("Mask tiles", total=len(inputs))
                if progress and not job.skip_masking
                else None
            )
            prep_task = progress.add_task("Binary mask", total=1) if progress else None
            projection_task = (
                progress.add_task("Projection", total=len(inputs)) if progress else None
            )
            mosaic_task = progress.add_task("Mosaic", total=1) if progress else None

            try:
                prepared_inputs = self._harmonize_radiometry(
                    inputs, tmpdir, progress, harmonization_task
                )

                if job.skip_masking:
                    masked_paths = prepared_inputs
                    if progress and mask_task is not None:
                        progress.update(mask_task, total=1, completed=1)
                else:
                    if not job.udms:
                        raise ValueError(
                            "UDM rasters are required unless --skip-masking is provided."
                        )
                    udms = self._expand(job.udms, label="UDMs")
                    if len(prepared_inputs) != len(udms):
                        raise ValueError(
                            "Input/UDM mismatch: expected "
                            f"{len(prepared_inputs)} UDMs but received {len(udms)}."
                        )
                    masked_paths = self._mask_inputs(
                        prepared_inputs, udms, progress, mask_task
                    )

                mosaic_inputs = self._harmonize_projections(
                    masked_paths, tmpdir, progress, projection_task
                )
                self._prepare_output_directory()

                # Binary mask prep is handled inside OTB; mark as ready before launch.
                if progress and prep_task is not None:
                    progress.update(prep_task, completed=1)

                mosaic_path = self._run_mosaic(
                    mosaic_inputs, tmpdir, progress, mosaic_task
                )
                if job.add_ndvi:
                    self._append_ndvi(mosaic_path, job.sr_bands)
            finally:
                self._cleanup_tmpdir()

        output_path = Path(job.output).expanduser()
        self.log.info("Mosaic complete: %s", output_path)
        return output_path

    def _expand(self, entries: Sequence[PathLike], label: str) -> List[str]:
        resolved: List[str] = []
        for entry in entries:
            path = Path(entry).expanduser()
            if path.is_dir():
                rasters = sorted(str(p) for p in path.glob("*.tif"))
                if not rasters:
                    raise ValueError(f"No .tif rasters found in directory '{path}'.")
                resolved.extend(rasters)
            elif path.exists():
                resolved.append(str(path))
            else:
                raise FileNotFoundError(f"{label} path '{entry}' does not exist.")

        if not resolved:
            raise ValueError(f"No rasters detected for {label}.")

        return resolved

    def _prepare_tmpdir(self) -> Path:
        if self.job.tmpdir:
            tmpdir = Path(self.job.tmpdir).expanduser()
            tmpdir.mkdir(parents=True, exist_ok=True)
            return tmpdir

        created = Path(tempfile.mkdtemp(prefix="otb_tmp_"))
        self._tmpdir_created = created
        return created

    def _ensure_workdir(self) -> Path:
        if self.job.workdir:
            workdir = Path(self.job.workdir).expanduser()
            workdir.mkdir(parents=True, exist_ok=True)
            return workdir

        if self._workdir_created is None:
            self._workdir_created = Path(tempfile.mkdtemp(prefix="mask_work_"))

        return self._workdir_created

    def _prepare_output_directory(self) -> None:
        output_path = Path(self.job.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    def _configure_environment(self) -> None:
        jobs = max(1, self.job.jobs)
        os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(jobs))
        os.environ.setdefault("OTB_MAX_RAM_HINT", str(self.job.ram))
        os.environ.setdefault("GTIFF_SRS_SOURCE", "EPSG")

    def _warn_on_proj_layout_mismatch(self) -> None:
        versions = _collect_proj_layout_versions()
        if len(versions) < 2:
            return

        unique_versions = {version for _, version in versions}
        if len(unique_versions) <= 1:
            return

        details = ", ".join(
            f"{directory} (layout {version[0]}.{version[1]})"
            for directory, version in versions
        )
        self.log.warning(
            "Detected mixed PROJ database layouts: %s. "
            "Set PROJ_DATA and PROJ_LIB to the same PROJ directory before running.",
            details,
        )

    def _center_lonlat_for_bounds(
        self, path: Path, crs: CRS, bounds: Tuple[float, float, float, float]
    ) -> Tuple[float, float]:
        try:
            with rasterio.Env(GTIFF_SRS_SOURCE="EPSG"):
                left, bottom, right, top = transform_bounds(
                    _canonicalize_crs(crs), "EPSG:4326", *bounds, densify_pts=21
                )
            return ((left + right) / 2.0, (bottom + top) / 2.0)
        except Exception as exc:
            self.log.warning(
                "Could not transform bounds to EPSG:4326 for '%s' (%s); "
                "using native bounds center for CRS tie-breaking.",
                path,
                exc,
            )
            left, bottom, right, top = bounds
            return ((left + right) / 2.0, (bottom + top) / 2.0)

    def _mask_inputs(
        self,
        strips: Sequence[str],
        udms: Sequence[str],
        progress: Optional[Progress],
        task_id: Optional[int],
    ) -> List[str]:
        workdir = self._ensure_workdir()
        jobs = max(1, self.job.jobs)
        self.log.debug("Masking %s strips using %s parallel jobs.", len(strips), jobs)

        masked_paths: List[str] = []
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = []
            for strip_path, udm_path in zip(strips, udms):
                strip = Path(strip_path)
                udm = Path(udm_path)
                masked = workdir / f"{strip.stem}_masked.tif"
                futures.append(pool.submit(self._mask_single_strip, strip, udm, masked))

            for future in as_completed(futures):
                masked_paths.append(str(future.result()))
                if progress and task_id is not None:
                    with self._progress_lock:
                        progress.advance(task_id)

        masked_paths.sort()
        return masked_paths

    def _expand_jsons(self, entries: Sequence[PathLike], label: str) -> List[str]:
        resolved: List[str] = []
        for entry in entries:
            path = Path(entry).expanduser()
            if path.is_dir():
                preferred = sorted(str(p) for p in path.glob("*_metadata.json"))
                jsons = preferred or sorted(str(p) for p in path.glob("*.json"))
                if not jsons:
                    raise ValueError(f"No .json files found in directory '{path}'.")
                resolved.extend(jsons)
            elif path.exists():
                if path.suffix.lower() != ".json":
                    raise ValueError(f"{label} path '{entry}' is not a .json file.")
                resolved.append(str(path))
            else:
                raise FileNotFoundError(f"{label} path '{entry}' does not exist.")

        if not resolved:
            raise ValueError(f"No metadata JSONs detected for {label}.")
        return resolved

    def _extract_scene_acquired(
        self, metadata: Dict[str, Any], metadata_path: Path
    ) -> datetime:
        properties = metadata.get("properties")
        if not isinstance(properties, dict):
            properties = {}

        candidates = (
            properties.get("acquired"),
            properties.get("datetime"),
            metadata.get("acquired"),
            metadata.get("datetime"),
        )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                try:
                    return _parse_iso_datetime(candidate)
                except ValueError:
                    continue

        raise ValueError(
            f"Metadata '{metadata_path}' is missing a parseable acquisition datetime."
        )

    def _match_metadata_to_rasters(
        self, rasters: Sequence[str], metadata_paths: Sequence[str]
    ) -> Dict[str, Path]:
        metadata_by_key: Dict[str, List[Path]] = {}
        for metadata_path in metadata_paths:
            metadata = Path(metadata_path)
            metadata_by_key.setdefault(_scene_key(metadata), []).append(metadata)

        assignment: Dict[str, Path] = {}
        unmatched: List[str] = []
        for raster_path in rasters:
            raster = Path(raster_path)
            key = _scene_key(raster)

            candidates = list(metadata_by_key.get(key, []))
            if not candidates:
                for candidate_key, candidate_paths in metadata_by_key.items():
                    if candidate_key.startswith(key) or key.startswith(candidate_key):
                        candidates.extend(candidate_paths)

            candidates = sorted(set(candidates))
            if len(candidates) == 1:
                chosen = candidates[0]
                assignment[raster_path] = chosen
                metadata_by_key[_scene_key(chosen)] = [
                    item
                    for item in metadata_by_key[_scene_key(chosen)]
                    if item != chosen
                ]
                continue

            if len(candidates) > 1:
                raise ValueError(
                    "Ambiguous metadata match for raster "
                    f"'{raster}': {[str(item) for item in candidates]}"
                )

            unmatched.append(raster_path)

        if unmatched:
            if len(rasters) == len(metadata_paths):
                self.log.warning(
                    "Falling back to sorted raster/metadata pairing for %s unmatched scenes.",
                    len(unmatched),
                )
                return {
                    raster: Path(metadata)
                    for raster, metadata in zip(sorted(rasters), sorted(metadata_paths))
                }
            raise ValueError(
                "Could not match metadata JSONs to rasters for: " + ", ".join(unmatched)
            )

        return assignment

    def _load_harmoni_scenes(
        self, rasters: Sequence[str], metadata_paths: Sequence[str]
    ) -> List[HarmoniScene]:
        metadata_lookup = self._match_metadata_to_rasters(rasters, metadata_paths)
        scenes: List[HarmoniScene] = []
        for raster_path in rasters:
            raster = Path(raster_path)
            metadata_path = metadata_lookup[raster_path]
            with metadata_path.open("r", encoding="utf-8") as src:
                metadata = json.load(src)

            acquired = self._extract_scene_acquired(metadata, metadata_path)
            bbox = _extract_bbox_from_metadata(metadata)
            if bbox is None:
                with rasterio.open(raster) as dataset:
                    if dataset.crs is None:
                        raise ValueError(
                            f"Input raster '{raster}' has no CRS metadata and cannot be used."
                        )
                    dataset_crs = _canonicalize_crs(dataset.crs)
                    try:
                        with rasterio.Env(GTIFF_SRS_SOURCE="EPSG"):
                            bbox = transform_bounds(
                                dataset_crs,
                                "EPSG:4326",
                                *dataset.bounds,
                                densify_pts=21,
                            )
                    except Exception as exc:
                        raise ValueError(
                            f"Unable to derive WGS84 bounds from '{raster}'."
                        ) from exc
            area = _bbox_area(bbox)
            if area <= 0.0:
                raise ValueError(
                    f"Metadata '{metadata_path}' does not provide a usable scene footprint."
                )

            scenes.append(
                HarmoniScene(
                    raster_path=raster,
                    metadata_path=metadata_path,
                    scene_key=_scene_key(raster),
                    acquired=acquired,
                    bbox_wgs84=bbox,
                    area_wgs84=area,
                )
            )
        return scenes

    def _construct_harmoni_graph(
        self, scenes: Sequence[HarmoniScene]
    ) -> Dict[int, Dict[int, float]]:
        graph: Dict[int, Dict[int, float]] = {index: {} for index in range(len(scenes))}
        for left in range(len(scenes)):
            for right in range(left + 1, len(scenes)):
                overlap_ratio = _bbox_overlap_ratio(
                    scenes[left].bbox_wgs84,
                    scenes[right].bbox_wgs84,
                    scenes[left].area_wgs84,
                    scenes[right].area_wgs84,
                )
                if overlap_ratio < self._HARMONI_MIN_OVERLAP_RATIO:
                    continue

                delta_days = abs((scenes[left].acquired - scenes[right].acquired).days)
                if delta_days > self._HARMONI_MAX_TIME_GAP_DAYS:
                    continue

                temporal_weight = math.exp(
                    -delta_days / float(self._HARMONI_MAX_TIME_GAP_DAYS)
                )
                weight = overlap_ratio * temporal_weight
                graph[left][right] = weight
                graph[right][left] = weight

        return graph

    def _graph_components(self, graph: Dict[int, Dict[int, float]]) -> List[Set[int]]:
        seen: Set[int] = set()
        components: List[Set[int]] = []
        for root in sorted(graph):
            if root in seen:
                continue
            stack = [root]
            component: Set[int] = set()
            while stack:
                node = stack.pop()
                if node in component:
                    continue
                component.add(node)
                stack.extend(neigh for neigh in graph[node] if neigh not in component)
            seen.update(component)
            components.append(component)
        return components

    def _select_harmoni_reference(
        self,
        component: Sequence[int],
        graph: Dict[int, Dict[int, float]],
        scenes: Sequence[HarmoniScene],
    ) -> int:
        timestamps = [scenes[index].acquired.timestamp() for index in component]
        mean_timestamp = float(np.mean(timestamps))
        return max(
            component,
            key=lambda index: (
                sum(graph[index].values()),
                -abs(scenes[index].acquired.timestamp() - mean_timestamp),
                -index,
            ),
        )

    def _select_harmonization_chain(
        self,
        component: Sequence[int],
        graph: Dict[int, Dict[int, float]],
        reference: int,
    ) -> List[Tuple[int, int, float]]:
        component_set = set(component)
        visited = {reference}
        chain: List[Tuple[int, int, float]] = []
        while len(visited) < len(component_set):
            best_parent = -1
            best_child = -1
            best_weight = -1.0
            for parent in sorted(visited):
                for child, weight in sorted(graph[parent].items()):
                    if child in visited or child not in component_set:
                        continue
                    if weight > best_weight:
                        best_parent = parent
                        best_child = child
                        best_weight = weight

            if best_child == -1:
                child = min(component_set - visited)
                best_parent = reference
                best_child = child
                best_weight = 0.0

            visited.add(best_child)
            chain.append((best_parent, best_child, best_weight))

        return chain

    def _harmonize_radiometry(
        self,
        rasters: Sequence[str],
        tmpdir: Path,
        progress: Optional[Progress],
        task_id: Optional[int],
    ) -> List[str]:
        if not self.job.harmonize_radiometry:
            return list(rasters)

        metadata_jsons = self.job.metadata_jsons or []
        metadata_paths = self._expand_jsons(metadata_jsons, label="metadata_jsons")
        scenes = self._load_harmoni_scenes(rasters, metadata_paths)
        graph = self._construct_harmoni_graph(scenes)
        edge_count = int(sum(len(edges) for edges in graph.values()) / 2)
        components = self._graph_components(graph)
        self.log.info(
            "Running graph-based radiometric harmonization (%s scenes, %s edges, %s components).",
            len(scenes),
            edge_count,
            len(components),
        )

        output_paths: List[Optional[str]] = [None] * len(scenes)
        harmonized_dir = tmpdir / "radiometry_harmonized"
        harmonized_dir.mkdir(parents=True, exist_ok=True)

        for component in components:
            nodes = sorted(component)
            reference = self._select_harmoni_reference(nodes, graph, scenes)
            output_paths[reference] = str(scenes[reference].raster_path)
            if progress and task_id is not None:
                with self._progress_lock:
                    progress.advance(task_id)

            chain = self._select_harmonization_chain(nodes, graph, reference)
            for parent, child, weight in chain:
                source_path = Path(output_paths[child] or scenes[child].raster_path)
                parent_path = Path(output_paths[parent] or scenes[parent].raster_path)
                factors = self._estimate_adjustment_factors(source_path, parent_path)
                if factors is None:
                    self.log.warning(
                        "Skipping radiometric adjustment for '%s' -> '%s' (insufficient overlap).",
                        source_path.name,
                        parent_path.name,
                    )
                    output_paths[child] = str(source_path)
                else:
                    out_path = harmonized_dir / (
                        f"{child:04d}_{source_path.stem}_harmoni.tif"
                    )
                    self.log.debug(
                        "Radiometric adjust %s -> %s (edge weight=%.3f, output=%s).",
                        source_path.name,
                        parent_path.name,
                        weight,
                        out_path.name,
                    )
                    self._apply_adjustment_factors(source_path, out_path, factors)
                    output_paths[child] = str(out_path)

                if progress and task_id is not None:
                    with self._progress_lock:
                        progress.advance(task_id)

        return [
            output_paths[index] if output_paths[index] else str(scene.raster_path)
            for index, scene in enumerate(scenes)
        ]

    def _estimate_adjustment_factors(
        self, source_path: Path, reference_path: Path
    ) -> Optional[List[Tuple[float, float]]]:
        with rasterio.open(reference_path) as ref, rasterio.open(source_path) as src:
            band_count = min(src.count, ref.count)
            if band_count <= 0:
                return None

            try:
                if src.crs is None or ref.crs is None:
                    return None
                src_crs = _canonicalize_crs(src.crs)
                ref_crs = _canonicalize_crs(ref.crs)
                with rasterio.Env(GTIFF_SRS_SOURCE="EPSG"):
                    source_bounds_in_ref = transform_bounds(
                        src_crs, ref_crs, *src.bounds, densify_pts=21
                    )
            except Exception:
                return None

            overlap_left = max(source_bounds_in_ref[0], ref.bounds.left)
            overlap_bottom = max(source_bounds_in_ref[1], ref.bounds.bottom)
            overlap_right = min(source_bounds_in_ref[2], ref.bounds.right)
            overlap_top = min(source_bounds_in_ref[3], ref.bounds.top)
            if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
                return None

            ref_window = (
                from_bounds(
                    overlap_left,
                    overlap_bottom,
                    overlap_right,
                    overlap_top,
                    transform=ref.transform,
                )
                .round_offsets()
                .round_lengths()
            )
            if ref_window.width < 1 or ref_window.height < 1:
                return None

            indexes = list(range(1, band_count + 1))
            ref_data = ref.read(indexes=indexes, window=ref_window, out_dtype="float32")
            ref_transform = ref.window_transform(ref_window)
            with WarpedVRT(
                src,
                crs=ref.crs,
                transform=ref_transform,
                width=int(ref_window.width),
                height=int(ref_window.height),
                resampling=Resampling.bilinear,
            ) as src_vrt:
                src_data = src_vrt.read(indexes=indexes, out_dtype="float32")

            factors: List[Tuple[float, float]] = []
            random_state = np.random.default_rng(42)
            for band_index in range(band_count):
                source_band = src_data[band_index]
                reference_band = ref_data[band_index]

                valid = np.isfinite(source_band) & np.isfinite(reference_band)
                if src.nodata is not None:
                    valid &= source_band != src.nodata
                if ref.nodata is not None:
                    valid &= reference_band != ref.nodata

                source_samples = source_band[valid]
                reference_samples = reference_band[valid]
                if source_samples.size < self._HARMONI_MIN_SAMPLES:
                    factors.append((1.0, 0.0))
                    continue

                if source_samples.size > self._HARMONI_MAX_SAMPLES:
                    chosen = random_state.choice(
                        source_samples.size,
                        size=self._HARMONI_MAX_SAMPLES,
                        replace=False,
                    )
                    source_samples = source_samples[chosen]
                    reference_samples = reference_samples[chosen]

                source_bounds = np.percentile(source_samples, [5, 95])
                reference_bounds = np.percentile(reference_samples, [5, 95])
                pif_mask = (
                    (source_samples >= source_bounds[0])
                    & (source_samples <= source_bounds[1])
                    & (reference_samples >= reference_bounds[0])
                    & (reference_samples <= reference_bounds[1])
                )
                if int(np.count_nonzero(pif_mask)) >= self._HARMONI_MIN_SAMPLES:
                    source_samples = source_samples[pif_mask]
                    reference_samples = reference_samples[pif_mask]

                factors.append(self._fit_adjustment(source_samples, reference_samples))

            while len(factors) < src.count:
                factors.append((1.0, 0.0))
            return factors

    def _fit_adjustment(
        self, source_samples: np.ndarray, reference_samples: np.ndarray
    ) -> Tuple[float, float]:
        slope = 1.0
        intercept = 0.0
        x = source_samples.reshape(-1, 1)
        y = reference_samples
        try:
            model = HuberRegressor(alpha=0.0, fit_intercept=True)
            model.fit(x, y)
            slope = float(model.coef_[0])
            intercept = float(model.intercept_)
        except Exception:
            try:
                slope, intercept = np.polyfit(source_samples, reference_samples, 1)
                slope = float(slope)
                intercept = float(intercept)
            except Exception:
                slope = 1.0
                intercept = 0.0

        if not np.isfinite(slope) or abs(slope) < 1e-6:
            slope = 1.0
        if not np.isfinite(intercept):
            intercept = 0.0

        slope = float(np.clip(slope, *self._HARMONI_SLOPE_BOUNDS))
        intercept = float(
            np.clip(
                intercept,
                -self._HARMONI_INTERCEPT_ABS_MAX,
                self._HARMONI_INTERCEPT_ABS_MAX,
            )
        )
        return (slope, intercept)

    def _apply_adjustment_factors(
        self,
        source_path: Path,
        destination_path: Path,
        factors: Sequence[Tuple[float, float]],
    ) -> None:
        with rasterio.open(source_path) as src:
            profile = src.profile.copy()
            profile.update(dtype="float32")
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(destination_path, "w", **profile) as dst:
                for band_index in range(1, src.count + 1):
                    slope, intercept = factors[band_index - 1]
                    band = src.read(band_index).astype("float32")
                    adjusted = band * slope + intercept
                    if src.nodata is not None:
                        adjusted = np.where(
                            band == src.nodata, float(src.nodata), adjusted
                        )
                    dst.write(adjusted, band_index)

    def _harmonize_projections(
        self,
        rasters: Sequence[str],
        tmpdir: Path,
        progress: Optional[Progress],
        task_id: Optional[int],
    ) -> List[str]:
        projections = self._collect_projection_info(rasters)
        target_crs = _choose_target_projection(projections, self.job.target_crs)
        target_label = _normalize_crs_label(target_crs)
        counts = Counter(info.label for info in projections)
        has_mismatch = any(info.label != target_label for info in projections)
        if not has_mismatch:
            self.log.debug("Input rasters already share CRS %s.", target_label)
            if progress and task_id is not None:
                progress.update(task_id, total=len(rasters), completed=len(rasters))
            return [str(info.path) for info in projections]

        self.log.info(
            "Normalizing projections to %s (input CRS counts: %s).",
            target_label,
            dict(sorted(counts.items())),
        )
        harmonized_dir = tmpdir / "harmonized"
        harmonized_dir.mkdir(parents=True, exist_ok=True)

        harmonized_paths: List[str] = []
        for index, info in enumerate(projections):
            if info.crs == target_crs:
                harmonized_paths.append(str(info.path))
            else:
                target_name = target_label.replace(":", "_")
                out_path = harmonized_dir / (
                    f"{index:04d}_{info.path.stem}_{target_name}.tif"
                )
                self._warp_to_crs(info.path, out_path, target_crs)
                harmonized_paths.append(str(out_path))

            if progress and task_id is not None:
                with self._progress_lock:
                    progress.advance(task_id)

        return harmonized_paths

    def _collect_projection_info(self, rasters: Sequence[str]) -> List[ProjectionInfo]:
        projections: List[ProjectionInfo] = []
        for raster in rasters:
            path = Path(raster)
            with rasterio.open(path) as src:
                if src.crs is None:
                    raise ValueError(
                        f"Input raster '{path}' has no CRS metadata and cannot be mosaicked."
                    )

                canonical_crs = _canonicalize_crs(src.crs)
                center = self._center_lonlat_for_bounds(
                    path, canonical_crs, tuple(src.bounds)
                )
                projections.append(
                    ProjectionInfo(
                        path=path,
                        crs=canonical_crs,
                        label=_normalize_crs_label(canonical_crs),
                        center_lonlat=center,
                    )
                )

        return projections

    def _warp_to_crs(self, source: Path, destination: Path, target_crs: CRS) -> None:
        cmd = [
            "gdalwarp",
            "-overwrite",
            "-multi",
            "-t_srs",
            target_crs.to_string(),
            str(source),
            str(destination),
        ]
        self.log.debug("Reprojecting raster with command: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    def _ensure_masked_georeferencing(
        self, strip_path: Path, masked_path: Path
    ) -> None:
        with rasterio.open(strip_path) as source:
            source_crs = source.crs
            source_transform = source.transform

        with rasterio.open(masked_path) as masked:
            masked_crs = masked.crs
            masked_transform = masked.transform

        has_crs = masked_crs is not None
        has_transform = not masked_transform.is_identity
        if has_crs and has_transform:
            return

        if source_crs is None or source_transform.is_identity:
            raise ValueError(
                f"Masked raster '{masked_path}' is missing CRS/transform metadata and "
                f"cannot be repaired from source '{strip_path}'. This often indicates "
                "mixed PROJ installations; set PROJ_DATA and PROJ_LIB to the same "
                "directory."
            )

        try:
            with rasterio.open(masked_path, "r+") as masked_fix:
                if masked_fix.crs is None:
                    masked_fix.crs = source_crs
                if masked_fix.transform.is_identity:
                    masked_fix.transform = source_transform
        except Exception as exc:
            raise ValueError(
                f"Failed to repair CRS/transform metadata for masked raster "
                f"'{masked_path}'. This may indicate a PROJ/GDAL environment mismatch."
            ) from exc

        with rasterio.open(masked_path) as repaired:
            repaired_ok = (
                repaired.crs is not None and not repaired.transform.is_identity
            )
        if not repaired_ok:
            raise ValueError(
                f"Masked raster '{masked_path}' is still missing CRS/transform metadata "
                "after attempted repair. This may indicate a PROJ/GDAL environment "
                "mismatch."
            )

        self.log.warning(
            "Masked raster '%s' was missing CRS/transform metadata; repaired from '%s'.",
            masked_path,
            strip_path,
        )

    def _mask_single_strip(
        self, strip_path: Path, udm_path: Path, masked_path: Path
    ) -> Path:
        cmd = [
            "gdal_calc.py",
            "-A",
            str(strip_path),
            "-B",
            str(udm_path),
            "--allBands",
            "A",
            "--calc",
            "A*(B==1)",
            "--NoDataValue",
            "0",
            "--overwrite",
            "--creation-option",
            "TILED=YES",
            "--creation-option",
            "BLOCKXSIZE=512",
            "--creation-option",
            "BLOCKYSIZE=512",
            "--creation-option",
            "BIGTIFF=YES",
            "--creation-option",
            "COMPRESS=NONE",
            "--outfile",
            str(masked_path),
        ]
        self.log.debug("Masking strip with command: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        self._ensure_masked_georeferencing(strip_path, masked_path)
        return masked_path

    def _run_mosaic(
        self,
        rasters: Sequence[str],
        tmpdir: Path,
        progress: Optional[Progress],
        task_id: Optional[int],
    ) -> Path:
        if otbApplication is None:  # pragma: no cover - environment specific
            raise RuntimeError(
                "otbApplication bindings are not available. Install the OTB Python "
                "packages before running the mosaic workflow."
            ) from _OTB_IMPORT_ERROR

        self.log.debug(
            "Running OTB Mosaic on %s strips with RAM=%s MB.",
            len(rasters),
            self.job.ram,
        )

        app = otbApplication.Registry.CreateApplication("Mosaic")
        params = {
            "comp.feather": "slim",
            "comp.feather.slim.exponent": 1,
            "comp.feather.slim.length": 0,
            "distancemap.sr": 10,
            "harmo.method": "none",
            "harmo.cost": "rmse",
            "il": list(rasters),
            "interpolator": "bco",
            "interpolator.bco.radius": 2,
            "nodata": 0,
            "output.spacingx": 3,
            "output.spacingy": 3,
            "tmpdir": str(tmpdir),
            "ram": self.job.ram,
            "out": str(Path(self.job.output).expanduser()),
        }

        for key, value in params.items():
            if value is None:
                continue
            if key == "il":
                app.SetParameterStringList(key, [str(v) for v in value])
            elif isinstance(value, list):
                app.SetParameterStringList(key, [str(v) for v in value])
            elif isinstance(value, str):
                app.SetParameterString(key, value)
            elif isinstance(value, int):
                app.SetParameterInt(key, value)
            elif isinstance(value, float):
                app.SetParameterFloat(key, value)

        app.ExecuteAndWriteOutput()
        if progress and task_id is not None:
            progress.update(task_id, completed=1)
        return Path(params["out"])

    def _append_ndvi(self, mosaic_path: Path, sr_bands: int) -> Path:
        """Compute NDVI and append as an extra band to the mosaic."""
        nir_band = 4 if sr_bands == 4 else 8
        red_band = 3 if sr_bands == 4 else 6
        with rasterio.open(mosaic_path, "r") as src:
            profile = src.profile.copy()
            profile.update(count=src.count + 1, dtype="float32")
            data = src.read()
            nir = data[nir_band - 1].astype("float32")
            red = data[red_band - 1].astype("float32")
            with rasterio.Env():
                with rasterio.open(mosaic_path, "w", **profile) as dst:
                    dst.write(data)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ndvi = (nir - red) / (nir + red)
                    if src.nodata is not None:
                        mask = (nir == src.nodata) | (red == src.nodata)
                        ndvi = np.where(mask, np.nan, ndvi)
                    ndvi = np.nan_to_num(ndvi)
                    dst.write(ndvi, src.count + 1)
        return mosaic_path

    def _cleanup_tmpdir(self) -> None:
        if self._tmpdir_created and self._tmpdir_created.exists():
            try:
                shutil.rmtree(self._tmpdir_created)
            except OSError:
                pass


def run_mosaic(job: MosaicJob, logger: Optional[logging.Logger] = None) -> Path:
    """Convenience helper for running the workflow in a single call."""
    workflow = MosaicWorkflow(job, logger=logger)
    return workflow.run()


# ------------------------------- CLI Helpers ------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the mosaic workflow."""
    parser = argparse.ArgumentParser(
        prog="plaknit mosaic",
        description=(
            "Mask Planet strips with UDM rasters, optionally harmonize radiometry, "
            "mosaic them with OTB, and optionally append NDVI."
        ),
    )
    parser.add_argument(
        "--inputs",
        "-il",
        nargs="+",
        required=True,
        help="Input strip GeoTIFFs or directories containing them.",
    )
    parser.add_argument(
        "--udms",
        "-udm",
        nargs="*",
        help="UDM GeoTIFFs (required unless --skip-masking).",
    )
    parser.add_argument(
        "--output",
        "-out",
        required=True,
        help="Destination GeoTIFF for the final mosaic.",
    )
    parser.add_argument(
        "--workdir",
        "-w",
        default="",
        help="Directory for intermediate masked strips (defaults to a temp directory).",
    )
    parser.add_argument(
        "--tmpdir",
        "-t",
        default="",
        help="Temporary directory for OTB scratch files (defaults to a temp directory).",
    )
    parser.add_argument(
        "--ram",
        "-r",
        type=int,
        default=131072,
        help="Maximum RAM available to OTB in MB (default: 131072 = 128 GB).",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=4,
        help="Parallel jobs for the masking step (default: 4).",
    )
    parser.add_argument(
        "--skip-masking",
        action="store_true",
        help="Skip masking and use --inputs directly for mosaicking.",
    )
    parser.add_argument(
        "--sr-bands",
        type=int,
        choices=(4, 8),
        default=4,
        help="Surface reflectance band count (4 or 8, default: 4).",
    )
    parser.add_argument(
        "--ndvi",
        action="store_true",
        help="Compute NDVI (NIR-Red / NIR+Red) and append as an extra band.",
    )
    parser.add_argument(
        "--harmonize-radiometry",
        action="store_true",
        help=(
            "Enable graph-based radiometric harmonization (Harmoni-Planet style) "
            "before masking/projection steps."
        ),
    )
    parser.add_argument(
        "--metadata-jsons",
        "-meta",
        nargs="*",
        help=(
            "Scene metadata JSON files or directories (required when "
            "--harmonize-radiometry is set)."
        ),
    )
    parser.add_argument(
        "--target-crs",
        default="",
        help=(
            "Optional CRS override (for example EPSG:32735). If omitted, the "
            "workflow uses the majority CRS and breaks ties using the tile nearest "
            "the geographic center."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity: -v (info), -vv (debug).",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = build_parser()
    return parser.parse_args(argv)


def _blank_to_none(value: str) -> Optional[str]:
    return value or None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point used by both python -m plaknit.mosaic and the plaknit CLI."""
    args = parse_args(argv)
    logger = configure_logging(args.verbose)

    job = MosaicJob(
        inputs=args.inputs,
        udms=args.udms,
        output=args.output,
        workdir=_blank_to_none(args.workdir),
        tmpdir=_blank_to_none(args.tmpdir),
        ram=args.ram,
        jobs=args.jobs,
        skip_masking=args.skip_masking,
        sr_bands=args.sr_bands,
        add_ndvi=args.ndvi,
        harmonize_radiometry=args.harmonize_radiometry,
        metadata_jsons=args.metadata_jsons,
        target_crs=_blank_to_none(args.target_crs),
    )

    workflow = MosaicWorkflow(job, logger=logger)
    workflow.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
