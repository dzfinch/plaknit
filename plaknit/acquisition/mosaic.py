"""Planet mosaic workflow orchestration and CLI."""

from __future__ import annotations

import argparse
from collections import Counter
import logging
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
from typing import Any, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

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


@dataclass(frozen=True)
class ProjectionInfo:
    """Projection metadata extracted from one raster."""

    path: Path
    crs: CRS
    label: str
    center_lonlat: Tuple[float, float]


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
        if job.ram <= 0:
            raise ValueError("RAM must be a positive integer.")
        if job.add_ndvi and job.sr_bands not in (4, 8):
            raise ValueError("sr_bands must be 4 or 8 when --ndvi is requested.")

        self._configure_environment()
        self._warn_on_proj_layout_mismatch()
        inputs = self._expand(job.inputs, label="inputs")
        tmpdir = self._prepare_tmpdir()
        with self._progress(enabled=self.log.isEnabledFor(logging.INFO)) as progress:
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
                if job.skip_masking:
                    masked_paths = inputs
                    if progress and mask_task is not None:
                        progress.update(mask_task, total=1, completed=1)
                else:
                    if not job.udms:
                        raise ValueError(
                            "UDM rasters are required unless --skip-masking is provided."
                        )
                    udms = self._expand(job.udms, label="UDMs")
                    if len(inputs) != len(udms):
                        raise ValueError(
                            "Input/UDM mismatch: expected "
                            f"{len(inputs)} UDMs but received {len(udms)}."
                        )
                    masked_paths = self._mask_inputs(inputs, udms, progress, mask_task)

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

    def _read_valid_georeferencing(self, path: Path) -> Optional[Tuple[CRS, Any]]:
        with rasterio.open(path) as source:
            source_crs = source.crs
            source_transform = source.transform
        if source_crs is None or source_transform.is_identity:
            return None
        return (source_crs, source_transform)

    def _ensure_masked_georeferencing(
        self, strip_path: Path, udm_path: Path, masked_path: Path
    ) -> None:

        with rasterio.open(masked_path) as masked:
            masked_crs = masked.crs
            masked_transform = masked.transform

        has_crs = masked_crs is not None
        has_transform = not masked_transform.is_identity
        if has_crs and has_transform:
            return

        source_candidate = self._read_valid_georeferencing(strip_path)
        repair_source = strip_path
        if source_candidate is None:
            source_candidate = self._read_valid_georeferencing(udm_path)
            repair_source = udm_path

        if source_candidate is None:
            raise ValueError(
                f"Masked raster '{masked_path}' is missing CRS/transform metadata and "
                f"cannot be repaired from strip '{strip_path}' or UDM '{udm_path}'. "
                "This often indicates mixed PROJ installations; set PROJ_DATA and "
                "PROJ_LIB to the same directory."
            )
        source_crs, source_transform = source_candidate

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
            repair_source,
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
        self._ensure_masked_georeferencing(strip_path, udm_path, masked_path)
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
            "Mask Planet strips with UDM rasters, mosaic them with OTB, "
            "and optionally append NDVI."
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
        target_crs=_blank_to_none(args.target_crs),
    )

    workflow = MosaicWorkflow(job, logger=logger)
    workflow.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
