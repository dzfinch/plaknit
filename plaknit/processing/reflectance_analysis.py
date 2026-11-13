"""Spatial-unit comparison of continuous rasters by land-cover class."""

from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, geometry_window
from rasterio.warp import reproject, transform_bounds, transform_geom
from rasterio.windows import Window, from_bounds
from scipy.stats import rankdata

from .landcover_analysis import _write_csv

MIN_INDEPENDENT_BLOCKS = 10
DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_PERMUTATION_REPLICATES = 9999


@dataclass
class ReflectanceBand:
    path: str
    band: int
    band_count: int


@dataclass
class ReflectanceInput:
    label: str
    classification: str
    rasters: Mapping[str, Union[str, ReflectanceBand]]


@dataclass
class SpatialUnit:
    unit_id: str
    geometry: object
    crs: rasterio.crs.CRS


@dataclass
class ReflectanceResult:
    outputs: List[Path]
    scaling: Dict[str, Dict[str, Dict[str, object]]]


def load_spatial_units(
    path: Union[str, Path], crs: Optional[str] = None
) -> List[SpatialUnit]:
    """Load unit polygons from a vector file or a bounds-based CSV."""
    source = Path(path)
    if source.suffix.lower() == ".csv":
        if not crs:
            raise ValueError(
                "CSV spatial units have no embedded CRS; provide "
                "--spatial-units-crs."
            )
        units = []
        with open(source, newline="", encoding="utf-8-sig") as stream:
            rows = csv.DictReader(stream)
            fields = set(rows.fieldnames or [])
            id_field = next(
                (name for name in ("unit_id", "square_id", "id") if name in fields),
                None,
            )
            bounds_fields = {"xmin", "ymin", "xmax", "ymax"}
            if id_field is None or not bounds_fields.issubset(fields):
                raise ValueError(
                    "Spatial-unit CSV must contain a unit_id (or square_id/id) "
                    "column and xmin, ymin, xmax, ymax bounds."
                )
            for row in rows:
                try:
                    bounds = tuple(
                        float(row[name]) for name in ("xmin", "ymin", "xmax", "ymax")
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError("Spatial-unit bounds must be numeric.") from exc
                from shapely.geometry import box

                units.append(
                    SpatialUnit(
                        str(row[id_field]),
                        box(*bounds),
                        rasterio.crs.CRS.from_user_input(crs),
                    )
                )
        return _validate_units(units)

    import geopandas as gpd

    frame = gpd.read_file(source)
    if frame.crs is None:
        raise ValueError(f"{source}: spatial-unit vector file must declare a CRS.")
    id_field = next(
        (name for name in ("unit_id", "square_id", "id") if name in frame.columns),
        None,
    )
    if id_field is None:
        raise ValueError(
            f"{source}: spatial-unit file needs a unit_id, square_id, or id field."
        )
    unit_crs = rasterio.crs.CRS.from_user_input(frame.crs)
    return _validate_units(
        [
            SpatialUnit(str(row[id_field]), row.geometry, unit_crs)
            for _, row in frame.iterrows()
        ]
    )


def _validate_units(units: Sequence[SpatialUnit]) -> List[SpatialUnit]:
    if not units:
        raise ValueError("At least one spatial unit is required.")
    ids = [unit.unit_id for unit in units]
    if any(not unit_id for unit_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Spatial-unit IDs must be non-empty and unique.")
    crs_values = {unit.crs for unit in units}
    if len(crs_values) != 1:
        raise ValueError("All spatial units must use the same CRS.")
    unit_crs = units[0].crs
    if not unit_crs.is_projected:
        raise ValueError("Spatial units must use a projected CRS with linear units.")
    for unit in units:
        if (
            unit.geometry is None
            or unit.geometry.is_empty
            or not unit.geometry.is_valid
            or not math.isfinite(float(unit.geometry.area))
            or unit.geometry.area <= 0
        ):
            raise ValueError(f"Spatial unit {unit.unit_id!r} has invalid geometry.")
    return list(units)


def _unit_geometry(unit: SpatialUnit, target_crs):
    if target_crs is None:
        raise ValueError("Classification rasters must declare a CRS.")
    from shapely.geometry import mapping, shape

    if unit.crs == target_crs:
        return unit.geometry
    return shape(transform_geom(unit.crs, target_crs, mapping(unit.geometry)))


def _unit_window(dataset, geometry):
    try:
        window = geometry_window(dataset, [geometry.__geo_interface__])
        window = window.round_offsets().round_lengths()
        window = window.intersection(Window(0, 0, dataset.width, dataset.height))
    except (ValueError, rasterio.errors.WindowError):
        return None, None
    if window.width <= 0 or window.height <= 0:
        return None, None
    transform = dataset.window_transform(window)
    inside = geometry_mask(
        [geometry.__geo_interface__],
        out_shape=(int(window.height), int(window.width)),
        transform=transform,
        all_touched=False,
        invert=True,
    )
    return window, inside


def _apply_scale_offset(values: np.ndarray, scale: float, offset: float) -> np.ndarray:
    if not (math.isfinite(scale) and math.isfinite(offset)) or scale <= 0:
        raise ValueError(
            "Raster scale/offset metadata must be finite with positive scale."
        )
    if scale != 1.0 or offset != 0.0:
        return values * scale + offset
    return values


def _scale_info(scale: float, offset: float) -> Dict[str, object]:
    if not (math.isfinite(scale) and math.isfinite(offset)) or scale <= 0:
        raise ValueError(
            "Raster scale/offset metadata must be finite with positive scale."
        )
    if scale == 1.0 and offset == 0.0:
        return {"method": "native", "scale": scale, "offset": offset}
    return {
        "method": "metadata-scale-offset",
        "scale": scale,
        "offset": offset,
    }


def _aligned_unit_values(
    source,
    band: int,
    target,
    target_window: Window,
) -> Tuple[np.ndarray, np.ndarray]:
    shape = (int(target_window.height), int(target_window.width))
    scale, offset = float(source.scales[band - 1]), float(source.offsets[band - 1])
    if (
        source.width == target.width
        and source.height == target.height
        and source.transform == target.transform
        and source.crs == target.crs
    ):
        raw = source.read(band, window=target_window).astype(np.float64)
        valid = (source.read_masks(band, window=target_window) > 0) & np.isfinite(raw)
        values = _apply_scale_offset(raw, scale, offset)
        return values, valid

    if source.crs is None or target.crs is None:
        raise ValueError("Source and classification grids need CRS values to align.")
    target_transform = target.window_transform(target_window)
    target_bounds = rasterio.windows.bounds(target_window, target.transform)
    source_bounds = transform_bounds(
        target.crs, source.crs, *target_bounds, densify_pts=21
    )
    source_window = from_bounds(*source_bounds, transform=source.transform)
    source_window = Window(
        math.floor(source_window.col_off) - 2,
        math.floor(source_window.row_off) - 2,
        math.ceil(source_window.width) + 4,
        math.ceil(source_window.height) + 4,
    )
    source_window = source_window.intersection(
        Window(0, 0, source.width, source.height)
    )
    if source_window.width <= 0 or source_window.height <= 0:
        return np.full(shape, np.nan), np.zeros(shape, dtype=bool)

    raw = source.read(band, window=source_window).astype(np.float64)
    source_valid = (source.read_masks(band, window=source_window) > 0) & np.isfinite(
        raw
    )
    values = _apply_scale_offset(raw, scale, offset)
    values[~source_valid] = np.nan
    aligned = np.full(shape, np.nan, dtype=np.float64)
    source_transform = source.window_transform(source_window)
    reproject(
        source=values,
        destination=aligned,
        src_transform=source_transform,
        src_crs=source.crs,
        src_nodata=np.nan,
        dst_transform=target_transform,
        dst_crs=target.crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    aligned_valid = np.zeros(shape, dtype=np.uint8)
    reproject(
        source=source_valid.astype(np.uint8),
        destination=aligned_valid,
        src_transform=source_transform,
        src_crs=source.crs,
        src_nodata=0,
        dst_transform=target_transform,
        dst_crs=target.crs,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    valid = (aligned_valid > 0) & np.isfinite(aligned)
    return aligned, valid


def _unit_summary(values: np.ndarray) -> Dict[str, object]:
    if values.size == 0:
        return {
            "n_valid_pixels": 0,
            "median": None,
            "lower_quartile": None,
            "upper_quartile": None,
        }
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {
        "n_valid_pixels": int(values.size),
        "median": float(median),
        "lower_quartile": float(q1),
        "upper_quartile": float(q3),
    }


def _paired_rank_biserial(differences: np.ndarray) -> Optional[float]:
    nonzero = differences[differences != 0]
    if not nonzero.size:
        return None
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / float(ranks.sum())


def _block_id(x_m: float, y_m: float, block_size_m: float) -> Tuple[int, int]:
    return math.floor(x_m / block_size_m), math.floor(y_m / block_size_m)


def _estimate_block_size(
    groups: Sequence[Tuple[np.ndarray, np.ndarray]],
    minimum_size_m: float,
) -> Dict[str, object]:
    """Estimate a pooled empirical variogram range for square-level differences."""
    distances = []
    semivariances = []
    usable_groups = 0
    for differences, coordinates in groups:
        if differences.size < 8 or np.var(differences) <= 1e-12:
            continue
        centered = differences - np.mean(differences)
        variance = float(np.mean(centered**2))
        delta_xy = coordinates[:, None, :] - coordinates[None, :, :]
        distance = np.sqrt(np.sum(delta_xy**2, axis=2))
        delta_value = centered[:, None] - centered[None, :]
        upper = np.triu(np.ones(distance.shape, dtype=bool), k=1) & (distance > 0)
        if not upper.any():
            continue
        distances.append(distance[upper])
        semivariances.append((0.5 * delta_value[upper] ** 2) / variance)
        usable_groups += 1
    if not distances:
        return {
            "status": "unresolved",
            "block_size_m": None,
            "reason": (
                "Not enough varying paired spatial units to estimate a "
                "variogram range."
            ),
            "usable_comparison_groups": 0,
        }

    all_distances = np.concatenate(distances)
    all_semivariances = np.concatenate(semivariances)
    edges = np.unique(np.quantile(all_distances, np.linspace(0, 1, 9)))
    if edges.size < 5:
        return {
            "status": "unresolved",
            "block_size_m": None,
            "reason": (
                "Spatial unit distances do not support enough distinct "
                "variogram lags."
            ),
            "usable_comparison_groups": usable_groups,
        }
    lag_values = []
    lag_counts = []
    lag_upper_edges = []
    for index in range(edges.size - 1):
        selected = (all_distances >= edges[index]) & (
            all_distances <= edges[index + 1]
            if index == edges.size - 2
            else all_distances < edges[index + 1]
        )
        if selected.sum() < 20:
            continue
        lag_values.append(float(np.mean(all_semivariances[selected])))
        lag_counts.append(int(selected.sum()))
        lag_upper_edges.append(float(edges[index + 1]))
    if len(lag_values) < 4:
        return {
            "status": "unresolved",
            "block_size_m": None,
            "reason": "Too few populated distance lags to estimate a stable range.",
            "usable_comparison_groups": usable_groups,
        }

    sill = float(np.mean(lag_values[-2:]))
    if not math.isfinite(sill) or sill <= 0:
        return {
            "status": "unresolved",
            "block_size_m": None,
            "reason": "Empirical variogram sill was not estimable.",
            "usable_comparison_groups": usable_groups,
        }
    candidate = None
    for index in range(1, len(lag_values) - 1):
        if lag_values[index] >= 0.95 * sill and all(
            value >= 0.75 * sill for value in lag_values[index + 1 :]
        ):
            candidate = lag_upper_edges[index]
            break
    if candidate is None:
        return {
            "status": "unresolved",
            "block_size_m": None,
            "reason": (
                "Empirical variogram did not show a stable sill within "
                "observed distances."
            ),
            "usable_comparison_groups": usable_groups,
            "lag_upper_edges_m": lag_upper_edges,
            "normalized_semivariance": lag_values,
            "lag_pair_counts": lag_counts,
        }
    candidate = max(float(candidate), minimum_size_m)
    return {
        "status": "estimated",
        "block_size_m": candidate,
        "method": (
            "First pooled empirical-semivariogram lag reaching 95% of the far-lag "
            "sill and remaining above 75%; floored at the largest unit width."
        ),
        "usable_comparison_groups": usable_groups,
        "lag_upper_edges_m": lag_upper_edges,
        "normalized_semivariance": lag_values,
        "lag_pair_counts": lag_counts,
    }


def _bootstrap_ci(
    differences: np.ndarray,
    block_ids: Sequence[Tuple[int, int]],
    rng: np.random.Generator,
    replicates: int,
) -> Tuple[float, float]:
    grouped: Dict[Tuple[int, int], List[float]] = {}
    for difference, block in zip(differences, block_ids):
        grouped.setdefault(block, []).append(float(difference))
    blocks = list(grouped.values())
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        draw = rng.integers(0, len(blocks), size=len(blocks))
        sample = np.concatenate([np.asarray(blocks[i]) for i in draw])
        estimates[index] = float(np.median(sample))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def _block_permutation_pvalue(
    differences: np.ndarray,
    block_ids: Sequence[Tuple[int, int]],
    rng: np.random.Generator,
    monte_carlo_replicates: int,
) -> Tuple[float, str, int]:
    blocks = sorted(set(block_ids))
    block_lookup = {block: index for index, block in enumerate(blocks)}
    observation_blocks = np.asarray([block_lookup[block] for block in block_ids])
    observed = abs(float(np.median(differences)))
    tolerance = np.finfo(float).eps * max(1.0, observed) * 8
    if len(blocks) <= 15:
        extreme = 0
        total = 1 << len(blocks)
        for code in range(total):
            signs = np.fromiter(
                (1.0 if code & (1 << bit) else -1.0 for bit in range(len(blocks))),
                dtype=np.float64,
                count=len(blocks),
            )
            statistic = abs(float(np.median(differences * signs[observation_blocks])))
            extreme += statistic + tolerance >= observed
        return extreme / total, "exact block sign-flip permutation", total

    extreme = 0
    for _ in range(monte_carlo_replicates):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(blocks))
        statistic = abs(float(np.median(differences * signs[observation_blocks])))
        extreme += statistic + tolerance >= observed
    return (
        (extreme + 1) / (monte_carlo_replicates + 1),
        "Monte Carlo block sign-flip permutation (plus-one correction)",
        monte_carlo_replicates,
    )


def _benjamini_hochberg(rows: List[Dict[str, object]]) -> int:
    tested = sorted(
        (float(row["p_value"]), index)
        for index, row in enumerate(rows)
        if row["p_value"] is not None
    )
    count = len(tested)
    running = 1.0
    for rank in range(count, 0, -1):
        p_value, row_index = tested[rank - 1]
        running = min(running, p_value * count / rank)
        rows[row_index]["p_value_bh"] = min(1.0, running)
    for row in rows:
        row.setdefault("p_value_bh", None)
    return count


def _methods_report(
    path: Path,
    comparisons: Sequence[Mapping[str, object]],
    settings: Mapping[str, object],
) -> None:
    tested = [row for row in comparisons if row.get("status") == "tested"]
    lines = [
        "# Spatial reflectance comparison",
        "",
        "## Methods",
        "",
        "Each supplied spatial polygon is one co-located sampling unit. For each "
        "input, raster name, class, and unit, valid source-raster values are "
        "summarized by their median on that input's own classification mask. "
        "Comparisons pair those summaries by unit ID and use the median of "
        "`input B - input A` differences as the estimate.",
        "",
        "Declared raster scale/offset metadata is applied. When it is absent, "
        "native raster values are retained; inputs are not independently "
        "min–max stretched. Units and physical comparability must still be "
        "confirmed by the analyst.",
        "",
        "Confidence intervals use a percentile bootstrap resampling whole "
        "coordinate-based spatial blocks. P-values use a paired block "
        "sign-flip permutation, exact for at most 15 blocks and otherwise "
        "Monte Carlo with a plus-one correction. Benjamini–Hochberg correction "
        "is applied across all comparisons with justified p-values in this run.",
        "",
        "## Assumptions and limitations",
        "",
        "- Spatial blocks are treated as independent; the empirical variogram "
        "estimate is based on the observed square-level differences and can be "
        "unstable with few or widely spaced units. Sign-flip inference additionally "
        "assumes block-level sign exchangeability under the null.",
        "- The class code is taken from each input's own classification. Matching "
        "codes are assumed to have comparable meanings; they are not validated "
        "against reference labels.",
        "- Results describe these raster products. Acquisition dates, scene IDs, "
        "processing histories, and quality-mask provenance were not supplied, "
        "so differences cannot be attributed to sensor alone.",
        "- Rows with too few independent blocks or unresolved block size retain "
        "descriptive estimates but do not receive inferential p-values or "
        "confidence intervals.",
        "",
        "## Run settings",
        "",
        f"- Seed: `{settings['seed']}`",
        f"- Spatial block size (m): `{settings['block_size_m']}` "
        f"({settings['block_size_source']})",
        f"- Minimum independent blocks for inference: "
        f"`{settings['minimum_independent_blocks']}`",
        f"- Bootstrap replicates: `{settings['bootstrap_replicates']}`",
        f"- Monte Carlo permutation replicates: "
        f"`{settings['permutation_replicates']}`",
        f"- Supplied spatial units: `{settings['spatial_unit_count']}`",
        f"- Inferential comparisons: `{len(tested)}` of `{len(comparisons)}`",
        "",
        "## Key results",
        "",
        "| Input A | Input B | Raster | Class | Paired units | Blocks | "
        "Median B − A | 95% CI | p | BH p | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparisons:
        estimate = (
            f"{float(row['median_difference_b_minus_a']):.6g}"
            if row["median_difference_b_minus_a"] is not None
            else ""
        )
        interval = (
            f"[{float(row['ci_lower']):.6g}, {float(row['ci_upper']):.6g}]"
            if row["ci_lower"] is not None and row["ci_upper"] is not None
            else ""
        )
        p_value = f"{float(row['p_value']):.6g}" if row["p_value"] is not None else ""
        p_value_bh = (
            f"{float(row['p_value_bh']):.6g}" if row["p_value_bh"] is not None else ""
        )
        block_count = (
            str(row["independent_block_count"])
            if row["independent_block_count"] is not None
            else ""
        )
        lines.append(
            f"| {row['input_a']} | {row['input_b']} | {row['raster']} | "
            f"{row['class_code']} | {row['paired_unit_count']} | {block_count} | "
            f"{estimate} | {interval} | {p_value} | {p_value_bh} | {row['status']} |"
        )
    if not tested:
        lines.extend(
            [
                "",
                "No comparisons met the independent-block and data requirements "
                "for inferential results; estimates above are descriptive only.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_reflectance(
    inputs: Sequence[ReflectanceInput],
    out_dir: Union[str, Path],
    spatial_units: Union[str, Path, Sequence[SpatialUnit]],
    prefix: str = "reflectance",
    seed: int = 42,
    classes: Optional[Iterable[int]] = None,
    spatial_units_crs: Optional[str] = None,
    block_size_m: Optional[float] = None,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    permutation_replicates: int = DEFAULT_PERMUTATION_REPLICATES,
    min_independent_blocks: int = MIN_INDEPENDENT_BLOCKS,
) -> ReflectanceResult:
    if len(inputs) < 2:
        raise ValueError("At least two classification inputs are required.")
    labels = [item.label for item in inputs]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("Input labels must be non-empty and unique.")
    raster_names = {name for item in inputs for name in item.rasters}
    if not raster_names:
        raise ValueError("At least one --raster is required.")
    for name in raster_names:
        if sum(name in item.rasters for item in inputs) < 2:
            raise ValueError(
                f"Raster {name!r} must be supplied for at least two inputs."
            )
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be a non-empty filename prefix, not a path.")
    if bootstrap_replicates < 1 or permutation_replicates < 1:
        raise ValueError("Bootstrap and permutation replicate counts must be positive.")
    if min_independent_blocks < 2:
        raise ValueError("At least two independent blocks are required.")
    if seed is None:
        raise ValueError("seed must be an integer for reproducible resampling.")
    if block_size_m is not None and (
        not math.isfinite(block_size_m) or block_size_m <= 0
    ):
        raise ValueError("block_size_m must be a positive finite distance.")

    units = (
        load_spatial_units(spatial_units, spatial_units_crs)
        if isinstance(spatial_units, (str, Path))
        else _validate_units(spatial_units)
    )
    unit_crs = units[0].crs
    unit_scale = float(unit_crs.linear_units_factor[1])
    if not math.isfinite(unit_scale) or unit_scale <= 0:
        raise ValueError("Spatial-unit CRS must define a linear unit conversion.")
    unit_centers = {
        unit.unit_id: (
            float(unit.geometry.centroid.x) * unit_scale,
            float(unit.geometry.centroid.y) * unit_scale,
        )
        for unit in units
    }
    unit_width_m = max(
        max(
            unit.geometry.bounds[2] - unit.geometry.bounds[0],
            unit.geometry.bounds[3] - unit.geometry.bounds[1],
        )
        * unit_scale
        for unit in units
    )
    if block_size_m is not None and block_size_m < unit_width_m:
        raise ValueError(
            f"block_size_m must be at least the largest spatial-unit width "
            f"({unit_width_m:.6g} m)."
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    class_filter = None if classes is None else {int(code) for code in classes}
    if class_filter is not None and not class_filter:
        raise ValueError("At least one class code is required when filtering.")

    unit_records: Dict[Tuple[str, str, int, str], Dict[str, object]] = {}
    all_codes = set()
    scaling: Dict[str, Dict[str, Dict[str, object]]] = {
        item.label: {} for item in inputs
    }
    with ExitStack() as stack:
        class_sources = [
            stack.enter_context(rasterio.open(item.classification)) for item in inputs
        ]
        raster_sources: Dict[Tuple[int, str], Tuple[object, int]] = {}
        for index, item in enumerate(inputs):
            for name, raster in item.rasters.items():
                spec = (
                    ReflectanceBand(raster, 1, 1) if isinstance(raster, str) else raster
                )
                if not 1 <= spec.band <= spec.band_count:
                    raise ValueError(
                        f"{item.label}:{name}: band index must be within the "
                        "declared band count."
                    )
                source = stack.enter_context(rasterio.open(spec.path))
                if source.count != spec.band_count:
                    raise ValueError(
                        f"{spec.path}: {item.label}:{name} declares "
                        f"{spec.band_count} band name(s), but raster has "
                        f"{source.count} band(s)."
                    )
                raster_sources[(index, name)] = (source, spec.band)
                scale = float(source.scales[spec.band - 1])
                offset = float(source.offsets[spec.band - 1])
                scaling[item.label][name] = {
                    "path": spec.path,
                    "band": spec.band,
                    **_scale_info(scale, offset),
                }

        for unit in units:
            for index, item in enumerate(inputs):
                classification = class_sources[index]
                geometry = _unit_geometry(unit, classification.crs)
                window, inside = _unit_window(classification, geometry)
                if window is None:
                    continue
                class_values = classification.read(1, window=window)
                class_valid = (
                    (classification.read_masks(1, window=window) > 0)
                    & np.isfinite(class_values)
                    & inside
                )
                if not class_valid.any():
                    continue
                rounded = np.rint(class_values[class_valid])
                if not np.allclose(class_values[class_valid], rounded):
                    raise ValueError(
                        f"{item.label}: classification contains non-integer "
                        "class codes."
                    )
                present_codes = {int(code) for code in np.unique(rounded)}
                all_codes.update(present_codes)
                codes = (
                    sorted(present_codes & class_filter)
                    if class_filter is not None
                    else sorted(present_codes)
                )
                for name in item.rasters:
                    source, band = raster_sources[(index, name)]
                    aligned, raster_valid = _aligned_unit_values(
                        source, band, classification, window
                    )
                    usable = raster_valid & inside & class_valid
                    for code in codes:
                        selected = aligned[usable & (class_values == code)]
                        unit_records[(item.label, name, code, unit.unit_id)] = {
                            "input_label": item.label,
                            "raster": name,
                            "class_code": code,
                            "unit_id": unit.unit_id,
                            "center_x": unit_centers[unit.unit_id][0],
                            "center_y": unit_centers[unit.unit_id][1],
                            "scaling_method": scaling[item.label][name]["method"],
                            **_unit_summary(selected),
                        }

    if class_filter is not None:
        missing = class_filter - all_codes
        if missing:
            raise ValueError(
                "Requested class code(s) not found in spatial-unit footprints: "
                + ", ".join(str(code) for code in sorted(missing))
            )
    class_codes = sorted(all_codes if class_filter is None else class_filter)

    # Emit explicit empty unit/class rows so missingness remains auditable.
    for item in inputs:
        for name in item.rasters:
            for unit in units:
                for code in class_codes:
                    key = (item.label, name, code, unit.unit_id)
                    unit_records.setdefault(
                        key,
                        {
                            "input_label": item.label,
                            "raster": name,
                            "class_code": code,
                            "unit_id": unit.unit_id,
                            "center_x": unit_centers[unit.unit_id][0],
                            "center_y": unit_centers[unit.unit_id][1],
                            "scaling_method": scaling[item.label][name]["method"],
                            **_unit_summary(np.asarray([])),
                        },
                    )

    unit_rows = sorted(
        unit_records.values(),
        key=lambda row: (
            row["raster"],
            row["class_code"],
            row["input_label"],
            row["unit_id"],
        ),
    )
    unit_path = out / f"{prefix}_spatial_unit_summary.csv"
    _write_csv(unit_path, unit_rows)

    grouped_units: Dict[Tuple[str, str, int], List[Mapping[str, object]]] = {}
    for row in unit_rows:
        grouped_units.setdefault(
            (str(row["input_label"]), str(row["raster"]), int(row["class_code"])), []
        ).append(row)
    class_rows = []
    for (label, name, code), rows in sorted(grouped_units.items()):
        populated = [row for row in rows if row["median"] is not None]
        medians = np.asarray([float(row["median"]) for row in populated])
        q1 = median = q3 = None
        if medians.size:
            q1, median, q3 = (
                float(value) for value in np.percentile(medians, [25, 50, 75])
            )
        class_rows.append(
            {
                "input_label": label,
                "raster": name,
                "class_code": code,
                "spatial_unit_count": int(medians.size),
                "valid_pixel_count": int(
                    sum(int(row["n_valid_pixels"]) for row in populated)
                ),
                "median_of_unit_medians": median,
                "lower_quartile_of_unit_medians": q1,
                "upper_quartile_of_unit_medians": q3,
                "scaling_method": scaling[label][name]["method"],
            }
        )
    class_path = out / f"{prefix}_class_summary.csv"
    _write_csv(class_path, class_rows)

    difference_groups = []
    comparison_data = []
    for left, right in itertools.combinations(range(len(inputs)), 2):
        a_item, b_item = inputs[left], inputs[right]
        for name in sorted(set(a_item.rasters) & set(b_item.rasters)):
            for code in class_codes:
                a_by_unit = {
                    str(row["unit_id"]): row
                    for row in grouped_units.get((a_item.label, name, code), [])
                }
                b_by_unit = {
                    str(row["unit_id"]): row
                    for row in grouped_units.get((b_item.label, name, code), [])
                }
                matched = sorted(
                    unit_id
                    for unit_id in set(a_by_unit) & set(b_by_unit)
                    if a_by_unit[unit_id]["median"] is not None
                    and b_by_unit[unit_id]["median"] is not None
                )
                differences = np.asarray(
                    [
                        float(b_by_unit[unit_id]["median"])
                        - float(a_by_unit[unit_id]["median"])
                        for unit_id in matched
                    ],
                    dtype=np.float64,
                )
                coordinates = np.asarray(
                    [unit_centers[unit_id] for unit_id in matched], dtype=np.float64
                ).reshape((-1, 2))
                if differences.size >= 8:
                    difference_groups.append((differences, coordinates))
                comparison_data.append(
                    (
                        a_item.label,
                        b_item.label,
                        name,
                        code,
                        matched,
                        differences,
                    )
                )

    range_estimate = _estimate_block_size(difference_groups, unit_width_m)
    if block_size_m is None:
        effective_block_size = range_estimate["block_size_m"]
        block_source = "empirical_variogram" if effective_block_size else "unresolved"
    else:
        effective_block_size = float(block_size_m)
        block_source = "user_override"
    block_map = (
        {
            unit.unit_id: _block_id(
                unit_centers[unit.unit_id][0],
                unit_centers[unit.unit_id][1],
                effective_block_size,
            )
            for unit in units
        }
        if effective_block_size is not None
        else {}
    )

    rng = np.random.default_rng(seed)
    comparison_rows = []
    for label_a, label_b, name, code, matched, differences in comparison_data:
        row: Dict[str, object] = {
            "input_a": label_a,
            "input_b": label_b,
            "raster": name,
            "class_code": code,
            "paired_unit_count": int(differences.size),
            "independent_block_count": None,
            "difference_direction": "input B minus input A",
            "median_difference_b_minus_a": (
                float(np.median(differences)) if differences.size else None
            ),
            "paired_rank_biserial_effect": (
                _paired_rank_biserial(differences) if differences.size else None
            ),
            "ci_level": 0.95,
            "ci_lower": None,
            "ci_upper": None,
            "p_value": None,
            "p_value_bh": None,
            "permutation_method": None,
            "permutation_count": None,
            "status": None,
        }
        if not differences.size:
            row["status"] = "no_paired_units"
        elif effective_block_size is None:
            row["status"] = "block_size_unresolved"
        else:
            selected_blocks = [block_map[unit_id] for unit_id in matched]
            block_count = len(set(selected_blocks))
            row["independent_block_count"] = block_count
            if block_count < min_independent_blocks:
                row["status"] = "insufficient_independent_blocks"
            else:
                low, high = _bootstrap_ci(
                    differences, selected_blocks, rng, bootstrap_replicates
                )
                p_value, permutation_method, permutation_count = (
                    _block_permutation_pvalue(
                        differences,
                        selected_blocks,
                        rng,
                        permutation_replicates,
                    )
                )
                row.update(
                    {
                        "ci_lower": low,
                        "ci_upper": high,
                        "p_value": p_value,
                        "permutation_method": permutation_method,
                        "permutation_count": permutation_count,
                        "status": "tested",
                    }
                )
        comparison_rows.append(row)
    bh_count = _benjamini_hochberg(comparison_rows)
    comparison_path = out / f"{prefix}_spatial_comparisons.csv"
    _write_csv(comparison_path, comparison_rows)

    block_settings = {
        "block_size_m": effective_block_size,
        "block_size_source": block_source,
        "minimum_independent_blocks": min_independent_blocks,
        "range_estimate": range_estimate,
    }
    settings = {
        "seed": seed,
        "spatial_unit_count": len(units),
        "block_size_m": effective_block_size,
        "block_size_source": block_source,
        "minimum_independent_blocks": min_independent_blocks,
        "bootstrap_replicates": bootstrap_replicates,
        "permutation_replicates": permutation_replicates,
    }
    report_path = out / f"{prefix}_methods_report.md"
    _methods_report(report_path, comparison_rows, settings)

    unit_crs = units[0].crs
    metadata = {
        "inputs": [
            {
                "label": item.label,
                "classification": item.classification,
                "rasters": scaling[item.label],
            }
            for item in inputs
        ],
        "spatial_units": {
            "count": len(units),
            "id_field": "unit_id (normalized from unit_id, square_id, or id)",
            "crs": unit_crs.to_string(),
            "geometry_source": (
                str(spatial_units)
                if isinstance(spatial_units, (str, Path))
                else "Python SpatialUnit sequence"
            ),
            "pairing": "same spatial-unit ID and class code across input products",
        },
        "class_filter": sorted(class_filter) if class_filter is not None else None,
        "value_scaling": (
            "Apply declared raster scale/offset metadata; otherwise preserve native "
            "values. No independent min-max scaling is applied."
        ),
        "alignment": (
            "Read each input on its own classification grid; align continuous "
            "rasters by bilinear resampling and their validity masks by nearest "
            "neighbor only when grids differ."
        ),
        "estimand": (
            "Median across paired spatial units of (input B unit-level class "
            "median minus input A unit-level class median)."
        ),
        "effect_size": (
            "Paired rank-biserial effect from signed ranks of nonzero unit "
            "differences; positive indicates input B tends to be higher. The "
            "primary estimate remains the median difference in source units."
        ),
        "uncertainty": {
            **block_settings,
            "bootstrap_replicates": bootstrap_replicates,
            "confidence_interval": "95% percentile bootstrap over whole blocks",
            "permutation": (
                "paired sign-flip of whole block clusters; exact up to 15 blocks, "
                "otherwise Monte Carlo with plus-one correction"
            ),
            "bh_family_size": bh_count,
            "p_value_underflow": (
                "Permutation p-values have finite resolution and plus-one "
                "correction for Monte Carlo; p-values are never reported as zero."
            ),
        },
        "acquisition_metadata": {
            "available": False,
            "limitation": (
                "Acquisition dates, scene IDs, processing histories, and quality "
                "mask provenance were not available; comparisons are product-level "
                "and cannot isolate sensor effects."
            ),
        },
        "class_code_assumption": (
            "Matching class codes across input classifications are assumed to "
            "have comparable meanings; this is not independently verified."
        ),
        "seed": seed,
        "comparison_count": len(comparison_rows),
        "tested_comparison_count": sum(
            row["status"] == "tested" for row in comparison_rows
        ),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
        },
    }
    metadata_path = out / f"{prefix}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    outputs = [
        class_path,
        unit_path,
        comparison_path,
        report_path,
        metadata_path,
    ]
    return ReflectanceResult(outputs, scaling)
