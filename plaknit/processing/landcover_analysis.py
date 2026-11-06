"""Mode-specific spatial and temporal land-cover analyses."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import rasterio
from scipy import ndimage

from . import landcover_compare as lc


@dataclass
class InputSpec:
    label: str
    classification: str
    date: Optional[str] = None


@dataclass
class AnalysisResult:
    selection: lc.SelectionResult
    outputs: List[Path]


def _validate_specs(
    mode: str,
    inputs: Sequence[InputSpec],
    group_by: str,
    patch_min_area_m2: float,
) -> List[date]:
    if mode not in ("spatial", "temporal"):
        raise ValueError("mode must be 'spatial' or 'temporal'.")
    if len(inputs) < 2:
        raise ValueError("At least two classification inputs are required.")
    labels = [item.label for item in inputs]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("Input labels must be non-empty and unique.")
    dates: List[date] = []
    if mode == "temporal":
        if any(not item.date for item in inputs):
            raise ValueError("Temporal mode requires a date for every input.")
        try:
            dates = [
                date.fromisoformat(item.date)
                for item in inputs
                if item.date is not None
            ]
        except ValueError as exc:
            raise ValueError("Dates must use the ISO YYYY-MM-DD format.") from exc
        if len(set(dates)) != len(dates):
            raise ValueError(
                "Temporal mode requires a unique acquisition date per input."
            )
        if len(set(dates)) < 2:
            raise ValueError(
                "Temporal mode requires at least two distinct dates."
            )
        if group_by not in ("date", "year"):
            raise ValueError("Temporal group_by must be 'date' or 'year'.")
    if patch_min_area_m2 < 0:
        raise ValueError("--patch-min-area-m2 cannot be negative.")
    return dates


def _cell_centers(layer: lc.Layer) -> Tuple[np.ndarray, np.ndarray]:
    xs = layer.left + (np.arange(layer.width, dtype=float) + 0.5) * layer.res_x
    ys = layer.top - (np.arange(layer.height, dtype=float) + 0.5) * layer.res_y
    return xs, ys


def _native_common_mask(layers: Sequence[lc.Layer], index: int) -> np.ndarray:
    """Keep native cell centroids in common bounds and valid in every input."""
    layer = layers[index]
    xmin, ymin, xmax, ymax = lc.common_bounds(layers)
    xs, ys = _cell_centers(layer)
    inside_bounds = ((xs >= xmin) & (xs < xmax))[None, :] & (
        (ys <= ymax) & (ys > ymin)
    )[:, None]
    common = layer.valid & inside_bounds
    for other_index, other in enumerate(layers):
        if other_index == index:
            continue
        cols = np.floor((xs - other.left) / other.res_x).astype(np.int64)
        rows = np.floor((other.top - ys) / other.res_y).astype(np.int64)
        inside_x = (cols >= 0) & (cols < other.width)
        inside_y = (rows >= 0) & (rows < other.height)
        cols = np.clip(cols, 0, other.width - 1)
        rows = np.clip(rows, 0, other.height - 1)
        common &= (
            inside_y[:, None]
            & inside_x[None, :]
            & other.valid[rows[:, None], cols[None, :]]
        )
    return common


def _pylandstats_metrics(
    layer: lc.Layer,
    mask: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    input_label: str,
    square_id: Optional[int],
) -> List[Dict[str, object]]:
    import pylandstats

    if square_id is None:
        r0, r1, c0, c1 = lc.intersect_window(layer, xmin, ymin, xmax, ymax)
    else:
        r0, r1, c0, c1 = lc.centroid_window(layer, xmin, ymin, xmax, ymax)
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, layer.height - 1), min(c1, layer.width - 1)
    if r0 > r1 or c0 > c1:
        return []
    selected = mask[r0 : r1 + 1, c0 : c1 + 1]
    if not selected.any():
        return []
    data = layer.data[r0 : r1 + 1, c0 : c1 + 1]
    classes = sorted(int(value) for value in np.unique(data[selected]))
    nodata = np.iinfo(np.int64).min
    while nodata in classes:
        nodata += 1
    array = np.asarray(data, dtype=np.int64).copy()
    array[~selected] = nodata
    landscape = pylandstats.Landscape(
        array,
        res=(layer.res_x, layer.res_y),
        nodata=nodata,
        neighborhood_rule="4",
    )
    class_metrics = landscape.compute_class_metrics_df(
        metrics=["proportion_of_landscape", "patch_density"],
        classes=classes,
        metrics_kwargs={"patch_density": {"percent": False, "hectares": True}},
    )
    landscape_metrics = landscape.compute_landscape_metrics_df(
        metrics=["edge_density"],
        metrics_kwargs={
            "edge_density": {"count_boundary": False, "hectares": True}
        },
    )
    valid_area = float(selected.sum() * layer.cell_area)
    common = {
        "input_label": input_label,
        "square_id": square_id,
        "resolution_x_m": layer.res_x,
        "resolution_y_m": layer.res_y,
        "valid_area_m2": valid_area,
        "valid_cell_count": int(selected.sum()),
        "boundary_rule": "4-neighbor; outer landscape and NoData borders excluded",
    }
    rows: List[Dict[str, object]] = []
    present_metrics = {
        int(class_val): row for class_val, row in class_metrics.iterrows()
    }
    for class_val in sorted(
        set(int(value) for value in np.unique(layer.data[layer.valid]))
    ):
        metric = present_metrics.get(class_val)
        rows.append(
            {
                **common,
                "metric_level": "class",
                "class_code": class_val,
                "class_status": "present" if metric is not None else "absent",
                "proportion_of_landscape_pct": (
                    float(metric["proportion_of_landscape"])
                    if metric is not None
                    else None
                ),
                "patch_density_patches_per_ha": (
                    float(metric["patch_density"])
                    if metric is not None
                    else None
                ),
                "landscape_edge_density_m_per_ha": None,
                "metric_definitions": (
                    "proportion_of_landscape = class area / valid area * 100; "
                    "patch density = patches / valid hectares"
                ),
            }
        )
    rows.append(
        {
            **common,
            "metric_level": "landscape",
            "class_code": None,
            "class_status": "defined",
            "proportion_of_landscape_pct": None,
            "patch_density_patches_per_ha": None,
            "landscape_edge_density_m_per_ha": float(
                landscape_metrics.iloc[0]["edge_density"]
            ),
            "metric_definitions": (
                "edge density = unlike shared 4-neighbor edge length "
                "/ valid hectares; "
                "landscape and NoData boundaries excluded"
            ),
        }
    )
    return rows


def _all_metric_rows(
    layers: Sequence[lc.Layer],
    common_masks: Sequence[np.ndarray],
    squares: Sequence[Tuple[float, float]],
    width_m: float,
    inputs: Sequence[InputSpec],
    mode: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for i, layer in enumerate(layers):
        half = width_m / 2.0
        for square_id, (cx, cy) in enumerate(squares, start=1):
            rows.extend(
                _pylandstats_metrics(
                    layer,
                    common_masks[i],
                    cx - half,
                    cy - half,
                    cx + half,
                    cy + half,
                    layer.label,
                    square_id,
                )
            )
    return rows


CLASS_METRIC_FIELDS = [
    "input_label",
    "class_code",
    "area_m2",
    "proportion_pct",
    "patch_count",
    "patch_density",
]


def _class_metric_rows(
    layers: Sequence[lc.Layer],
    common_masks: Sequence[np.ndarray],
) -> List[Dict[str, object]]:
    import pylandstats

    bounds = lc.common_bounds(layers)
    all_classes = sorted(
        {
            int(value)
            for layer, mask in zip(layers, common_masks)
            for value in np.unique(layer.data[mask])
        }
    )
    rows: List[Dict[str, object]] = []
    for layer, mask in zip(layers, common_masks):
        r0, r1, c0, c1 = lc.intersect_window(layer, *bounds)
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, layer.height - 1), min(c1, layer.width - 1)
        selected = mask[r0 : r1 + 1, c0 : c1 + 1]
        data = np.asarray(
            layer.data[r0 : r1 + 1, c0 : c1 + 1], dtype=np.int64
        ).copy()
        nodata = np.iinfo(np.int64).min
        present = set(int(value) for value in np.unique(data[selected]))
        while nodata in present:
            nodata += 1
        data[~selected] = nodata
        landscape = pylandstats.Landscape(
            data,
            res=(layer.res_x, layer.res_y),
            nodata=nodata,
            neighborhood_rule="4",
        )
        metrics = landscape.compute_class_metrics_df(
            metrics=[
                "proportion_of_landscape",
                "number_of_patches",
                "patch_density",
            ],
            classes=sorted(present),
            metrics_kwargs={
                "patch_density": {"percent": False, "hectares": True}
            },
        )
        for class_code in all_classes:
            if class_code in present:
                metric = metrics.loc[class_code]
                proportion = float(metric["proportion_of_landscape"])
                count = int(metric["number_of_patches"])
                density = float(metric["patch_density"])
                area = float(
                    np.count_nonzero(data[selected] == class_code)
                    * layer.cell_area
                )
            else:
                proportion, count, density, area = 0.0, 0, 0.0, 0.0
            rows.append(
                {
                    "input_label": layer.label,
                    "class_code": class_code,
                    "area_m2": area,
                    "proportion_pct": proportion,
                    "patch_count": count,
                    "patch_density": density,
                }
            )
    return rows


def _group_name(item: InputSpec, mode: str, group_by: str) -> str:
    if mode == "spatial":
        return item.label
    assert item.date is not None
    if group_by == "year":
        return str(date.fromisoformat(item.date).year)
    return item.date


def _agreement_summaries(
    layers: Sequence[lc.Layer],
    finest_index: int,
    agreement_path: Path,
    centers: Sequence[Tuple[float, float]],
    width_m: float,
    out: Path,
    prefix: str,
    patch_min_area_m2: float,
) -> List[Path]:
    fine = layers[finest_index]
    with rasterio.open(agreement_path) as src:
        data = src.read(1)
    valid = data > 0
    count_valid = int(valid.sum())
    if count_valid == 0:
        raise ValueError("The agreement raster has no valid common cells.")
    cell_area = fine.cell_area
    ids, counts = np.unique(data[valid], return_counts=True)
    full_count = int(counts[ids == 1].sum())
    disagreement_count = count_valid - full_count

    square_rows = []
    half = width_m / 2.0
    for square_id, (cx, cy) in enumerate(centers, start=1):
        r0, r1, c0, c1 = lc.centroid_window(
            fine, cx - half, cy - half, cx + half, cy + half
        )
        block = data[
            max(0, r0) : min(fine.height - 1, r1) + 1,
            max(0, c0) : min(fine.width - 1, c1) + 1,
        ]
        square_valid = block[block > 0]
        if square_valid.size:
            patterns, pattern_counts = np.unique(
                square_valid, return_counts=True
            )
            order = np.argsort(pattern_counts)[::-1][:3]
            leading = []
            for j in order:
                pattern_id = int(patterns[j])
                sequence = lc.format_sequence(
                    lc.sequence_from_id(pattern_id, len(layers))
                )
                leading.append(f"{pattern_id}:{sequence}")
            full = int(np.count_nonzero(square_valid == 1))
            square_rows.append(
                {
                    "square_id": square_id,
                    "valid_cell_count": int(square_valid.size),
                    "full_agreement_fraction": float(full / square_valid.size),
                    "disagreement_fraction": float(
                        1 - full / square_valid.size
                    ),
                    "leading_patterns": "|".join(leading),
                    "valid_area_m2": float(square_valid.size * cell_area),
                }
            )
        else:
            square_rows.append(
                {
                    "square_id": square_id,
                    "valid_cell_count": 0,
                    "full_agreement_fraction": None,
                    "disagreement_fraction": None,
                    "leading_patterns": "",
                    "valid_area_m2": 0.0,
                }
            )
    square_csv = out / f"{prefix}_agreement_squares.csv"
    _write_csv(
        square_csv,
        square_rows,
        [
            "square_id",
            "valid_cell_count",
            "full_agreement_fraction",
            "disagreement_fraction",
            "leading_patterns",
            "valid_area_m2",
        ],
    )
    import geopandas as gpd
    from shapely.geometry import box

    square_geometries = [
        box(cx - half, cy - half, cx + half, cy + half) for cx, cy in centers
    ]
    if square_rows:
        square_gdf = gpd.GeoDataFrame(
            square_rows, geometry=square_geometries, crs=fine.crs
        )
    else:
        square_gdf = gpd.GeoDataFrame(
            {
                "square_id": [],
                "valid_cell_count": [],
                "full_agreement_fraction": [],
                "disagreement_fraction": [],
                "leading_patterns": [],
                "valid_area_m2": [],
                "geometry": [],
            },
            geometry="geometry",
            crs=fine.crs,
        )
    square_gpkg = out / f"{prefix}_agreement_squares.gpkg"
    square_gdf.to_file(square_gpkg, driver="GPKG")

    components, n_components = ndimage.label(
        valid & (data != 1),
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    )
    sizes = np.bincount(components.ravel(), minlength=n_components + 1)
    patch_count = int(
        sum(
            1
            for component_id in range(1, n_components + 1)
            if sizes[component_id] * cell_area + lc.EPS >= patch_min_area_m2
        )
    )
    overview_csv = out / f"{prefix}_agreement_overview.csv"
    _write_csv(
        overview_csv,
        [
            {
                "valid_area_m2": float(count_valid * cell_area),
                "full_agreement_area_m2": float(full_count * cell_area),
                "full_agreement_fraction": float(full_count / count_valid),
                "disagreement_area_m2": float(disagreement_count * cell_area),
                "disagreement_fraction": float(
                    disagreement_count / count_valid
                ),
                "distinct_pattern_count": int(ids.size),
                "disagreement_patch_count": patch_count,
            }
        ],
    )
    return [overview_csv, square_csv, square_gpkg]

def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    keys = (
        list(fieldnames)
        if fieldnames is not None
        else list(dict.fromkeys(key for row in rows for key in row))
    )
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        if keys:
            writer.writeheader()
            writer.writerows(rows)


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return result or "band"


def run_analysis(
    mode: str,
    inputs: Sequence[InputSpec],
    out_dir,
    prefix: str = "compare",
    width_m: float = 100.0,
    n_samples: int = 100,
    seed: Optional[int] = 42,
    search_limit: Optional[int] = None,
    reproject_to_finest: bool = True,
    classes: Optional[Iterable[int]] = None,
    n_jobs: Optional[int] = 1,
    block_rows: int = 512,
    group_by: str = "date",
    patch_min_area_m2: float = 0.0,
) -> AnalysisResult:
    _validate_specs(mode, inputs, group_by, patch_min_area_m2)
    if not prefix or Path(prefix).name != prefix:
        raise ValueError(
            "prefix must be a non-empty filename prefix, not a path."
        )
    class_filter = None if classes is None else list(classes)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    comparison = lc.compare_landcover(
        [item.classification for item in inputs],
        out,
        width_m=width_m,
        n_samples=n_samples,
        seed=seed,
        search_limit=search_limit,
        reproject_to_finest=reproject_to_finest,
        classes=class_filter,
        prefix=prefix,
        n_jobs=n_jobs,
        block_rows=block_rows,
        labels=[item.label for item in inputs],
    )
    layers = comparison.layers
    common_masks = [_native_common_mask(layers, i) for i in range(len(layers))]
    for item, mask in zip(inputs, common_masks):
        if not mask.any():
            raise ValueError(
                f"{item.label}: no native cell centroids are valid on the common "
                "classification footprint."
            )
    metric_rows = _all_metric_rows(
        layers,
        common_masks,
        comparison.selection.centers,
        width_m,
        inputs,
        mode,
    )
    spec_by_label = {item.label: item for item in inputs}
    for row in metric_rows:
        row["group"] = _group_name(
            spec_by_label[str(row["input_label"])], mode, group_by
        )
    class_metrics_path = out / f"{prefix}_class_metrics.csv"
    _write_csv(
        class_metrics_path,
        _class_metric_rows(layers, common_masks),
        CLASS_METRIC_FIELDS,
    )
    metrics_path = out / f"{prefix}_squares_metrics.csv"
    _write_csv(metrics_path, metric_rows)

    output_paths: List[Path] = [
        comparison.squares_gpkg,
        comparison.squares_csv,
        comparison.agreement_tif,
        comparison.legend_csv,
        class_metrics_path,
        metrics_path,
    ]
    agreement_outputs = _agreement_summaries(
        layers,
        min(range(len(layers)), key=lambda i: layers[i].cell_area),
        comparison.agreement_tif,
        comparison.selection.centers,
        width_m,
        out,
        prefix,
        patch_min_area_m2,
    )
    output_paths.extend(agreement_outputs)
    xmin, ymin, xmax, ymax = lc.common_bounds(layers)
    metadata = {
        "mode": mode,
        "group_by": group_by if mode == "temporal" else "input",
        "inputs": [
            {
                "order": index + 1,
                "label": item.label,
                "classification": item.classification,
                **({"date": item.date} if item.date is not None else {}),
                "normalized_crs": str(layers[index].crs),
                "native_resolution_m": [
                    layers[index].res_x,
                    layers[index].res_y,
                ],
                "reprojected": layers[index].reprojected,
                "class_codes": sorted(
                    int(value)
                    for value in np.unique(
                        layers[index].data[layers[index].valid]
                    )
                ),
            }
            for index, item in enumerate(inputs)
        ],
        "class_meaning_assumption": (
            "Matching integer class codes are assumed to have comparable meanings; "
            "meaning equivalence is not independently verified."
        ),
        "common_classification_footprint": {
            "bounds": [xmin, ymin, xmax, ymax],
            "definition": (
                "extent intersection; for each native grid, retain valid cell "
                "centroids "
                "inside it whose containing cell is valid in every classification"
            ),
            "valid_native_area_m2": {
                layer.label: float(common_masks[i].sum() * layer.cell_area)
                for i, layer in enumerate(layers)
            },
        },
        "square_selection": {
            "width_m": width_m,
            "requested": comparison.selection.requested,
            "found": comparison.selection.found,
            "candidate_centers_examined": comparison.selection.draws,
            "search_limit": comparison.selection.search_limit,
            "seed": seed,
            "non_overlapping": True,
            "all_intersecting_class_cells_valid": True,
        },
        "thresholds": {
            "patch_min_area_m2": patch_min_area_m2,
        },
        "analysis_settings": {
            "class_filter": class_filter,
            "reproject_to_finest": reproject_to_finest,
            "agreement_workers": n_jobs,
            "agreement_block_rows": block_rows,
            "metric_engine": "pylandstats.Landscape",
        },
        "metric_definitions": {
            "class_metrics": (
                "whole-raster per-class area_m2, proportion_pct, patch_count, "
                "and patch_density (patches/ha) on each native grid within the "
                "common footprint; 4-neighbor patches"
            ),
            "proportion_of_landscape": "class cell area / valid analysis area, percent",
            "patch_density": "pylandstats patches per hectare, 4-neighbor patches",
            "landscape_edge_density": (
                "pylandstats unlike shared 4-neighbor edge length per valid hectare; "
                "outer landscape and NoData boundaries excluded"
            ),
            "square_complexity": (
                "unlike shared horizontal/vertical edge length among centroid-selected "
                "native cells divided by geometric square area; not pylandstats "
                "edge density"
            ),
            "native_grid_rule": (
                "whole-raster and square landscape metrics use each classification's "
                "native "
                "grid; square pixel inclusion follows its cell-centroid rule"
            ),
            "agreement": (
                "inter-input agreement, not classification accuracy without "
                "independent ground truth; fine-grid cells inheriting coarse "
                "classifications are not independent observations"
            ),
        },
        "exclusions": {
            layer.label: {
                "invalid_or_nodata_cells_on_analysis_grid": int(
                    layer.valid.size - layer.valid.sum()
                ),
                "valid_cells_excluded_from_common_footprint": int(
                    layer.valid.sum() - common_masks[index].sum()
                ),
            }
            for index, layer in enumerate(layers)
        },
        "interpretation_caveats": {
            "spatial": (
                "No harmonization between inputs is performed; differences in "
                "sensor or classification method can affect the classifications."
            ),
            "temporal": (
                "Acquisition dates expose possible seasonal differences, so "
                "differences are not attributed solely to land-cover change."
            ),
            "native_resolution": (
                "Native-resolution patch and edge differences can reflect spatial "
                "resolution "
                "as well as classification differences."
            ),
        },
    }
    metadata_path = out / f"{prefix}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    output_paths.append(metadata_path)
    unique_paths = list(dict.fromkeys(output_paths))
    return AnalysisResult(comparison.selection, unique_paths)
