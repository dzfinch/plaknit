"""Mode-specific spatial and temporal land-cover analyses."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import rasterio

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
            raise ValueError("Temporal mode requires at least two distinct dates.")
        if group_by not in ("date", "year"):
            raise ValueError("Temporal group_by must be 'date' or 'year'.")
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
        data = np.asarray(layer.data[r0 : r1 + 1, c0 : c1 + 1], dtype=np.int64).copy()
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
            metrics_kwargs={"patch_density": {"percent": False, "hectares": True}},
        )
        for class_code in all_classes:
            if class_code in present:
                metric = metrics.loc[class_code]
                proportion = float(metric["proportion_of_landscape"])
                count = int(metric["number_of_patches"])
                density = float(metric["patch_density"])
                area = float(
                    np.count_nonzero(data[selected] == class_code) * layer.cell_area
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


def _transition_metrics(
    comparison: lc.CompareResult,
    width_m: float,
) -> Tuple[List[Dict[str, object]], List[str], Dict[str, object]]:
    layers = comparison.layers
    fine = layers[comparison.finest_index]
    reference_index = max(range(len(layers)), key=lambda i: layers[i].cell_area)
    class_codes = sorted(
        {int(value) for layer in layers for value in np.unique(layer.data[layer.valid])}
    )
    other_indices = [i for i in range(len(layers)) if i != reference_index]
    transitions = [(i, code) for i in other_indices for code in class_codes]

    def transition_column(index: int, code: int) -> str:
        return f"input_{index + 1}_{layers[index].label}_class_{code}_fraction"

    columns = [
        {
            "column": transition_column(i, code),
            "input_order": i + 1,
            "input_label": layers[i].label,
            "class_code": code,
        }
        for i, code in transitions
    ]
    fields = [
        "square_id",
        "reference_input_label",
        "reference_class_code",
        "reference_class_cell_count",
        "reference_class_fraction",
    ] + [transition_column(i, code) for i, code in transitions]
    maps = [lc.overlap_maps(fine, layer) for layer in layers]
    rows: List[Dict[str, object]] = []
    half = width_m / 2.0
    with rasterio.open(comparison.agreement_tif) as src:
        for square_id, (cx, cy) in enumerate(comparison.selection.centers, start=1):
            r0, r1, c0, c1 = lc.centroid_window(
                fine, cx - half, cy - half, cx + half, cy + half
            )
            r0, c0 = max(0, r0), max(0, c0)
            r1, c1 = min(fine.height - 1, r1), min(fine.width - 1, c1)
            if r0 <= r1 and c0 <= c1:
                valid = (
                    src.read(
                        1,
                        window=rasterio.windows.Window(
                            c0, r0, c1 - c0 + 1, r1 - r0 + 1
                        ),
                    )
                    > 0
                )
                values = [
                    lc.sample_with_maps(
                        layer, row_map[r0 : r1 + 1], col_map[c0 : c1 + 1]
                    )[0]
                    for layer, (row_map, col_map) in zip(layers, maps)
                ]
            else:
                valid = np.zeros((0, 0), dtype=bool)
                values = [np.empty((0, 0), dtype=int) for _ in layers]
            valid_count = int(valid.sum())
            for code in class_codes:
                selected = valid & (values[reference_index] == code)
                count = int(selected.sum())
                row: Dict[str, object] = {
                    "square_id": square_id,
                    "reference_input_label": layers[reference_index].label,
                    "reference_class_code": code,
                    "reference_class_cell_count": count,
                    "reference_class_fraction": (
                        count / valid_count if valid_count else None
                    ),
                }
                for index, target in transitions:
                    row[transition_column(index, target)] = (
                        float(
                            np.count_nonzero(selected & (values[index] == target))
                            / count
                        )
                        if count
                        else None
                    )
                rows.append(row)
    metadata: Dict[str, object] = {
        "reference_input_order": reference_index + 1,
        "reference_input_label": layers[reference_index].label,
        "reference_resolution_m": [
            layers[reference_index].res_x,
            layers[reference_index].res_y,
        ],
        "comparison_inputs": [
            {
                "input_order": i + 1,
                "input_label": layers[i].label,
                "resolution_m": [layers[i].res_x, layers[i].res_y],
            }
            for i in other_indices
        ],
        "class_codes": class_codes,
        "transition_columns": columns,
    }
    return rows, fields, metadata


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
) -> AnalysisResult:
    _validate_specs(mode, inputs, group_by)
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be a non-empty filename prefix, not a path.")
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
    if not comparison.agreement_counts:
        raise ValueError("The agreement raster has no valid common cells.")
    metric_rows, metric_fields, transition_metadata = _transition_metrics(
        comparison, width_m
    )
    class_metrics_path = out / f"{prefix}_class_metrics.csv"
    _write_csv(
        class_metrics_path,
        _class_metric_rows(layers, common_masks),
        CLASS_METRIC_FIELDS,
    )
    metrics_path = out / f"{prefix}_squares_metrics.csv"
    _write_csv(metrics_path, metric_rows, metric_fields)

    output_paths: List[Path] = [
        comparison.squares_gpkg,
        comparison.squares_csv,
        comparison.agreement_tif,
        comparison.legend_csv,
        class_metrics_path,
        metrics_path,
    ]
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
                    for value in np.unique(layers[index].data[layers[index].valid])
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
        "square_class_transitions": transition_metadata,
        "analysis_settings": {
            "class_filter": class_filter,
            "reproject_to_finest": reproject_to_finest,
            "agreement_workers": n_jobs,
            "agreement_block_rows": block_rows,
            "metric_engine": "pylandstats.Landscape",
            "metric_engine_version": version("pylandstats"),
        },
        "metric_definitions": {
            "class_metrics": (
                "whole-raster per-class area_m2, proportion_pct, patch_count, "
                "and patch_density (patches/ha) on each native grid within the "
                "common footprint; 4-neighbor patches"
            ),
            "square_class_transitions": (
                "For each reference class, count valid agreement-grid cells whose "
                "centroids are inside the square and whose mapped reference class "
                "matches; reference_class_fraction divides that count by all valid "
                "square agreement-grid cells. Each transition fraction divides the "
                "count additionally mapped to the comparison class by the reference "
                "class count; a zero denominator yields a blank. Mapping uses the "
                "agreement raster's largest-overlap selection."
            ),
            "square_shannon_diversity_index": (
                "PyLandStats Landscape.shannon_diversity_index on centroid-selected "
                "native cells, natural logarithm; single-class squares are reported "
                "as zero instead of PyLandStats' undefined result"
            ),
            "square_joint_entropy": (
                "PyLandStats Landscape.joint_entropy on centroid-selected native "
                "cells, base 2, using horizontal and vertical adjacencies; "
                "single-class squares are reported as zero instead of PyLandStats' "
                "undefined result"
            ),
            "native_grid_rule": (
                "Whole-raster class metrics and existing square metrics use each "
                "classification's native grid; square pixel inclusion follows its "
                "cell-centroid rule. Class transitions use the agreement grid."
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
