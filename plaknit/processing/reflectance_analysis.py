"""Class-wise comparison of continuous rasters (reflectance bands, NDVI, ...)."""

from __future__ import annotations

import itertools
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import stats

from . import landcover_compare as lc
from .landcover_analysis import _cell_centers, _native_common_mask, _slug, _write_csv

SAMPLE_FRACTION = 0.95
MIN_TEST_PIXELS = 5
UNIT_TOLERANCE = 1e-6


@dataclass
class ReflectanceInput:
    label: str
    classification: str
    rasters: Mapping[str, str]


@dataclass
class ReflectanceResult:
    outputs: List[Path]
    scaling: Dict[str, Dict[str, Dict[str, object]]]


def scale_to_unit(
    values: np.ndarray,
    valid: np.ndarray,
    scale: float,
    offset: float,
    description: str,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Scale valid values to 0-1 using raster metadata, else min-max.

    Metadata is used only when a non-default scale/offset is declared and maps all
    valid values into 0-1; otherwise the valid pixel range is stretched to 0-1.
    """
    if not valid.any():
        raise ValueError(f"{description}: raster has no valid pixels.")
    scaled = np.full(values.shape, np.nan, dtype=np.float64)
    if (scale, offset) == (1.0, 0.0):
        reason = "no scale/offset metadata"
    elif not (math.isfinite(scale) and math.isfinite(offset)) or scale <= 0:
        reason = "unusable scale/offset metadata"
    else:
        candidate = values[valid] * scale + offset
        low, high = float(candidate.min()), float(candidate.max())
        if low >= -UNIT_TOLERANCE and high <= 1 + UNIT_TOLERANCE:
            scaled[valid] = candidate
            return scaled, {"method": "metadata", "scale": scale, "offset": offset}
        reason = f"scale/offset maps values to [{low:.6g}, {high:.6g}], not 0-1"
    low, high = float(values[valid].min()), float(values[valid].max())
    if high <= low:
        raise ValueError(f"{description}: cannot min-max scale a constant raster.")
    scaled[valid] = (values[valid] - low) / (high - low)
    warnings.warn(
        f"{description}: {reason}; scaled by min/max ({low:.6g}, {high:.6g}).",
        stacklevel=2,
    )
    return scaled, {"method": "min-max", "min": low, "max": high, "reason": reason}


def _read_raster(
    path: str, classification: str, layer: lc.Layer, description: str
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    with rasterio.open(classification) as src:
        source_grid = (src.width, src.height, src.transform, src.crs)
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(f"{path}: continuous rasters must be single-band.")
        if (src.width, src.height, src.transform, src.crs) != source_grid:
            raise ValueError(
                f"{path}: {description} must exactly match its classification grid "
                "(CRS, transform, width, and height)."
            )
        values = src.read(1).astype(np.float64)
        valid = (src.read_masks(1) > 0) & np.isfinite(values)
        scale, offset = float(src.scales[0]), float(src.offsets[0])
        transform, crs = src.transform, src.crs
    scaled, info = scale_to_unit(values, valid, scale, offset, description)
    if scaled.shape != layer.data.shape or transform != layer.transform or crs != layer.crs:
        aligned = np.full(layer.data.shape, np.nan, dtype=np.float64)
        reproject(
            source=scaled,
            destination=aligned,
            src_transform=transform,
            src_crs=crs,
            src_nodata=np.nan,
            dst_transform=layer.transform,
            dst_crs=layer.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        aligned_valid = np.zeros(layer.data.shape, dtype=np.uint8)
        reproject(
            source=valid.astype(np.uint8),
            destination=aligned_valid,
            src_transform=transform,
            src_crs=crs,
            dst_transform=layer.transform,
            dst_crs=layer.crs,
            resampling=Resampling.nearest,
        )
        scaled, valid = aligned, aligned_valid.astype(bool) & np.isfinite(aligned)
    return scaled, valid, info


def _eligible_masks(
    inputs: Sequence[ReflectanceInput],
    layers: Sequence[lc.Layer],
    common_masks: Sequence[np.ndarray],
    valid: Mapping[Tuple[int, str], np.ndarray],
) -> Dict[Tuple[int, str], np.ndarray]:
    """Per raster name, keep cells that are valid in every input supplying it."""
    result: Dict[Tuple[int, str], np.ndarray] = {}
    for name in sorted({name for item in inputs for name in item.rasters}):
        participants = [i for i, item in enumerate(inputs) if name in item.rasters]
        for index in participants:
            xs, ys = _cell_centers(layers[index])
            mask = common_masks[index] & valid[(index, name)]
            for other_index in participants:
                if other_index == index:
                    continue
                other = layers[other_index]
                other_ok = common_masks[other_index] & valid[(other_index, name)]
                cols = np.floor((xs - other.left) / other.res_x).astype(np.int64)
                rows = np.floor((other.top - ys) / other.res_y).astype(np.int64)
                inside_x = (cols >= 0) & (cols < other.width)
                inside_y = (rows >= 0) & (rows < other.height)
                cols = np.clip(cols, 0, other.width - 1)
                rows = np.clip(rows, 0, other.height - 1)
                mask &= (
                    inside_y[:, None]
                    & inside_x[None, :]
                    & other_ok[rows[:, None], cols[None, :]]
                )
            result[(index, name)] = mask
    return result


def _benjamini_hochberg(rows: List[Dict[str, object]]) -> None:
    tested = sorted(
        (float(row["p_value"]), i)
        for i, row in enumerate(rows)
        if row["p_value"] is not None
    )
    m = len(tested)
    running = 1.0
    for rank in range(m, 0, -1):
        p_value, row_index = tested[rank - 1]
        running = min(running, p_value * m / rank)
        rows[row_index]["p_value_bh"] = running
    for row in rows:
        row.setdefault("p_value_bh", None)


def _validate(inputs: Sequence[ReflectanceInput]) -> None:
    if len(inputs) < 2:
        raise ValueError("At least two classification inputs are required.")
    labels = [item.label for item in inputs]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("Input labels must be non-empty and unique.")
    names = {name for item in inputs for name in item.rasters}
    if not names:
        raise ValueError("At least one --raster is required.")
    for name in names:
        if sum(name in item.rasters for item in inputs) < 2:
            raise ValueError(
                f"Raster {name!r} must be supplied for at least two inputs."
            )


def _boxplot(
    path: Path,
    groups: Sequence[Tuple[str, np.ndarray]],
    name: str,
    class_code: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    boxes = []
    for label, values in groups:
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        iqr = q3 - q1
        inside = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
        boxes.append(
            {
                "label": label,
                "q1": float(q1),
                "med": float(median),
                "q3": float(q3),
                "whislo": float(inside.min()),
                "whishi": float(inside.max()),
                "fliers": [],
            }
        )
    fig, ax = plt.subplots(figsize=(max(6, len(boxes) * 1.2), 4))
    ax.bxp(boxes, showfliers=False)
    ax.set_xticklabels([b["label"] for b in boxes], rotation=30, ha="right")
    ax.set_ylabel(f"{name} (scaled 0-1)")
    ax.set_title(f"{name}, class {class_code} (95% eligible-pixel sample)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_reflectance(
    inputs: Sequence[ReflectanceInput],
    out_dir,
    prefix: str = "reflectance",
    seed: Optional[int] = 42,
    classes: Optional[Iterable[int]] = None,
    reproject_to_finest: bool = True,
) -> ReflectanceResult:
    _validate(inputs)
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be a non-empty filename prefix, not a path.")
    class_filter = None if classes is None else {int(c) for c in classes}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    layers, _ = lc.load_layers(
        [item.classification for item in inputs],
        reproject_to_finest,
        [item.label for item in inputs],
    )
    lc.validate_class_codes(layers, classes)
    common_masks = [_native_common_mask(layers, i) for i in range(len(layers))]
    for item, mask in zip(inputs, common_masks):
        if not mask.any():
            raise ValueError(
                f"{item.label}: no valid cell centroids in the common "
                "classification footprint."
            )

    values: Dict[Tuple[int, str], np.ndarray] = {}
    valid: Dict[Tuple[int, str], np.ndarray] = {}
    scaling: Dict[str, Dict[str, Dict[str, object]]] = {}
    for index, item in enumerate(inputs):
        scaling[item.label] = {}
        for name, path in item.rasters.items():
            values[(index, name)], valid[(index, name)], info = _read_raster(
                path, item.classification, layers[index], f"{item.label}:{name}"
            )
            scaling[item.label][name] = {"path": path, **info}
    eligible = _eligible_masks(inputs, layers, common_masks, valid)

    rng = np.random.default_rng(seed)
    summary_rows: List[Dict[str, object]] = []
    samples: Dict[Tuple[int, str, int], np.ndarray] = {}
    for index, item in enumerate(inputs):
        layer = layers[index]
        codes = sorted(int(v) for v in np.unique(layer.data[common_masks[index]]))
        for name in item.rasters:
            for code in codes:
                if class_filter is not None and code not in class_filter:
                    continue
                pool = values[(index, name)][
                    eligible[(index, name)] & (layer.data == code)
                ]
                size = int(math.ceil(pool.size * SAMPLE_FRACTION))
                picked = rng.choice(pool, size=size, replace=False)
                samples[(index, name, code)] = picked
                q1 = median = q3 = None
                if picked.size:
                    q1, median, q3 = (
                        float(v) for v in np.percentile(picked, [25, 50, 75])
                    )
                summary_rows.append(
                    {
                        "input_label": item.label,
                        "raster": name,
                        "class_code": code,
                        "eligible_pixel_count": int(pool.size),
                        "sampled_pixel_count": size,
                        "median": median,
                        "lower_quartile": q1,
                        "upper_quartile": q3,
                        "scaling_method": scaling[item.label][name]["method"],
                    }
                )
    summary_path = out / f"{prefix}_class_summary.csv"
    _write_csv(summary_path, summary_rows)

    test_rows: List[Dict[str, object]] = []
    for left, right in itertools.combinations(range(len(inputs)), 2):
        for name in sorted(set(inputs[left].rasters) & set(inputs[right].rasters)):
            shared = {
                code
                for (i, n, code) in samples
                if n == name and i in (left, right)
            }
            for code in sorted(shared):
                a = samples.get((left, name, code), np.asarray([]))
                b = samples.get((right, name, code), np.asarray([]))
                p_value = effect = None
                if min(a.size, b.size) < MIN_TEST_PIXELS:
                    status = "insufficient_pixels"
                else:
                    try:
                        result = stats.mannwhitneyu(a, b, alternative="two-sided")
                        p_value = float(result.pvalue)
                        effect = float(2 * result.statistic / (a.size * b.size) - 1)
                        status = "tested"
                    except ValueError as exc:
                        status = f"not_tested: {exc}"
                test_rows.append(
                    {
                        "input_a": inputs[left].label,
                        "input_b": inputs[right].label,
                        "raster": name,
                        "class_code": code,
                        "n_a": int(a.size),
                        "n_b": int(b.size),
                        "median_difference_b_minus_a": (
                            float(np.median(b) - np.median(a))
                            if a.size and b.size
                            else None
                        ),
                        "test": "Mann-Whitney U (two-sided)",
                        "rank_biserial_a_vs_b": effect,
                        "p_value": p_value,
                        "p_value_bh": None,
                        "status": status,
                    }
                )
    _benjamini_hochberg(test_rows)
    tests_path = out / f"{prefix}_mann_whitney.csv"
    _write_csv(tests_path, test_rows)

    plots: List[Path] = []
    for name in sorted({name for item in inputs for name in item.rasters}):
        for code in sorted({c for (_, n, c) in samples if n == name}):
            groups = [
                (inputs[i].label, samples[(i, name, code)])
                for i in range(len(inputs))
                if samples.get((i, name, code), np.asarray([])).size
            ]
            if groups:
                path = out / f"{prefix}_{_slug(name)}_class_{code}.png"
                _boxplot(path, groups, name, code)
                plots.append(path)

    metadata = {
        "inputs": [
            {
                "label": item.label,
                "classification": item.classification,
                "rasters": scaling[item.label],
            }
            for item in inputs
        ],
        "seed": seed,
        "sample_fraction": SAMPLE_FRACTION,
        "minimum_pixels_per_group_for_test": MIN_TEST_PIXELS,
        "scaling": (
            "Raster scale/offset metadata when it maps valid values to 0-1; "
            "otherwise min-max over the raster's valid pixels. Min-max scaling is "
            "relative to each raster's own range."
        ),
        "eligibility": (
            "For each raster name, cells in the common classification footprint "
            "with valid values in every input supplying that raster."
        ),
        "interpretation_caveats": (
            "Class masks come from each input's own classification, so a class "
            "need not cover the same locations. Neighboring pixels are spatially "
            "autocorrelated, so Mann-Whitney p-values are overstated; prefer the "
            "effect size. Differences are not attributable solely to sensor or "
            "land-cover change."
        ),
        "effect_size": (
            "rank-biserial correlation (2U/(n_a n_b) - 1); positive means "
            "input_a values tend to exceed input_b values"
        ),
    }
    metadata_path = out / f"{prefix}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    outputs = [summary_path, tests_path, *plots, metadata_path]
    return ReflectanceResult(outputs, scaling)
