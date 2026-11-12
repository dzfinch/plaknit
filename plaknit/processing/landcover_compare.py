"""Compare two or more categorical land-cover rasters at different resolutions.

Workflow (see ``compare_landcover``):

1. Validate inputs (integer classes, projected CRS in metres, north-up grids) and
   reproject any raster whose CRS differs from the finest raster's CRS.
2. Randomly select non-overlapping squares centred on finest-raster cell
   centroids, fully inside the common coverage and valid in every raster.
3. Compute per-raster, per-square metrics from each raster's native centroids.
4. Write an agreement GeoTIFF on the finest raster's grid plus a legend.

Boundary convention used everywhere: a cell or square covers
``[west, east) x (south, north]`` for the purpose of point membership, i.e. west
and north edges are inclusive. A point exactly on a cell boundary therefore
belongs to the cell with the higher column / higher row index (east / south).
"""

from __future__ import annotations

import csv
import math
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject

EPS = 1e-9
DEFAULT_SEARCH_FACTOR = 100
LEGEND_FULL_LIMIT = 10000


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #
@dataclass
class Layer:
    """A single-band categorical raster held in memory."""

    path: str
    label: str
    data: np.ndarray
    valid: np.ndarray
    transform: Affine
    crs: CRS
    reprojected: bool = False

    @property
    def res_x(self) -> float:
        return float(self.transform.a)

    @property
    def res_y(self) -> float:
        return float(-self.transform.e)

    @property
    def left(self) -> float:
        return float(self.transform.c)

    @property
    def top(self) -> float:
        return float(self.transform.f)

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax)."""
        return (
            self.left,
            self.top - self.height * self.res_y,
            self.left + self.width * self.res_x,
            self.top,
        )

    @property
    def cell_area(self) -> float:
        return self.res_x * self.res_y


def _check_projected_metres(crs: Optional[CRS], path: str) -> None:
    if crs is None:
        raise ValueError(f"{path}: raster has no CRS.")
    if not crs.is_projected:
        raise ValueError(
            f"{path}: CRS must be projected (metres) to measure distances/areas."
        )
    factor = crs.linear_units_factor[1]
    if factor is None or abs(factor - 1.0) > 1e-9:
        raise ValueError(f"{path}: CRS linear unit must be the metre (got {factor}).")


def _check_north_up(transform: Affine, path: str) -> None:
    if abs(transform.b) > 1e-12 or abs(transform.d) > 1e-12:
        raise ValueError(f"{path}: rotated/sheared rasters are not supported.")
    if transform.a <= 0 or transform.e >= 0:
        raise ValueError(f"{path}: raster must be north-up (positive x, negative y).")


def inspect_raster(path) -> Dict[str, object]:
    """Read CRS and cell size after validating the raster's basic properties.

    ``projected`` is True for a metre-based projected CRS. Other CRSs (e.g.
    geographic degrees) are allowed here because such rasters can be reprojected
    to the target CRS; ``res_x``/``res_y`` are then in CRS units, not metres.
    """
    path = str(path)
    with rasterio.open(path) as src:
        if np.dtype(src.dtypes[0]).kind not in "iuf":
            raise ValueError(f"{path}: categorical rasters must be integer or float.")
        if src.crs is None:
            raise ValueError(f"{path}: raster has no CRS.")
        projected = bool(src.crs.is_projected)
        if projected:
            _check_projected_metres(src.crs, path)
        _check_north_up(src.transform, path)
        return {
            "path": path,
            "crs": src.crs,
            "projected": projected,
            "res_x": float(src.transform.a),
            "res_y": float(-src.transform.e),
        }


def _compact_int(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.int32)
    lo, hi = values.min(), values.max()
    if lo >= np.iinfo(np.int32).min and hi <= np.iinfo(np.int32).max:
        return values.astype(np.int32)
    return values.astype(np.int64)


def align_to_crs(layer: Layer, target_crs: CRS) -> Layer:
    """Reproject ``layer`` to ``target_crs`` with nearest-neighbour resampling.

    Class codes are never interpolated. Validity is reprojected as a separate
    mask so cells with no source coverage become NoData. Metre-based sources keep
    their nominal cell size; others (e.g. geographic) use GDAL's default size.
    """
    if layer.crs == target_crs:
        return layer
    height, width = layer.data.shape
    kwargs = {}
    if layer.crs.is_projected and layer.crs.linear_units_factor[1] == 1.0:
        # Keep the nominal metre cell size so per-square counts match the source grid.
        kwargs["resolution"] = (layer.res_x, layer.res_y)
    dst_transform, dst_w, dst_h = calculate_default_transform(
        layer.crs, target_crs, width, height, *layer.bounds, **kwargs
    )
    src_data = layer.data
    if src_data.dtype.itemsize > 4:
        raise ValueError(f"{layer.path}: class codes exceed 32 bits; cannot reproject.")
    dst_data = np.zeros((dst_h, dst_w), dtype=src_data.dtype)
    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=layer.transform,
        src_crs=layer.crs,
        dst_transform=dst_transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest,
    )
    dst_valid = np.zeros((dst_h, dst_w), dtype=np.uint8)
    reproject(
        source=layer.valid.astype(np.uint8),
        destination=dst_valid,
        src_transform=layer.transform,
        src_crs=layer.crs,
        dst_transform=dst_transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest,
    )
    return Layer(
        path=layer.path,
        label=layer.label,
        data=dst_data,
        valid=dst_valid.astype(bool),
        transform=dst_transform,
        crs=CRS.from_user_input(target_crs),
        reprojected=True,
    )


def _read_layer(path: str, label: str) -> Layer:
    with rasterio.open(path) as src:
        raw = src.read(1)
        valid = src.read_masks(1) > 0
        if raw.dtype.kind == "f":
            # Float rasters are accepted only when they hold whole-number classes.
            valid &= np.isfinite(raw)
            vals = raw[valid]
            if vals.size and not np.array_equal(vals, np.round(vals)):
                raise ValueError(
                    f"{path}: float raster has non-integer values, so it does not "
                    "look categorical (continuous data such as NDVI cannot be compared)."
                )
            raw = np.where(valid, raw, 0)
        return Layer(path, label, _compact_int(raw), valid, src.transform, src.crs)


def _unique_labels(paths: Sequence[str]) -> List[str]:
    labels = []
    for i, p in enumerate(paths, start=1):
        stem = "".join(c if c.isalnum() else "_" for c in Path(p).stem)
        labels.append(f"r{i}_{stem}")
    return labels


def load_layers(
    paths: Sequence,
    reproject_to_finest: bool = True,
    labels: Optional[Sequence[str]] = None,
) -> Tuple[List[Layer], int]:
    """Load rasters in input order; return ``(layers, finest_index)``.

    The finest raster has the smallest native cell area (first input on ties); its
    CRS is the target CRS. Rasters in another CRS are reprojected when
    ``reproject_to_finest`` is true, otherwise a ``ValueError`` is raised.
    """
    paths = [str(p) for p in paths]
    if len(paths) < 2:
        raise ValueError("At least two rasters are required.")
    pattern_dtype(len(paths))
    infos = [inspect_raster(p) for p in paths]
    projected = [i for i, info in enumerate(infos) if info["projected"]]
    if not projected:
        raise ValueError(
            "At least one input raster must have a projected CRS in metres to "
            "define the target CRS."
        )
    # Cell sizes of geographic rasters are in degrees, so only projected rasters
    # compete to be the finest; the others are checked after reprojection.
    finest = min(projected, key=lambda i: infos[i]["res_x"] * infos[i]["res_y"])
    target = infos[finest]["crs"]
    if labels is None:
        labels = _unique_labels(paths)
    elif len(labels) != len(paths) or len(set(labels)) != len(labels):
        raise ValueError("Raster labels must be unique and match the input count.")
    layers = []
    for layer_index, (p, label) in enumerate(zip(paths, labels)):
        layer = _read_layer(p, label)
        if layer.crs != target:
            if not reproject_to_finest:
                raise ValueError(
                    f"{p}: CRS differs from the finest raster's CRS and "
                    "reprojection is disabled."
                )
            native_res = (infos[layer_index]["res_x"], infos[layer_index]["res_y"])
            layer = align_to_crs(layer, target)
            if (
                abs(layer.res_x - native_res[0]) > 0.01 * native_res[0]
                or abs(layer.res_y - native_res[1]) > 0.01 * native_res[1]
            ):
                warnings.warn(
                    f"{p}: reprojection changed the cell size from "
                    f"{native_res[0]:.4g} x {native_res[1]:.4g} (source CRS units) to "
                    f"{layer.res_x:.4g} x {layer.res_y:.4g} m; per-square cell counts "
                    "reflect the reprojected grid, not the source grid.",
                    stacklevel=2,
                )
            if layer.cell_area < infos[finest]["res_x"] * infos[finest]["res_y"] - EPS:
                raise ValueError(
                    f"{p}: after reprojection its cells ({layer.res_x:.3g} x "
                    f"{layer.res_y:.3g} m) are finer than the finest projected "
                    "raster; reproject the finest raster to a metre CRS first."
                )
        layers.append(layer)
    return layers, finest


def validate_class_codes(
    layers: Sequence[Layer], classes: Optional[Iterable[int]] = None
) -> List[int]:
    """Check class codes are comparable; return the sorted union of codes.

    Raises if the rasters share no class, or (when ``classes`` is given) if any
    raster contains a code outside it. A warning is issued when the rasters'
    class sets differ.
    """
    sets = [set(np.unique(layer.data[layer.valid]).tolist()) for layer in layers]
    union = set().union(*sets)
    if not union:
        raise ValueError("No valid cells found in the input rasters.")
    if not set.intersection(*sets):
        raise ValueError("The rasters share no class codes; they are not comparable.")
    if classes is not None:
        allowed = set(int(c) for c in classes)
        for layer, s in zip(layers, sets):
            extra = sorted(s - allowed)
            if extra:
                raise ValueError(
                    f"{layer.path}: class codes {extra} are not in the allowed set."
                )
    elif any(s != union for s in sets):
        warnings.warn(
            "Rasters contain different class-code sets: "
            + "; ".join(f"{l.label}={sorted(s)}" for l, s in zip(layers, sets)),
            stacklevel=2,
        )
    return sorted(union)


def common_bounds(layers: Sequence[Layer]) -> Tuple[float, float, float, float]:
    """Intersection of the raster extents (xmin, ymin, xmax, ymax)."""
    xmin = max(l.bounds[0] for l in layers)
    ymin = max(l.bounds[1] for l in layers)
    xmax = min(l.bounds[2] for l in layers)
    ymax = min(l.bounds[3] for l in layers)
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("The rasters have no common coverage.")
    return xmin, ymin, xmax, ymax


# --------------------------------------------------------------------------- #
# Square windows and metrics
# --------------------------------------------------------------------------- #
def _centroid_range(lo: float, hi: float, origin: float, step: float) -> Tuple[int, int]:
    """Indices ``c`` with centroid ``origin + (c + .5) * step`` in ``[lo, hi)``."""
    a = (lo - origin) / step - 0.5
    b = (hi - origin) / step - 0.5
    return math.ceil(a - EPS), math.ceil(b - EPS) - 1


def centroid_window(
    layer: Layer, xmin: float, ymin: float, xmax: float, ymax: float
) -> Tuple[int, int, int, int]:
    """Inclusive ``(row0, row1, col0, col1)`` of cells whose centroids lie in the square.

    A centroid on the west or north edge is included; one on the east or south
    edge is excluded.
    """
    c0, c1 = _centroid_range(xmin, xmax, layer.left, layer.res_x)
    r0, r1 = _centroid_range(-ymax, -ymin, -layer.top, layer.res_y)
    return r0, r1, c0, c1


def intersect_window(
    layer: Layer, xmin: float, ymin: float, xmax: float, ymax: float
) -> Tuple[int, int, int, int]:
    """Inclusive ``(row0, row1, col0, col1)`` of cells whose interior meets the square."""
    c0 = math.floor((xmin - layer.left) / layer.res_x + EPS)
    c1 = math.ceil((xmax - layer.left) / layer.res_x - EPS) - 1
    r0 = math.floor((layer.top - ymax) / layer.res_y + EPS)
    r1 = math.ceil((layer.top - ymin) / layer.res_y - EPS) - 1
    return r0, r1, c0, c1


def square_is_valid(
    layers: Sequence[Layer], xmin: float, ymin: float, xmax: float, ymax: float
) -> bool:
    """True if every cell intersecting the square is inside and valid in all layers."""
    for layer in layers:
        r0, r1, c0, c1 = intersect_window(layer, xmin, ymin, xmax, ymax)
        if r0 < 0 or c0 < 0 or r1 >= layer.height or c1 >= layer.width:
            return False
        if not layer.valid[r0 : r1 + 1, c0 : c1 + 1].all():
            return False
    return True


def squares_overlap(a: Tuple[float, float], b: Tuple[float, float], width: float) -> bool:
    """True if axis-aligned squares of ``width`` centred at ``a`` and ``b`` overlap.

    Squares that only touch along an edge or corner do not overlap.
    """
    return abs(a[0] - b[0]) < width - EPS and abs(a[1] - b[1]) < width - EPS


def shannon_entropy(counts: Sequence[int]) -> float:
    """Shannon entropy (bits, log base 2) of class proportions."""
    counts = np.asarray([c for c in counts if c > 0], dtype=float)
    if counts.size <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def majority_class(values: np.ndarray) -> int:
    """Most frequent class; ties go to the smallest class code."""
    classes, counts = np.unique(values, return_counts=True)
    return int(classes[int(np.argmax(counts))])


def edge_length_between_classes(block: np.ndarray, res_x: float, res_y: float) -> float:
    """Total length of shared edges between 4-adjacent cells of different classes."""
    horiz = int((block[:, 1:] != block[:, :-1]).sum())  # vertical edges, length res_y
    vert = int((block[1:, :] != block[:-1, :]).sum())  # horizontal edges, length res_x
    return horiz * res_y + vert * res_x


def square_metrics(
    layer: Layer, xmin: float, ymin: float, xmax: float, ymax: float
) -> Dict[str, float]:
    """Majority class and PyLandStats diversity metrics on native cell centroids."""
    import pylandstats

    r0, r1, c0, c1 = centroid_window(layer, xmin, ymin, xmax, ymax)
    block = layer.data[r0 : r1 + 1, c0 : c1 + 1]
    if block.size == 0:
        raise ValueError("Square contains no cell centroids for this raster.")
    classes, counts = np.unique(block, return_counts=True)
    if len(classes) == 1:
        shannon, joint = 0.0, 0.0
    else:
        nodata = np.iinfo(np.int64).min
        while nodata in classes:
            nodata += 1
        landscape = pylandstats.Landscape(
            np.asarray(block, dtype=np.int64),
            res=(layer.res_x, layer.res_y),
            nodata=nodata,
            neighborhood_rule="4",
        )
        shannon = float(landscape.shannon_diversity_index())
        joint = float(landscape.joint_entropy())
        if not math.isfinite(shannon) or not math.isfinite(joint):
            raise ValueError(
                f"{layer.label}: PyLandStats returned a non-finite square diversity metric."
            )
    return {
        "majority": int(classes[int(np.argmax(counts))]),
        "shannon_diversity_index": shannon,
        "joint_entropy": joint,
        "n_cells": int(block.size),
    }


# --------------------------------------------------------------------------- #
# Square selection
# --------------------------------------------------------------------------- #
@dataclass
class SelectionResult:
    centers: List[Tuple[float, float]]
    requested: int
    draws: int
    search_limit: int
    pool_size: int
    message: str = ""

    @property
    def found(self) -> int:
        return len(self.centers)

    @property
    def limit_reached(self) -> bool:
        return self.found < self.requested and self.draws >= self.search_limit

    @property
    def pool_exhausted(self) -> bool:
        return self.found < self.requested and self.draws >= self.pool_size


def select_squares(
    layers: Sequence[Layer],
    finest_index: int,
    width_m: float,
    n_samples: int,
    seed: Optional[int] = 42,
    search_limit: Optional[int] = None,
) -> SelectionResult:
    """Randomly pick non-overlapping valid squares centred on finest-grid centroids.

    Candidate centres are the finest raster's cell centroids whose square fits in
    the common coverage. Candidates are drawn without replacement in random order
    (``numpy.random.default_rng(seed)``); at most ``search_limit`` candidates
    (default ``100 * n_samples``) are examined.
    """
    if not math.isfinite(width_m) or width_m <= 0 or n_samples < 1:
        raise ValueError("width_m must be positive and n_samples at least 1.")
    if search_limit is not None and int(search_limit) < 1:
        raise ValueError("search_limit must be at least 1 when provided.")
    coarsest = max(max(l.res_x, l.res_y) for l in layers)
    if width_m < coarsest:
        raise ValueError(
            f"Square width {width_m} m is smaller than the coarsest cell ({coarsest} m)."
        )
    limit = (
        int(search_limit)
        if search_limit is not None
        else DEFAULT_SEARCH_FACTOR * n_samples
    )
    fine = layers[finest_index]
    cxmin, cymin, cxmax, cymax = common_bounds(layers)
    half = width_m / 2.0
    if cxmax - cxmin < width_m - EPS or cymax - cymin < width_m - EPS:
        return SelectionResult([], n_samples, 0, limit, 0, _message([], n_samples, 0, limit, 0))

    # Fine cells whose centroid lies within [cxmin + half, cxmax - half] (inclusive).
    c0 = math.ceil((cxmin + half - fine.left) / fine.res_x - 0.5 - EPS)
    c1 = math.floor((cxmax - half - fine.left) / fine.res_x - 0.5 + EPS)
    r0 = math.ceil((fine.top - (cymax - half)) / fine.res_y - 0.5 - EPS)
    r1 = math.floor((fine.top - (cymin + half)) / fine.res_y - 0.5 + EPS)
    c0, r0 = max(c0, 0), max(r0, 0)
    c1, r1 = min(c1, fine.width - 1), min(r1, fine.height - 1)
    if c1 < c0 or r1 < r0:
        return SelectionResult([], n_samples, 0, limit, 0, _message([], n_samples, 0, limit, 0))

    ncols = c1 - c0 + 1
    pool = ncols * (r1 - r0 + 1)
    rng = np.random.default_rng(seed)
    n_draw = min(limit, pool)
    picks = rng.choice(pool, size=n_draw, replace=False)

    centers: List[Tuple[float, float]] = []
    draws = 0
    for flat in picks:
        if len(centers) >= n_samples:
            break
        draws += 1
        row, col = r0 + int(flat) // ncols, c0 + int(flat) % ncols
        cx = fine.left + (col + 0.5) * fine.res_x
        cy = fine.top - (row + 0.5) * fine.res_y
        if any(squares_overlap((cx, cy), c, width_m) for c in centers):
            continue
        if not square_is_valid(layers, cx - half, cy - half, cx + half, cy + half):
            continue
        centers.append((cx, cy))
    result = SelectionResult(centers, n_samples, draws, limit, pool)
    result.message = _message(centers, n_samples, draws, limit, pool)
    return result


def _message(centers, requested, draws, limit, pool) -> str:
    found = len(centers)
    if found >= requested:
        return f"Selected {found} squares after examining {draws} candidate centres."
    reason = (
        "the search limit was reached"
        if draws >= limit and limit < pool
        else "all eligible candidate centres were examined"
    )
    return (
        f"Found {found} of {requested} requested squares after examining {draws} "
        f"candidate centres (search limit: {limit}; {reason}). This does not show "
        "that more squares cannot exist."
    )


# --------------------------------------------------------------------------- #
# Overlap-based cell selection (agreement surface)
# --------------------------------------------------------------------------- #
def select_cells_1d(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Pick the coarse cell with the largest 1-D overlap for each output interval.

    ``a``/``b`` are the interval ends in coarse-cell units (coarse cell ``k`` spans
    ``[k, k + 1]``). Cells outside ``[0, n)`` count as candidates with no data; if
    one wins, ``-1`` is returned. Ties go to the cell containing the interval
    midpoint; a midpoint exactly on a boundary resolves to the higher index.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    k0 = np.floor(a + EPS).astype(np.int64)
    k1 = np.ceil(b - EPS).astype(np.int64)
    kc = np.floor((a + b) / 2.0 + EPS).astype(np.int64)
    span = int((k1 - k0).max()) if a.size else 0

    overlaps = []
    for j in range(span):
        k = k0 + j
        ov = np.minimum(b, k + 1) - np.maximum(a, k)
        overlaps.append(np.where(k < k1, ov, -np.inf))
    best = np.max(overlaps, axis=0)
    chosen = np.full(a.shape, -1, dtype=np.int64)
    centre_tied = np.zeros(a.shape, dtype=bool)
    for j, ov in enumerate(overlaps):
        k = k0 + j
        tied = ov >= best - EPS
        chosen = np.where(tied, k, chosen)  # ends as the highest tied index
        centre_tied |= tied & (k == kc)
    result = np.where(centre_tied, kc, chosen)
    return np.where((result < 0) | (result >= n), -1, result)


def overlap_maps(fine: Layer, other: Layer) -> Tuple[np.ndarray, np.ndarray]:
    """Row and column index of ``other`` chosen for each fine row / column (-1 = none)."""
    cols = np.arange(fine.width, dtype=float)
    x_lo = fine.left + cols * fine.res_x
    x_hi = x_lo + fine.res_x
    col_map = select_cells_1d(
        (x_lo - other.left) / other.res_x, (x_hi - other.left) / other.res_x, other.width
    )
    rows = np.arange(fine.height, dtype=float)
    y_top = fine.top - rows * fine.res_y
    y_bot = y_top - fine.res_y
    row_map = select_cells_1d(
        (other.top - y_top) / other.res_y, (other.top - y_bot) / other.res_y, other.height
    )
    return row_map, col_map


def sample_with_maps(
    layer: Layer, row_map: np.ndarray, col_map: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Gather values/validity for each fine cell; unmapped cells are invalid."""
    inside = (row_map >= 0)[:, None] & (col_map >= 0)[None, :]
    rr = np.where(row_map >= 0, row_map, 0)[:, None]
    cc = np.where(col_map >= 0, col_map, 0)[None, :]
    values = layer.data[rr, cc]
    valid = layer.valid[rr, cc] & inside
    return values, valid


# --------------------------------------------------------------------------- #
# Agreement encoding
# --------------------------------------------------------------------------- #
def canonical_groups(values: np.ndarray) -> np.ndarray:
    """Canonical grouping sequence for stacked class values (axis 0 = input order).

    The first value is group 1; each later value takes the group of an earlier
    equal value, or the next unused group number.
    """
    values = np.asarray(values)
    n = values.shape[0]
    groups = np.zeros(values.shape, dtype=np.int64)
    groups[0] = 1
    for i in range(1, n):
        matched = np.zeros(values.shape[1:], dtype=bool)
        for j in range(i):
            m = (values[i] == values[j]) & ~matched
            groups[i] = np.where(m, groups[j], groups[i])
            matched |= m
        groups[i] = np.where(matched, groups[i], groups[:i].max(axis=0) + 1)
    return groups


@lru_cache(maxsize=None)
def _completions(remaining: int, current_max: int) -> int:
    """Number of canonical sequences completing a prefix with ``current_max`` groups."""
    if remaining == 0:
        return 1
    return current_max * _completions(remaining - 1, current_max) + _completions(
        remaining - 1, current_max + 1
    )


def bell_number(n: int) -> int:
    """Number of distinct canonical grouping sequences for ``n`` rasters."""
    return _completions(n, 0)


def pattern_ids(groups: np.ndarray) -> np.ndarray:
    """Numeric pattern ID (1-based lexicographic rank of the grouping sequence).

    IDs are ranks among all ``bell_number(n)`` canonical sequences for ``n``
    rasters, so they are data-independent. They are labels, not the sequence.
    """
    n = groups.shape[0]
    rank = np.zeros(groups.shape[1:], dtype=np.uint64)
    current_max = np.zeros(groups.shape[1:], dtype=np.int64)
    for i in range(n):
        g = groups[i]
        table = np.array(
            [_completions(n - i - 1, m) for m in range(n + 1)], dtype=np.uint64
        )
        for v in range(1, n + 1):
            rank += np.where(v < g, table[np.maximum(current_max, v)], 0)
        current_max = np.maximum(current_max, g)
    return rank + 1


def sequence_from_id(pattern_id: int, n: int) -> Tuple[int, ...]:
    """Inverse of :func:`pattern_ids` for a single ID."""
    if not 1 <= pattern_id <= bell_number(n):
        raise ValueError(f"Pattern ID {pattern_id} is invalid for {n} rasters.")
    rank = pattern_id - 1
    seq: List[int] = []
    current_max = 0
    for i in range(n):
        for v in range(1, current_max + 2):
            count = _completions(n - i - 1, max(current_max, v))
            if rank < count:
                seq.append(v)
                current_max = max(current_max, v)
                break
            rank -= count
    return tuple(seq)


def format_sequence(seq: Sequence[int], compact: bool = False) -> str:
    """Render a grouping sequence.

    ``compact=True`` concatenates single-digit groups (``11223``) and uses ``-``
    separators whenever any group number is 10 or greater.
    """
    seq = [int(s) for s in seq]
    if compact and max(seq) < 10:
        return "".join(str(s) for s in seq)
    return "-".join(str(s) for s in seq)


def pattern_dtype(n_rasters: int) -> np.dtype:
    """Smallest unsigned dtype that can hold every pattern ID (0 is NoData)."""
    top = bell_number(n_rasters)
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if top <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    raise ValueError(
        f"Bell({n_rasters}) pattern IDs exceed the largest safely representable "
        "GeoTIFF unsigned integer."
    )


def compute_agreement_block(
    fine_rows: slice,
    layers: Sequence[Layer],
    maps: Sequence[Tuple[np.ndarray, np.ndarray]],
    dtype: np.dtype,
) -> np.ndarray:
    """Agreement pattern IDs (0 = NoData) for a block of fine rows."""
    values = []
    valid = None
    for layer, (row_map, col_map) in zip(layers, maps):
        v, ok = sample_with_maps(layer, row_map[fine_rows], col_map)
        values.append(v)
        valid = ok if valid is None else valid & ok
    ids = pattern_ids(canonical_groups(np.stack(values)))
    return np.where(valid, ids, 0).astype(dtype)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
@dataclass
class CompareResult:
    selection: SelectionResult
    squares_gpkg: Path
    squares_csv: Path
    agreement_tif: Path
    legend_csv: Path
    layers: List[Layer] = field(default_factory=list)
    finest_index: int = 0
    agreement_counts: Dict[int, int] = field(default_factory=dict)


def square_records(
    layers: Sequence[Layer], centers: Sequence[Tuple[float, float]], width_m: float
) -> List[Dict[str, object]]:
    half = width_m / 2.0
    records = []
    for sid, (cx, cy) in enumerate(centers, start=1):
        rec: Dict[str, object] = {
            "square_id": sid,
            "center_x": cx,
            "center_y": cy,
            "xmin": cx - half,
            "ymin": cy - half,
            "xmax": cx + half,
            "ymax": cy + half,
        }
        for layer in layers:
            m = square_metrics(layer, cx - half, cy - half, cx + half, cy + half)
            rec[f"{layer.label}_majority"] = m["majority"]
            rec[f"{layer.label}_shannon_diversity_index"] = m["shannon_diversity_index"]
            rec[f"{layer.label}_joint_entropy"] = m["joint_entropy"]
            rec[f"{layer.label}_n_cells"] = m["n_cells"]
        records.append(rec)
    return records


def _write_squares(records, crs, gpkg: Path, csv_path: Path, columns=None) -> None:
    import geopandas as gpd
    from shapely.geometry import box

    if columns is None:
        columns = (
            list(records[0].keys())
            if records
            else ["square_id", "center_x", "center_y", "xmin", "ymin", "xmax", "ymax"]
        )
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    geoms = [box(r["xmin"], r["ymin"], r["xmax"], r["ymax"]) for r in records]
    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs=crs)
    if records:
        gdf.to_file(gpkg, driver="GPKG")
    else:
        gdf = gpd.GeoDataFrame({c: [] for c in columns}, geometry=[], crs=crs)
        gdf.to_file(gpkg, driver="GPKG")


def write_agreement(
    layers: Sequence[Layer],
    finest_index: int,
    out_tif: Path,
    block_rows: int = 512,
    n_jobs: Optional[int] = 1,
) -> Dict[int, int]:
    """Write the agreement GeoTIFF on the finest grid; return ``{id: cell_count}``.

    The grid is processed in bands of ``block_rows`` rows, optionally in parallel
    (``n_jobs`` threads; ``-1`` = all CPUs). Output is identical for any setting.
    """
    fine = layers[finest_index]
    dtype = pattern_dtype(len(layers))
    maps = [overlap_maps(fine, layer) for layer in layers]
    counts: Dict[int, int] = {}
    profile = {
        "driver": "GTiff",
        "height": fine.height,
        "width": fine.width,
        "count": 1,
        "dtype": dtype.name,
        "crs": fine.crs,
        "transform": fine.transform,
        "nodata": 0,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if block_rows < 1:
        raise ValueError("block_rows must be at least 1.")
    workers = _resolve_jobs(n_jobs)
    starts = list(range(0, fine.height, block_rows))
    slices = [slice(s, min(s + block_rows, fine.height)) for s in starts]

    def work(rows: slice) -> np.ndarray:
        return compute_agreement_block(rows, layers, maps, dtype)

    with rasterio.open(out_tif, "w", **profile) as dst:
        # Blocks are independent; compute `workers` at a time (threads share the
        # in-memory layers) and write sequentially so memory stays bounded.
        for i in range(0, len(slices), workers):
            batch = slices[i : i + workers]
            if workers > 1 and len(batch) > 1:
                from joblib import Parallel, delayed

                blocks = Parallel(n_jobs=workers, backend="threading")(
                    delayed(work)(rows) for rows in batch
                )
            else:
                blocks = [work(rows) for rows in batch]
            for rows, block in zip(batch, blocks):
                dst.write(
                    block,
                    1,
                    window=rasterio.windows.Window(
                        0, rows.start, fine.width, block.shape[0]
                    ),
                )
                ids, n = np.unique(block[block > 0], return_counts=True)
                for pid, c in zip(ids.tolist(), n.tolist()):
                    counts[pid] = counts.get(pid, 0) + c
    return counts


def _resolve_jobs(n_jobs: Optional[int]) -> int:
    """Worker count: ``None``/1 -> 1, ``-1`` -> all CPUs, else the given count."""
    import os

    if n_jobs is None or n_jobs == 1:
        return 1
    if n_jobs == -1:
        return os.cpu_count() or 1
    if n_jobs < 1:
        raise ValueError("n_jobs must be -1 or a positive integer.")
    return int(n_jobs)


def write_legend(
    path: Path, layers: Sequence[Layer], counts: Dict[int, int]
) -> None:
    """Write grouping sequences and their fractions of valid agreement cells."""
    n = len(layers)
    total = bell_number(n)
    valid_count = sum(counts.values())
    ids = range(1, total + 1) if total <= LEGEND_FULL_LIMIT else sorted(counts)
    order = "|".join(f"{i}:{l.label}" for i, l in enumerate(layers, start=1))
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["pattern_id", "grouping_sequence", "compact_sequence", "n_groups",
             "cell_count", "cell_proportion", "input_order"]
        )
        for pid in ids:
            seq = sequence_from_id(pid, n)
            compact = format_sequence(seq, compact=True)
            writer.writerow([
                pid,
                format_sequence(seq),
                compact if "-" not in compact else "",
                max(seq),
                counts.get(pid, 0),
                counts.get(pid, 0) / valid_count if valid_count else None,
                order,
            ])


def compare_landcover(
    raster_paths: Sequence,
    out_dir,
    width_m: float = 100.0,
    n_samples: int = 100,
    seed: Optional[int] = 42,
    search_limit: Optional[int] = None,
    reproject_to_finest: bool = True,
    classes: Optional[Iterable[int]] = None,
    prefix: str = "landcover",
    n_jobs: Optional[int] = 1,
    block_rows: int = 512,
    labels: Optional[Sequence[str]] = None,
) -> CompareResult:
    """Run the full comparison workflow and write all outputs to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    layers, finest = load_layers(raster_paths, reproject_to_finest, labels)
    validate_class_codes(layers, classes)
    selection = select_squares(layers, finest, width_m, n_samples, seed, search_limit)
    tif, legend = (
        out / f"{prefix}_agreement.tif",
        out / f"{prefix}_agreement_legend.csv",
    )
    counts = write_agreement(layers, finest, tif, block_rows, n_jobs)
    write_legend(legend, layers, counts)
    records = square_records(layers, selection.centers, width_m)
    fine = layers[finest]
    with rasterio.open(tif) as src:
        for record, (cx, cy) in zip(records, selection.centers):
            half = width_m / 2.0
            r0, r1, c0, c1 = centroid_window(
                fine, cx - half, cy - half, cx + half, cy + half
            )
            r0, c0 = max(0, r0), max(0, c0)
            r1, c1 = min(fine.height - 1, r1), min(fine.width - 1, c1)
            if r0 <= r1 and c0 <= c1:
                block = src.read(
                    1, window=rasterio.windows.Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
                )
                valid = block[block > 0]
            else:
                valid = np.array([], dtype=np.uint8)
            fraction = (
                float(np.count_nonzero(valid == 1) / valid.size) if valid.size else None
            )
            record["full_agreement_fraction"] = fraction
            record["disagreement_fraction"] = (
                1.0 - fraction if fraction is not None else None
            )
    columns = ["square_id", "center_x", "center_y", "xmin", "ymin", "xmax", "ymax"]
    for layer in layers:
        columns.extend(
            f"{layer.label}_{suffix}"
            for suffix in (
                "majority",
                "shannon_diversity_index",
                "joint_entropy",
                "n_cells",
            )
        )
    columns.extend(["full_agreement_fraction", "disagreement_fraction"])
    gpkg, csv_path = out / f"{prefix}_squares.gpkg", out / f"{prefix}_squares.csv"
    _write_squares(records, fine.crs, gpkg, csv_path, columns)
    return CompareResult(
        selection, gpkg, csv_path, tif, legend, list(layers), finest, counts
    )
