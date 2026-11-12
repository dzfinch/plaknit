import itertools
import math

import numpy as np
import pandas as pd
import pylandstats
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform

from plaknit.cli import compare_cli
from plaknit.processing import landcover_compare as lc

CRS = "EPSG:32618"
NODATA = 255


def write_raster(path, arr, res, x0, y_top, crs=CRS, nodata=NODATA):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=from_origin(x0, y_top, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(arr.astype("uint8"), 1)
    return str(path)


def blobs(shape, seed, n_classes=3):
    rng = np.random.default_rng(seed)
    coarse = rng.integers(1, n_classes + 1, size=(-(-shape[0] // 4), -(-shape[1] // 4)))
    return np.kron(coarse, np.ones((4, 4), dtype=int))[: shape[0], : shape[1]]


# (resolution, origin_x, origin_y_top, size_m) -- deliberately misaligned, all
# covering 0..180 m in x and y.
SPECS = [
    (3, 0.0, 180.0, 180),
    (10, -4.0, 184.0, 190),
    (30, -10.0, 200.0, 210),
    (6, -2.0, 182.0, 186),
    (15, -5.0, 185.0, 195),
]


def make_rasters(tmp_path, n):
    paths = []
    for i, (res, x0, ytop, size) in enumerate(SPECS[:n]):
        cells = int(size / res)
        paths.append(write_raster(tmp_path / f"r{i}.tif", blobs((cells, cells), i), res, x0, ytop))
    return paths


# ---------------------------------------------------------------- encoding
def test_canonical_groups_examples():
    cases = {
        (7, 7, 7, 7, 7): [1, 1, 1, 1, 1],
        (5, 5, 9, 9): [1, 1, 2, 2],
        (1, 2, 3, 4): [1, 2, 3, 4],
        (4, 8, 4, 8): [1, 2, 1, 2],
        (3, 4, 3, 5, 4): [1, 2, 1, 3, 2],
    }
    for values, expected in cases.items():
        got = lc.canonical_groups(np.array(values).reshape(-1, 1, 1))
        assert got[:, 0, 0].tolist() == expected


@pytest.mark.parametrize("n,bell", [(2, 2), (3, 5), (4, 15), (5, 52)])
def test_pattern_ids_unique_dense_and_invertible(n, bell):
    assert lc.bell_number(n) == bell
    seen = {}
    for values in itertools.product(range(n), repeat=n):
        groups = lc.canonical_groups(np.array(values).reshape(-1, 1))
        pid = int(lc.pattern_ids(groups)[0])
        seq = tuple(groups[:, 0].tolist())
        assert seen.setdefault(pid, seq) == seq
    assert sorted(seen) == list(range(1, bell + 1))
    for pid, seq in seen.items():
        assert lc.sequence_from_id(pid, n) == seq
    assert lc.sequence_from_id(1, n) == (1,) * n


def test_format_sequence_and_dtype():
    assert lc.format_sequence((1, 1, 1, 1, 1), compact=True) == "11111"
    assert lc.format_sequence((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), compact=True) == "1-2-3-4-5-6-7-8-9-10"
    assert lc.format_sequence((1, 1, 2)) == "1-1-2"
    assert lc.pattern_dtype(5) == np.uint8
    assert lc.pattern_dtype(7) == np.uint16
    assert lc.pattern_dtype(10) == np.uint32
    assert lc.pattern_dtype(16) == np.uint64
    with pytest.raises(ValueError, match="safely representable"):
        lc.pattern_dtype(26)


# ------------------------------------------------------------- overlap rule
def test_select_cells_largest_overlap_and_ties():
    # Fine cell mostly inside coarse cell 0 -> 0
    assert lc.select_cells_1d(np.array([0.0]), np.array([0.6]), 3).tolist() == [0]
    assert lc.select_cells_1d(np.array([0.7]), np.array([1.3]), 3).tolist() == [1]
    assert lc.select_cells_1d(np.array([0.4]), np.array([1.1]), 3).tolist() == [0]
    # Exact tie: midpoint on boundary -> higher index
    assert lc.select_cells_1d(np.array([0.85]), np.array([1.15]), 3).tolist() == [1]
    # Outside coverage / winning cell outside
    assert lc.select_cells_1d(np.array([-0.9]), np.array([0.1]), 3).tolist() == [-1]
    assert lc.select_cells_1d(np.array([2.5]), np.array([3.5]), 3).tolist() == [-1]
    assert lc.select_cells_1d(np.array([-0.2]), np.array([0.8]), 3).tolist() == [0]


def brute_cell(coarse_layer, fine_layer, row, col):
    """Independent rectangle-overlap search (ties -> higher row/col)."""
    x0 = fine_layer.left + col * fine_layer.res_x
    x1 = x0 + fine_layer.res_x
    y1 = fine_layer.top - row * fine_layer.res_y
    y0 = y1 - fine_layer.res_y
    best, pick = 0.0, None
    for r in range(coarse_layer.height):
        for c in range(coarse_layer.width):
            cx0 = coarse_layer.left + c * coarse_layer.res_x
            cy1 = coarse_layer.top - r * coarse_layer.res_y
            ox = min(x1, cx0 + coarse_layer.res_x) - max(x0, cx0)
            oy = min(y1, cy1) - max(y0, cy1 - coarse_layer.res_y)
            if ox <= 0 or oy <= 0:
                continue
            area = ox * oy
            if area > best + 1e-9 or (abs(area - best) <= 1e-9 and (r, c) > pick):
                best, pick = area, (r, c)
    return pick


def test_overlap_maps_match_bruteforce_on_misaligned_grids(tmp_path):
    paths = make_rasters(tmp_path, 3)
    layers, finest = lc.load_layers(paths)
    fine = layers[finest]
    for other in layers[1:]:
        row_map, col_map = lc.overlap_maps(fine, other)
        for row in range(0, fine.height, 7):
            for col in range(0, fine.width, 5):
                expected = brute_cell(other, fine, row, col)
                got = (row_map[row], col_map[col])
                if -1 in got:
                    got = None
                assert expected == (got if got is None else tuple(int(g) for g in got))


def test_overlap_not_nearest_neighbour():
    # Fine cell [2.9, 5.9] vs coarse 10 m: centre 4.4 -> cell 0 either way, but use
    # a case where the centroid cell differs from the max-overlap cell is
    # impossible in 1-D; verify the overlap winner on an off-centre interval.
    a, b = np.array([0.46]), np.array([1.2])  # overlap cell0=.54, cell1=.2
    assert lc.select_cells_1d(a, b, 2).tolist() == [0]


# ------------------------------------------------------------------ metrics
def make_layer(arr, res=1.0, x0=0.0, top=None, valid=None):
    top = arr.shape[0] * res if top is None else top
    valid = np.ones(arr.shape, bool) if valid is None else valid
    return lc.Layer(
        "x",
        "x",
        arr,
        valid,
        from_origin(x0, top, res, res),
        rasterio.crs.CRS.from_epsg(32618),
    )


def test_metrics_values():
    arr = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]])
    layer = make_layer(arr)
    m = lc.square_metrics(layer, 0, 0, 4, 4)
    assert m["majority"] == 1  # 8/8 tie -> smallest code
    expected = pylandstats.Landscape(
        arr, res=(1, 1), nodata=-999, neighborhood_rule="4"
    )
    assert m["shannon_diversity_index"] == pytest.approx(
        expected.shannon_diversity_index()
    )
    assert m["joint_entropy"] == pytest.approx(expected.joint_entropy())
    assert m["shannon_diversity_index"] == pytest.approx(math.log(2))
    assert m["joint_entropy"] != pytest.approx(4 / 16)
    uniform = make_layer(np.full((4, 4), 3))
    mu = lc.square_metrics(uniform, 0, 0, 4, 4)
    assert (mu["majority"], mu["shannon_diversity_index"], mu["joint_entropy"]) == (
        3,
        0.0,
        0.0,
    )
    checker = make_layer(np.indices((2, 2)).sum(0) % 2)
    checker_metrics = lc.square_metrics(checker, 0, 0, 2, 2)
    assert checker_metrics["shannon_diversity_index"] == pytest.approx(math.log(2))
    assert checker_metrics["joint_entropy"] == pytest.approx(1.0)
    thirds = make_layer(np.array([[1, 2, 3]]))
    assert lc.shannon_entropy([1, 1, 1]) == pytest.approx(math.log2(3))
    metrics = lc.square_metrics(thirds, 0, 0, 3, 1)
    expected = pylandstats.Landscape(
        thirds.data, res=(1, 1), nodata=-999, neighborhood_rule="4"
    )
    assert metrics["shannon_diversity_index"] == pytest.approx(
        expected.shannon_diversity_index()
    )
    assert metrics["joint_entropy"] == pytest.approx(expected.joint_entropy())
    assert lc.majority_class(np.array([5, 5, 2, 2, 9])) == 2


@pytest.mark.parametrize(
    "values",
    [
        [[0, 0, 1], [1, 2, 2]],
        [[-3, 0], [1, -3]],
        [[1, 1, 1], [1, 1, 2]],
    ],
)
def test_square_diversity_matches_pylandstats(values):
    arr = np.array(values)
    layer = make_layer(arr, res=3)
    actual = lc.square_metrics(layer, 0, 0, arr.shape[1] * 3, arr.shape[0] * 3)
    landscape = pylandstats.Landscape(
        arr, res=(3, 3), nodata=np.iinfo(np.int64).min, neighborhood_rule="4"
    )
    assert actual["shannon_diversity_index"] == pytest.approx(
        landscape.shannon_diversity_index()
    )
    assert actual["joint_entropy"] == pytest.approx(landscape.joint_entropy())


def test_single_cell_diversity_is_zero():
    layer = make_layer(np.array([[0]]))
    metrics = lc.square_metrics(layer, 0, 0, 1, 1)
    assert metrics["shannon_diversity_index"] == 0
    assert metrics["joint_entropy"] == 0


def test_native_centroids_per_raster_and_boundary_rule():
    layer = make_layer(np.arange(16).reshape(4, 4), res=1.0)
    # Centroids at .5, 1.5, 2.5 ... Square x in [0.5, 2.5]: west edge included, east excluded.
    r0, r1, c0, c1 = lc.centroid_window(layer, 0.5, 1.5, 2.5, 3.5)
    assert (c0, c1) == (0, 1)
    # y: north inclusive (3.5 -> row 0 centroid at y=3.5), south excluded (1.5)
    assert (r0, r1) == (0, 1)
    coarse = make_layer(np.zeros((2, 2), int), res=2.0)
    assert lc.centroid_window(coarse, 0, 0, 4, 4) == (0, 1, 0, 1)
    # Different resolutions give different native sample counts.
    fine = make_layer(np.zeros((8, 8), int), res=0.5)
    assert lc.square_metrics(fine, 0, 0, 4, 4)["n_cells"] == 64
    assert lc.square_metrics(coarse, 0, 0, 4, 4)["n_cells"] == 4


@pytest.mark.parametrize("shift", [0.0, 1.3, 4.9, 7.0, 9.99])
def test_square_cell_counts_are_shift_invariant(shift):
    # 300 m square: 3 m -> 10,000, 10 m -> 900, 30 m -> 100 centroids, wherever
    # the square sits relative to each raster's grid.
    expected = {3.0: 10000, 10.0: 900, 30.0: 100}
    for res, count in expected.items():
        n = int(1200 / res)
        layer = make_layer(np.ones((n, n), int), res=res, x0=0.0, top=1200.0)
        cx, cy = 600.0 + shift, 600.0 - shift * 0.7
        m = lc.square_metrics(layer, cx - 150, cy - 150, cx + 150, cy + 150)
        assert m["n_cells"] == count


def test_metre_raster_reprojection_keeps_native_cell_size(tmp_path, recwarn):
    # A 10 m raster in another projected CRS lands on a 10 m grid in the common
    # UTM CRS, so a 300 m square holds 900 centroids.
    x, y = warp_transform("EPSG:4326", "EPSG:3857", [-82.0], [27.0])
    merc = write_raster(
        tmp_path / "merc.tif", np.ones((500, 500)), 10, x[0], y[0], crs="EPSG:3857"
    )
    ux, uy = warp_transform("EPSG:4326", "EPSG:32617", [-82.0], [27.0])
    fine = write_raster(
        tmp_path / "fine.tif", np.ones((1000, 1000)), 3, ux[0], uy[0], crs="EPSG:32617"
    )
    layers, _ = lc.load_layers([fine, merc])
    reproj = layers[1]
    assert reproj.reprojected and reproj.res_x == pytest.approx(10.0)
    assert not [w for w in recwarn if "changed the cell size" in str(w.message)]
    cx, cy = reproj.left + 1500, reproj.top - 1500
    n = lc.square_metrics(reproj, cx - 150, cy - 150, cx + 150, cy + 150)["n_cells"]
    assert n == 900


def test_reprojection_preserving_cell_size_keeps_counts(tmp_path, recwarn):
    ux, uy = warp_transform("EPSG:4326", "EPSG:32617", [-82.0], [27.0])
    fine = write_raster(
        tmp_path / "fine.tif", np.ones((1000, 1000)), 3, ux[0], uy[0], crs="EPSG:32617"
    )
    other = write_raster(
        tmp_path / "s10.tif", np.ones((500, 500)), 10, ux[0], uy[0], crs="EPSG:26917"
    )
    layers, _ = lc.load_layers([fine, other])
    assert not [w for w in recwarn if "changed the cell size" in str(w.message)]
    s = layers[1]
    cx, cy = s.left + 1500, s.top - 1500
    assert lc.square_metrics(s, cx - 150, cy - 150, cx + 150, cy + 150)["n_cells"] == 900


def test_intersect_window_includes_edge_cells():
    layer = make_layer(np.zeros((10, 10), int), res=1.0)
    assert lc.intersect_window(layer, 2.5, 2.5, 5.5, 5.5)[2:] == (2, 5)
    assert lc.intersect_window(layer, 3.0, 3.0, 5.0, 5.0)[2:] == (3, 4)


# --------------------------------------------------------------- selection
def brute_valid(layers, xmin, ymin, xmax, ymax):
    for l in layers:
        for r in range(l.height):
            for c in range(l.width):
                cx0, cx1 = l.left + c * l.res_x, l.left + (c + 1) * l.res_x
                cy1 = l.top - r * l.res_y
                cy0 = cy1 - l.res_y
                if cx0 < xmax and cx1 > xmin and cy0 < ymax and cy1 > ymin:
                    if not l.valid[r, c]:
                        return False
    return True


def test_selection_validity_nonoverlap_nodata_and_reproducible(tmp_path):
    paths = make_rasters(tmp_path, 3)
    # Punch NoData holes, one in each of two rasters.
    for p, (r, c) in zip(paths[:2], [(20, 20), (9, 3)]):
        with rasterio.open(p, "r+") as ds:
            a = ds.read(1)
            a[r, c] = NODATA
            ds.write(a, 1)
    layers, finest = lc.load_layers(paths)
    sel = lc.select_squares(layers, finest, 30, 8, seed=3)
    again = lc.select_squares(layers, finest, 30, 8, seed=3)
    other = lc.select_squares(layers, finest, 30, 8, seed=4)
    assert sel.centers == again.centers
    assert sel.centers != other.centers
    assert sel.found > 0
    xmin, ymin, xmax, ymax = lc.common_bounds(layers)
    for (cx, cy) in sel.centers:
        assert cx - 15 >= xmin - 1e-9 and cx + 15 <= xmax + 1e-9
        assert cy - 15 >= ymin - 1e-9 and cy + 15 <= ymax + 1e-9
        assert brute_valid(layers, cx - 15, cy - 15, cx + 15, cy + 15)
        # centre is a finest-grid centroid
        fine = layers[finest]
        assert ((cx - fine.left) / fine.res_x - 0.5) == pytest.approx(round((cx - fine.left) / fine.res_x - 0.5))
    for a, b in itertools.combinations(sel.centers, 2):
        assert not lc.squares_overlap(a, b, 30)
        assert abs(a[0] - b[0]) >= 30 - 1e-9 or abs(a[1] - b[1]) >= 30 - 1e-9


def test_float_raster_with_whole_number_classes_and_nan_nodata(tmp_path):
    arr = np.full((10, 10), 2.0, "float32")
    arr[0, 0] = np.nan
    p = tmp_path / "f.tif"
    with rasterio.open(p, "w", driver="GTiff", height=10, width=10, count=1,
                       dtype="float32", crs=CRS, transform=from_origin(0, 30, 3, 3),
                       nodata=np.nan) as d:
        d.write(arr, 1)
    other = write_raster(tmp_path / "o.tif", np.full((3, 3), 2), 10, 0, 30)
    layers, _ = lc.load_layers([str(p), other])
    assert layers[0].data.dtype.kind == "i"
    assert not layers[0].valid[0, 0] and layers[0].valid[1, 1]
    assert lc.validate_class_codes(layers) == [2]


def test_edge_cell_nodata_rejects_square():
    arr = np.ones((10, 10), int)
    valid = np.ones((10, 10), bool)
    valid[5, 7] = False  # cell x 7..8, y 4..5
    layer = make_layer(arr, valid=valid)
    # Square touching the bad cell's interior only at its edge region.
    assert not lc.square_is_valid([layer], 7.5, 4.5, 9.5, 6.5)
    assert lc.square_is_valid([layer], 8.0, 5.0, 10.0, 7.0)  # only touches the corner


def test_search_limit_reported_without_impossibility_claim(tmp_path):
    paths = make_rasters(tmp_path, 2)
    layers, finest = lc.load_layers(paths)
    sel = lc.select_squares(layers, finest, 60, 50, seed=1, search_limit=40)
    assert sel.found < 50
    assert sel.search_limit == 40 and sel.draws == 40
    assert sel.limit_reached
    assert "search limit: 40" in sel.message
    assert "Found" in sel.message and "does not show" in sel.message


def test_search_limit_must_be_positive(tmp_path):
    paths = make_rasters(tmp_path, 2)
    layers, finest = lc.load_layers(paths)
    with pytest.raises(ValueError, match="search_limit must be at least 1"):
        lc.select_squares(layers, finest, 60, 2, search_limit=0)


def test_square_width_must_be_finite(tmp_path):
    paths = make_rasters(tmp_path, 2)
    layers, finest = lc.load_layers(paths)
    with pytest.raises(ValueError, match="width_m must be positive"):
        lc.select_squares(layers, finest, float("nan"), 2)


def test_width_smaller_than_coarse_cell_errors(tmp_path):
    layers, finest = lc.load_layers(make_rasters(tmp_path, 3))
    with pytest.raises(ValueError):
        lc.select_squares(layers, finest, 20, 5)


# --------------------------------------------------------------- validation
def test_validation_errors(tmp_path):
    a = write_raster(tmp_path / "a.tif", np.ones((10, 10)), 3, 0, 30)
    with pytest.raises(ValueError):
        lc.load_layers([a])
    geo = tmp_path / "geo.tif"
    with rasterio.open(geo, "w", driver="GTiff", height=2, width=2, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(0, 1, 0.1, 0.1)) as d:
        d.write(np.ones((2, 2), "uint8"), 1)
    with pytest.raises(ValueError, match="At least one"):
        lc.load_layers([str(geo), str(geo)])
    flt = tmp_path / "f.tif"
    with rasterio.open(flt, "w", driver="GTiff", height=2, width=2, count=1, dtype="float32",
                       crs=CRS, transform=from_origin(0, 6, 3, 3)) as d:
        d.write(np.full((2, 2), 1.5, "float32"), 1)
    with pytest.raises(ValueError, match="non-integer"):
        lc.load_layers([a, str(flt)])
    b = write_raster(tmp_path / "b.tif", np.full((10, 10), 2), 10, 0, 100)
    layers, _ = lc.load_layers([a, b])
    with pytest.raises(ValueError, match="share no class"):
        lc.validate_class_codes(layers)
    c = write_raster(tmp_path / "c.tif", np.full((10, 10), 1), 10, 0, 100)
    layers, _ = lc.load_layers([a, c])
    with pytest.raises(ValueError, match="allowed"):
        lc.validate_class_codes(layers, classes=[2])
    far = write_raster(tmp_path / "far.tif", np.ones((10, 10)), 10, 5000, 5100)
    layers, _ = lc.load_layers([a, far])
    with pytest.raises(ValueError, match="common coverage"):
        lc.common_bounds(layers)


# ------------------------------------------------------------- reprojection
def _utm_pair(tmp_path):
    lon, lat = -77.9, 40.0
    (x18,), (y18,) = warp_transform("EPSG:4326", "EPSG:32618", [lon], [lat])
    (x17,), (y17,) = warp_transform("EPSG:4326", "EPSG:32617", [lon], [lat])
    fine = write_raster(tmp_path / "fine.tif", np.full((100, 100), 1), 3, x18, y18 + 300)
    coarse = write_raster(
        tmp_path / "coarse.tif", np.full((60, 60), 1), 10, x17 - 150, y17 + 450, crs="EPSG:32617"
    )
    return fine, coarse


def test_reprojects_to_finest_crs(tmp_path):
    fine, coarse = _utm_pair(tmp_path)
    layers, finest = lc.load_layers([coarse, fine])  # finest is second input
    assert finest == 1
    assert all(l.crs == layers[finest].crs for l in layers)
    assert layers[0].reprojected and not layers[1].reprojected
    assert set(np.unique(layers[0].data[layers[0].valid])) == {1}
    assert layers[0].valid.any()
    assert abs(layers[0].res_x - 10) < 1.0
    with pytest.raises(ValueError, match="reprojection is disabled"):
        lc.load_layers([coarse, fine], reproject_to_finest=False)


def test_geographic_raster_is_reprojected_to_projected_finest(tmp_path):
    lon, lat = -77.9, 40.0
    (x,), (y,) = warp_transform("EPSG:4326", CRS, [lon], [lat])
    fine = write_raster(tmp_path / "fine.tif", np.full((200, 200), 1), 3, x, y + 600)
    deg = 30 / 111_000.0  # ~30 m
    geo = tmp_path / "geo.tif"
    with rasterio.open(geo, "w", driver="GTiff", height=40, width=40, count=1,
                       dtype="uint8", crs="EPSG:4326", nodata=NODATA,
                       transform=from_origin(lon - 5 * deg, lat + 25 * deg, deg, deg)) as d:
        d.write(np.ones((40, 40), "uint8"), 1)
    layers, finest = lc.load_layers([str(geo), fine])
    assert finest == 1
    assert layers[0].reprojected and layers[0].crs == layers[1].crs
    assert 20 < layers[0].res_x < 40
    res = lc.compare_landcover([str(geo), fine], tmp_path / "o", width_m=60, n_samples=3)
    assert res.selection.found > 0
    with pytest.raises(ValueError, match="reprojection is disabled"):
        lc.load_layers([str(geo), fine], reproject_to_finest=False)


def test_end_to_end_with_reprojection(tmp_path):
    fine, coarse = _utm_pair(tmp_path)
    res = lc.compare_landcover([fine, coarse], tmp_path / "o", width_m=30, n_samples=3, seed=1)
    with rasterio.open(res.agreement_tif) as ds:
        arr = ds.read(1)
        assert ds.crs.to_epsg() == 32618
    assert set(np.unique(arr)) <= {0, 1}  # all agree (ID 1) or NoData
    assert (arr == 1).any()


# ----------------------------------------------------------- end-to-end
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_end_to_end_outputs(tmp_path, n):
    paths = make_rasters(tmp_path, n)
    out = tmp_path / "out"
    res = lc.compare_landcover(paths, out, width_m=30, n_samples=6, seed=9)
    layers, finest = res.layers, 0
    fine = layers[finest]

    # CSV + GPKG
    df = pd.read_csv(res.squares_csv)
    import geopandas as gpd

    gdf = gpd.read_file(res.squares_gpkg)
    assert len(df) == len(gdf) == res.selection.found > 0
    assert df["square_id"].tolist() == list(range(1, len(df) + 1))
    for l in layers:
        for suffix in (
            "majority",
            "shannon_diversity_index",
            "joint_entropy",
            "n_cells",
        ):
            assert f"{l.label}_{suffix}" in df.columns
            assert f"{l.label}_{suffix}" in gdf.columns
        assert f"{l.label}_entropy" not in df.columns
        assert f"{l.label}_complexity" not in df.columns
        for suffix in ("shannon_diversity_index", "joint_entropy"):
            assert df[f"{l.label}_{suffix}"].to_numpy() == pytest.approx(
                gdf[f"{l.label}_{suffix}"].to_numpy()
            )
        for row in df.itertuples(index=False):
            r0, r1, c0, c1 = lc.centroid_window(
                l, row.xmin, row.ymin, row.xmax, row.ymax
            )
            block = l.data[r0 : r1 + 1, c0 : c1 + 1]
            if np.unique(block).size < 2:
                shannon, joint = 0.0, 0.0
            else:
                landscape = pylandstats.Landscape(
                    block,
                    res=(l.res_x, l.res_y),
                    nodata=np.iinfo(np.int64).min,
                    neighborhood_rule="4",
                )
                shannon, joint = (
                    landscape.shannon_diversity_index(),
                    landscape.joint_entropy(),
                )
            assert getattr(row, f"{l.label}_shannon_diversity_index") == pytest.approx(
                shannon
            )
            assert getattr(row, f"{l.label}_joint_entropy") == pytest.approx(joint)
    assert all(abs(g.area - 900) < 1e-6 for g in gdf.geometry)
    assert gdf.crs.to_epsg() == 32618

    # Agreement raster on the finest native grid
    with rasterio.open(res.agreement_tif) as ds:
        assert ds.transform == fine.transform
        assert (ds.height, ds.width) == (fine.height, fine.width)
        assert ds.nodata == 0
        assert ds.dtypes[0] == "uint8"
        arr = ds.read(1)
    legend = pd.read_csv(res.legend_csv, dtype={"compact_sequence": str})
    assert len(legend) == lc.bell_number(n)
    seqs = {
        int(r.pattern_id): tuple(int(x) for x in r.grouping_sequence.split("-"))
        for r in legend.itertuples()
    }
    assert legend["cell_count"].sum() == (arr > 0).sum()
    assert legend["cell_proportion"].sum() == pytest.approx(1.0)
    for row in legend.itertuples():
        count = np.count_nonzero(arr == row.pattern_id)
        assert row.cell_count == count
        assert row.cell_proportion == pytest.approx(count / np.count_nonzero(arr))

    # Verify sampled pixels against brute-force overlap selection.
    rng = np.random.default_rng(0)
    for _ in range(40):
        row, col = int(rng.integers(fine.height)), int(rng.integers(fine.width))
        vals, ok = [], True
        for l in layers:
            pick = brute_cell(l, fine, row, col) if l is not fine else (row, col)
            if pick is None or not l.valid[pick]:
                ok = False
                break
            vals.append(int(l.data[pick]))
        if not ok:
            assert arr[row, col] == 0
            continue
        expected = tuple(lc.canonical_groups(np.array(vals).reshape(-1, 1))[:, 0].tolist())
        assert seqs[int(arr[row, col])] == expected


def test_nodata_and_outside_coverage_in_agreement(tmp_path):
    fine = write_raster(tmp_path / "f.tif", np.ones((20, 20)), 3, 0, 60)
    coarse_arr = np.ones((4, 4), int)
    coarse_arr[0, 0] = NODATA
    # coarse covers x 0..40 only partially; the right part of the fine grid is outside.
    coarse = write_raster(tmp_path / "c.tif", coarse_arr, 10, 0, 60)
    layers, finest = lc.load_layers([fine, coarse])
    out = tmp_path / "a.tif"
    lc.write_agreement(layers, finest, out)
    with rasterio.open(out) as ds:
        arr = ds.read(1)
    assert (arr[0:3, 0:3] == 0).all()  # inside NoData coarse cell
    assert (arr[:, 14:] == 0).all()  # fine cols >= 42 m are outside coverage
    assert arr[10, 5] == 1


def test_legend_proportions_include_zero_patterns_and_handle_empty_counts(tmp_path):
    layers = [make_layer(np.ones((2, 2), dtype=int)) for _ in range(3)]
    path = tmp_path / "legend.csv"
    lc.write_legend(path, layers, {1: 3, 3: 1})
    legend = pd.read_csv(path).set_index("pattern_id")
    assert len(legend) == 5
    assert legend.loc[1, "cell_proportion"] == pytest.approx(0.75)
    assert legend.loc[3, "cell_proportion"] == pytest.approx(0.25)
    assert legend.loc[2, "cell_proportion"] == 0
    assert legend["cell_proportion"].sum() == pytest.approx(1)

    lc.write_legend(path, layers, {})
    empty = pd.read_csv(path)
    assert (empty["cell_count"] == 0).all()
    assert empty["cell_proportion"].isna().all()


def test_observed_only_legend_proportions(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "LEGEND_FULL_LIMIT", 1)
    layers = [make_layer(np.ones((2, 2), dtype=int)) for _ in range(3)]
    path = tmp_path / "legend.csv"
    lc.write_legend(path, layers, {1: 2, 3: 6})
    legend = pd.read_csv(path).set_index("pattern_id")
    assert list(legend.index) == [1, 3]
    assert legend.loc[1, "cell_proportion"] == pytest.approx(0.25)
    assert legend.loc[3, "cell_proportion"] == pytest.approx(0.75)


def test_separate_agreeing_pairs_encoded(tmp_path):
    arrs = [np.full((4, 4), v) for v in (1, 1, 2, 2)]
    paths = [write_raster(tmp_path / f"p{i}.tif", a, 10 + i, 0, 40 + 10 * i) for i, a in enumerate(arrs)]
    layers, finest = lc.load_layers(paths)
    out = tmp_path / "pairs.tif"
    lc.write_agreement(layers, finest, out)
    lc.write_legend(tmp_path / "leg.csv", layers, {})
    with rasterio.open(out) as ds:
        pid = int(ds.read(1)[0, 0])
    assert lc.sequence_from_id(pid, 4) == (1, 1, 2, 2)


@pytest.mark.parametrize("n_jobs,block_rows", [(1, 7), (3, 7), (-1, 1000), (2, 13)])
def test_blocks_and_parallel_match_serial(tmp_path, n_jobs, block_rows):
    layers, finest = lc.load_layers(make_rasters(tmp_path, 4))
    ref, out = tmp_path / "ref.tif", tmp_path / "out.tif"
    c_ref = lc.write_agreement(layers, finest, ref, block_rows=512, n_jobs=1)
    c_out = lc.write_agreement(layers, finest, out, block_rows=block_rows, n_jobs=n_jobs)
    with rasterio.open(ref) as a, rasterio.open(out) as b:
        assert (a.read(1) == b.read(1)).all()
    assert c_ref == c_out
    with pytest.raises(ValueError):
        lc.write_agreement(layers, finest, out, n_jobs=0)


def test_cli(tmp_path, capsys):
    paths = make_rasters(tmp_path, 3)
    args = ["spatial"]
    for i, path in enumerate(paths):
        label = f"input{i + 1}"
        args.extend(["--input", f"{label}={path}"])
    args.extend(
        [
            "--out-dir",
            str(tmp_path / "cli"),
            "--width",
            "30",
            "--samples",
            "3",
            "--seed",
            "1",
        ]
    )
    code = compare_cli.main(args)
    assert code == 0
    assert "Selected" in capsys.readouterr().out
    assert (tmp_path / "cli" / "compare_agreement.tif").exists()
