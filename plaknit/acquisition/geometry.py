"""Geometry utilities for plaknit."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from shapely.geometry import base as shapely_geom


def load_aoi_geometry(aoi_path: str) -> Tuple[shapely_geom.BaseGeometry, str | None]:
    """
    Load an AOI polygon/multipolygon from a vector file.

    Supports GeoJSON (.geojson / .json), ESRI Shapefile (.shp), and GeoPackage (.gpkg).

    Parameters
    ----------
    aoi_path : str
        Path to the AOI file.

    Returns
    -------
    geometry : shapely.geometry.BaseGeometry
        A single (multi)polygon geometry representing the AOI (dissolved if multiple
        features are present).
    crs : str or None
        The CRS of the input AOI as an EPSG code or PROJ string, if available.
    """

    gdf = gpd.read_file(aoi_path)
    if gdf.empty:
        raise ValueError(f"No geometries found in '{aoi_path}'.")

    geometry = gdf.unary_union
    crs_string = gdf.crs.to_string() if gdf.crs else "EPSG:4326"
    return geometry, crs_string


def reproject_geometry(
    geometry: shapely_geom.BaseGeometry,
    src_crs: str | None,
    dst_crs: str,
) -> shapely_geom.BaseGeometry:
    """Reproject a geometry to a new CRS."""

    source = src_crs or "EPSG:4326"
    if source == dst_crs:
        return geometry

    series = gpd.GeoSeries([geometry], crs=source)
    return series.to_crs(dst_crs).iloc[0]


def geometry_vertex_count(geometry: shapely_geom.BaseGeometry) -> int:
    """Return the number of coordinate vertices in a geometry (outer + inner rings)."""

    if geometry.is_empty:
        return 0

    geom_type = geometry.geom_type
    if geom_type == "Polygon":
        count = len(geometry.exterior.coords)
        count += sum(len(interior.coords) for interior in geometry.interiors)
        return count
    if geom_type == "MultiPolygon":
        return sum(geometry_vertex_count(part) for part in geometry.geoms)
    if geom_type in ("LineString", "LinearRing"):
        return len(geometry.coords)
    if geom_type == "MultiLineString":
        return sum(len(line.coords) for line in geometry.geoms)
    if geom_type == "Point":
        return 1
    if geom_type == "MultiPoint":
        return len(geometry.geoms)
    if geom_type == "GeometryCollection":
        return sum(geometry_vertex_count(part) for part in geometry.geoms)
    return 0


def simplify_geometry_to_vertex_limit(
    geometry: shapely_geom.BaseGeometry,
    max_vertices: int,
    *,
    max_iterations: int = 40,
    tolerance_factor: float = 2.0,
) -> shapely_geom.BaseGeometry:
    """
    Simplify an AOI geometry until the vertex count is within Planet's limits.

    Parameters
    ----------
    geometry : BaseGeometry
        Input geometry projected in the desired CRS.
    max_vertices : int
        Maximum allowed vertices (Planet currently restricts ROIs to 1500 vertices).
    max_iterations : int, optional
        Maximum number of simplification attempts (default: 40).
    tolerance_factor : float, optional
        Multiplier applied to the tolerance between iterations (default: 2.0).
    """

    if geometry.is_empty:
        return geometry

    current_count = geometry_vertex_count(geometry)
    if current_count <= max_vertices:
        return geometry

    minx, miny, maxx, maxy = geometry.bounds
    span = max(maxx - minx, maxy - miny)
    tolerance = max(span * 1e-4, 1e-9)
    candidate = geometry
    best_candidate = geometry
    best_count = current_count

    for _ in range(max_iterations):
        candidate = geometry.simplify(tolerance, preserve_topology=True)
        if candidate.is_empty:
            tolerance *= tolerance_factor
            continue

        candidate_count = geometry_vertex_count(candidate)
        if candidate_count <= max_vertices:
            return candidate

        if candidate_count < best_count:
            best_candidate = candidate
            best_count = candidate_count

        tolerance *= tolerance_factor

    if best_count <= max_vertices:
        return best_candidate

    raise ValueError(
        f"Unable to simplify geometry to <= {max_vertices} vertices "
        f"(best achieved {best_count})."
    )


def distance_to_vector(
    template_raster: str | Path,
    vector_path: str | Path,
    output_raster: str | Path,
    *,
    backend: str = "cpu",
) -> None:
    """Write distances from every template cell to vector features.

    The template controls the output extent, resolution, transform, and CRS.
    Distances use the template CRS map units. Vector and raster formats are
    those supported by GeoPandas and Rasterio, respectively.
    """

    if backend not in {"cpu", "gpu"}:
        raise ValueError("backend must be 'cpu' or 'gpu'")

    with rasterio.open(template_raster) as source:
        if source.crs is None:
            raise ValueError("Input raster has no CRS.")
        if source.crs.is_geographic:
            raise ValueError(
                "Input raster is geographic. Reproject it to a projected CRS "
                "before calculating Euclidean distances."
            )
        if source.transform.b != 0 or source.transform.d != 0:
            raise ValueError("Rotated or sheared rasters are not supported.")

        vector = gpd.read_file(vector_path)
        if vector.empty or vector.geometry.dropna().empty:
            raise ValueError(f"No geometries found in '{vector_path}'.")
        if vector.crs is None:
            raise ValueError("Input vector has no CRS.")
        vector = vector.to_crs(source.crs)
        geometries = [
            geometry
            for geometry in vector.geometry
            if geometry is not None and not geometry.is_empty
        ]
        if not geometries:
            raise ValueError(f"No non-empty geometries found in '{vector_path}'.")

        vector_mask = rasterize(
            ((geometry, 1) for geometry in geometries),
            out_shape=(source.height, source.width),
            transform=source.transform,
            fill=0,
            dtype="uint8",
        )
        non_vector = (vector_mask == 0).astype("uint8")
        sampling = (abs(source.transform.e), abs(source.transform.a))

        if backend == "gpu":
            try:
                import cupy as cp
                from cupyx.scipy.ndimage import (
                    distance_transform_edt as gpu_distance_transform_edt,
                )
            except ImportError as exc:
                raise ImportError(
                    "The GPU backend requires CuPy; install a CUDA-matched "
                    "cupy package or use backend='cpu'."
                ) from exc
            distances = cp.asnumpy(
                gpu_distance_transform_edt(cp.asarray(non_vector), sampling=sampling)
            )
        else:
            distances = distance_transform_edt(non_vector, sampling=sampling)

        distances = distances.astype("float32", copy=False)
        distances[vector_mask == 1] = 0.0
        profile = source.profile.copy()
        profile.update(count=1, dtype="float32", nodata=None)
        with rasterio.open(output_raster, "w", **profile) as destination:
            destination.write(distances, 1)


__all__ = [
    "load_aoi_geometry",
    "reproject_geometry",
    "geometry_vertex_count",
    "simplify_geometry_to_vertex_limit",
    "distance_to_vector",
]
