"""Geometry utilities for plaknit."""

from __future__ import annotations

from typing import Tuple

import geopandas as gpd
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


__all__ = ["load_aoi_geometry", "reproject_geometry"]
