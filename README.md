# plaknit


[![image](https://img.shields.io/pypi/v/plaknit.svg)](https://pypi.python.org/pypi/plaknit)
[![image](https://img.shields.io/conda/vn/conda-forge/plaknit.svg)](https://anaconda.org/conda-forge/plaknit)


**Processing Large-Scale PlanetScope Data**

- Planet data is phenomenal for tracking change, but the current acquisition
  strategy sprays dozens of narrow strips across a scene. Without careful
  masking and mosaicking, even "cloud free" searches still include haze,
  seams, and nodata gaps.
- PlanetScope scenes are also huge. Building clean, analysis-ready products
  requires an automated workflow that can run on laptops _or_ HPC clusters
  where GDAL, rasterio, and Orfeo Toolbox are already available.
- `plaknit` packages the masking + mosaicking flow I rely on for regional
  mapping so the Planet community can stitch together reliable time series
  without copying shell scripts from old notebooks.

- Free software: MIT License
- Documentation: https://dzfinch.github.io/plaknit


## Features

- GDAL-powered parallel masking of Planet strips with their UDM rasters.
- Tuned Orfeo Toolbox mosaicking pipeline with RAM hints for large jobs.
- CLI + Python API that scale from local experimentation to HPC batch runs.
- Raster analysis helpers (e.g., normalized difference indices) built on rasterio.
- Random Forest training + inference utilities for classifying Planet stacks.
- Planning workflow that searches Planet's STAC/Data API, scores scenes, and (optionally) submits Orders API requests for clipped SR bundles.

## Planning & Ordering Monthly Planet Composites

`plaknit plan` runs on your laptop or login node to query Planet's STAC/Data
API, apply environmental filters (clouds, sun elevation), tile the AOI, and
select a minimal set of scenes per month that hit both coverage and clear
observation depth targets. The same command can immediately turn those plans
into Planet orders that deliver clipped surface reflectance scenes (4- or 8-band,
optionally harmonized to Sentinel-2) as one ZIP per scene/bundle.

```bash
plaknit plan \
  --aoi aoi.gpkg \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --cloud-max 0.1 \
  --sun-elev-min 35 \
  --coverage-target 0.98 \
  --min-clear-fraction 0.8 \
  --min-clear-obs 3 \
  --tile-size-m 1000 \
  --sr-bands 8 \
  --harmonize-to sentinel2 \
  --out monthly_plan.json \
  --order \
  --order-prefix plk_region01
```

Planning + ordering stay on the non-HPC side; once scenes arrive (clipped to
the AOI and optionally harmonized), push them through `plaknit mosaic` or future
compositing tools on HPC to build median reflectance mosaics.
