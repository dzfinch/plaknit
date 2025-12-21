# 🧶 plaknit


[![image](https://img.shields.io/pypi/v/plaknit.svg)](https://pypi.python.org/pypi/plaknit)
[![image](https://img.shields.io/conda/vn/conda-forge/plaknit.svg)](https://anaconda.org/conda-forge/plaknit)


**Processing Large-Scale PlanetScope Data**

> Note: plaknit is in active early-stage development. Expect frequent updates, and please share feedback or ideas through the GitHub Issues tab.

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

## Masking & Mosaicking CLI (stitch)

When the SR scenes land, run the bundled stitch driver (no extra scripting
required). Point it at the clipped strips, their UDMs, and the desired output
path; the command handles GDAL masking + Orfeo Toolbox mosaicking with parallel
workers, RAM hints, and concise progress bars (Mask tiles → Binary mask → Mosaic):

```bash
plaknit stitch \
  --inputs /data/planet/strips/*.tif \
  --udms /data/planet/strips/*.udm2.tif \
  --output /data/mosaics/planet_mosaic_2024.tif \
  --sr-bands 8 \
  --ndvi \
  --jobs 8 \
  --ram 196608
```

Customize `--jobs`, `--ram`, or `--workdir/--tmpdir` as needed for your local or
HPC environment. You can also invoke it as `plaknit mosaic` for backward compatibility.
Pass `--ndvi` to append NDVI (bands 4/3 for 4-band SR, 8/6 for 8-band SR) to the
output mosaic.

## Planning & Ordering Monthly Planet Composites (Beta)

`plaknit plan` runs on your laptop or login node to query Planet's STAC/Data
API, apply environmental filters (clouds, sun elevation), tile the AOI, and
select a minimal set of scenes per month that hit both coverage and clear
observation depth targets. The same command can immediately turn those plans
into Planet orders that deliver clipped surface reflectance scenes (4- or 8-band,
optionally harmonized to Sentinel-2) as single-archive ZIPs chunked into orders
of up to 100 scenes.

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
the AOI and optionally harmonized), push them through `plaknit stitch` (alias
`plaknit mosaic`) or future compositing tools on HPC to build median reflectance
mosaics.

Already have a stored plan JSON/GeoJSON? Submit the corresponding orders later
without replanning via:

```bash
plaknit order \
  --plan monthly_plan.json \
  --aoi aoi.gpkg \
  --sr-bands 4 \
  --harmonize-to none \
  --order-prefix plk_region01 \
  --archive-type zip
```

`plaknit order` reuses the original AOI for clip/harmonization settings,
applies optional harmonization, and prints a summary of each submitted order ID
(orders split into batches of ≤100 scenes with order/ZIP names suffixed `_1`,
`_2`, ... when needed).
