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
