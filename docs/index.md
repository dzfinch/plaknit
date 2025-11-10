# Welcome to plaknit

<p align="center">
  <img src="images/plaknit_logo.png" alt="plaknit logo" width="280">
</p>


[![image](https://img.shields.io/pypi/v/plaknit.svg)](https://pypi.python.org/pypi/plaknit)


**Processing Large-Scale PlanetScope Data**


-   Free software: MIT License
-   Documentation: <https://dzfinch.github.io/plaknit>


## Why plaknit exists

PlanetScope's daily cadence is unmatched for answering regional and global
questions, but their strip-based acquisition leaves messy seams, clouds, and
nodata gaps once you move beyond postcard-sized areas. Getting to a clean,
analysis-ready mosaic typically involves a pile of ad-hoc shell scripts that
rarely survive outside a single project notebook.

`plaknit` bundles the workflow I use to operationalize large-area mosaics so
you can run the same process locally or on an HPC scheduler where GDAL,
rasterio, and Orfeo Toolbox already live. The goal is to spend time answering
planet-scale questions -- not chasing temp files or re-learning OTB flags.

## Features

-   Mask PlanetScope strips against their UDM rasters using efficient GDAL workflows.
-   Build seamless mosaics with pre-tuned Orfeo Toolbox parameters and RAM hints.
-   Run everything from a single CLI (`plaknit`) that works cross-platform.
-   Train and apply Random Forest classifiers on multi-band stacks directly from Python.
