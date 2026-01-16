# 🧶 plaknit


[![image](https://img.shields.io/pypi/v/plaknit.svg)](https://pypi.python.org/pypi/plaknit)
[![image](https://img.shields.io/conda/vn/conda-forge/plaknit.svg)](https://anaconda.org/conda-forge/plaknit)


**Processing Large-Scale PlanetScope Data**

> Note: plaknit is in active early-stage development. Expect frequent updates, and please share feedback or ideas through the GitHub Issues tab.

PlanetScope Scene (PSS) data are reveared for its quality and distinct ability to
balance spatial and temporal resolution in Earth Observation data. While PSS has
proven itself a valuable asset in monitoring small-scale areas, the literature
has pointed out the shortcomings when creating a single image from individual tiles
(Frazier & Hemingway, 2021).

`plaknit` bundles the workflow I use to operationalize large-area mosaics so
you can run the same process locally or in an HPC environment. The goal is to
spend time answering big questions, not making a big mess of your data.

- Free software: MIT License
- Documentation: https://dzfinch.github.io/plaknit

<p align="center">
  <img src="docs/images/flowchart.png" alt="plaknit logo" width="720">
</p>


## Features

- CLI + Python API that scale from local experimentation to HPC batch runs.
- Planning workflow that searches Planet's STAC/Data API, scores scenes, and (optionally) submits Orders API requests for clipped SR bundles.
- GDAL-powered parallel masking of Planet strips with their UDM rasters.
- Tuned Orfeo Toolbox mosaicking pipeline with RAM hints for large jobs.
- Random Forest training + inference utilities for classifying Planet stacks.
