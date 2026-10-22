# 🧶 plaknit


[![image](https://img.shields.io/pypi/v/plaknit.svg)](https://pypi.python.org/pypi/plaknit)
[![image](https://img.shields.io/conda/vn/conda-forge/plaknit.svg)](https://anaconda.org/conda-forge/plaknit)


**Processing Large-Scale PlanetScope Data**

> Note: plaknit is fully operational as of v0.3.0. Continue to look for updates, and please share feedback or ideas through the GitHub Issues tab.

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
- Planning workflow that searches Planet's STAC/Data API and scores scenes, with ordering handled by a separate `plaknit order` workflow.
- GDAL-powered parallel masking of Planet strips with their UDM rasters.
- Tuned Orfeo Toolbox mosaicking pipeline with RAM hints for large jobs.
- Random Forest and Boosted Regression Tree ensemble training + inference utilities for classifying Planet stacks.

## Quick Start

### Train a BRT Ensemble Classifier

```bash
plaknit brt train \
  --image stack.tif \
  --labels training.gpkg \
  --label-column class_id \
  --output ./brt_ensemble/ \
  --n-models 5 \
  --gpu
```

### Apply the Ensemble to New Imagery

```bash
plaknit brt predict \
  --image new_stack.tif \
  --ensemble-dir ./brt_ensemble/ \
  --output-dir ./probability_outputs/ \
  --feature-importance-out predictor_importance.csv
```

See [Usage](https://dzfinch.github.io/plaknit/usage.html) for complete examples and all supported options.

### Compare Land-Cover Rasters

```bash
plaknit landcover \
  --rasters lc_3m.tif lc_10m.tif lc_30m.tif \
  --out-dir ./landcover_compare/ \
  --width 100 --samples 100 --seed 42
```

Two or more categorical rasters (integer, or float holding whole-number class codes; NaN/NoData cells are
treated as NoData) are compared in the given order; the
count and resolutions are not fixed. Options: `--search-limit N` (max candidate
centres examined, default `100 x samples`), `--classes 1 2 3` (allowed codes), `--no-reproject`
(error on CRS mismatch), `--n-jobs N` (threads for the agreement surface, `-1` = all CPUs;
output is identical for any value) and `--block-rows N` (rows per agreement block, default 512). Outputs, with prefix `landcover` by default:

- **CRS handling.** At least one raster must have a projected CRS in metres, and all grids must be north-up. The
  *finest projected* raster (smallest native cell area; first input on ties) defines the
  target CRS. Any raster in another CRS, including a geographic (degrees) one, is reprojected to it with nearest-neighbour
  resampling (class codes are never interpolated; cells without source coverage become
  NoData). Reprojected cells only approximate the original cells; rasters already in the
  target CRS are used untouched. Class sets must overlap (a warning is issued when they differ).
- **Square selection.** Candidate centres are finest-raster cell centroids whose
  square (axis-aligned, `--width` metres) fits inside the common raster extent. They are
  drawn in random order without replacement from `numpy.random.default_rng(seed)`
  (reproducible). A square is kept if it overlaps no kept square (touching edges is
  allowed) and every cell whose interior intersects it, including edge cells, is valid in
  every raster. At most the search limit of candidates is examined; if fewer squares than
  requested are found, the message reports the count found and the limit reached, which does
  not imply more squares are impossible.
- **`*_squares.gpkg` / `*_squares.csv`.** One row per square: `square_id`, `center_x`,
  `center_y`, `xmin`, `ymin`, `xmax`, `ymax` (GeoPackage also holds the polygon) and,
  per raster `rN_<name>`: `_majority`, `_entropy`, `_complexity`, `_n_cells`. Each raster
  is sampled at its **own native cell centroids** inside the square. A centroid exactly
  on the west or north edge is inside; on the east or south edge it is outside.
  - `majority`: most frequent class; ties go to the smallest class code.
  - `entropy`: Shannon entropy, `-sum(p * log2 p)` in bits, of class proportions.
  - `complexity`: total length (m) of shared edges between included, horizontally or
    vertically adjacent cells of different classes, divided by the square area (m²).
- **`*_agreement.tif`.** Categorical GeoTIFF on the finest raster's native grid (same
  transform, size and CRS), unsigned integer, NoData `0`. For each output cell and each
  other raster, the raster cell with the **largest overlap area** with the output cell is
  used (the finest raster contributes its own cell). If overlaps tie, the cell containing the
  output cell's centroid is used; a centroid exactly on a cell boundary resolves to the higher
  column / higher row (east / south). The output is NoData if any chosen cell is NoData or
  outside a raster's coverage. No nearest-neighbour resampling is used.
- **Pattern encoding.** Class values are read in input order and turned into a canonical
  grouping sequence: the first value is group `1`; each later value takes the group of an
  equal earlier value, otherwise the next unused number. `1,1,1,1,1` is five-way agreement,
  `1,1,2,2` two separate agreeing pairs, `1,2,3,4` four-way disagreement. The GeoTIFF stores
  a numeric **pattern ID**, the 1-based lexicographic rank of that sequence among all
  `Bell(n)` sequences for `n` rasters (so IDs do not depend on the data). The ID is a label
  only, never the sequence. The dtype is `uint8`, `uint16` or `uint32` as needed (at most 15 rasters).
- **`*_agreement_legend.csv`.** `pattern_id`, `grouping_sequence` (always `-`-separated,
  e.g. `1-1-2-2`), `compact_sequence` (e.g. `1122`, blank when any group number is 10 or more),
  `n_groups`, `cell_count`, `input_order` (`position:raster label`). All patterns are listed
  when `Bell(n) <= 10000`, otherwise only observed ones.

Rasters are read fully into memory.
