# Usage

## Land-cover comparison

Land-cover comparison has two classification modes plus a reflectance command.
Spatial mode compares classifications from different sources (for example
different sensors); temporal mode compares classifications from different
dates. Each input has a unique label, and the label identifies the input in
tables, plots, and metadata. Temporal inputs additionally require a keyed date.
This avoids positional metadata ambiguity for three or more inputs.

### Spatial mode

```bash
plaknit compare spatial \
  --input planet=planet_class.tif \
  --input sentinel=sentinel_class.tif \
  --out-dir ./spatial_compare/ \
  --width 100 --samples 100 --seed 42
```

Spatial mode does not take dates. Tables and plots are grouped by input label,
so choose labels that name the sensor or source.

### Temporal mode

```bash
plaknit compare temporal \
  --input june=class_2023_06.tif --date june=2023-06-15 \
  --input july=class_2024_07.tif --date july=2024-07-15 \
  --group-by year \
  --out-dir ./temporal_compare/ \
  --width 100 --samples 100 --seed 42
```

All inputs must have distinct ISO dates.
`--group-by date|year` controls temporal table and plot labels (default:
`date`).

### Inputs and analysis options

- Repeat `--input LABEL=CLASSIFICATION.tif` for each input. Labels must be
  unique. Temporal mode also requires `--date LABEL=YYYY-MM-DD` for each
  input; spatial mode does not accept or record dates. Input order is retained
  in outputs.
- `--width` is square width in metres (default 100); `--samples` is the number
  of non-overlapping squares requested (default 100); `--seed` controls
  reproducible selection and sampling (default 42). `--search-limit` caps
  candidate centers (default `100 x samples`). If the limit is reached before
  finding the requested count, the message reports the count and limit; this
  does not prove that more squares are impossible.
- `--classes` restricts allowed classification codes; `--no-reproject`
  rejects classification CRS mismatches; `--n-jobs` controls agreement
  workers (`-1` uses all CPUs); `--block-rows` sets agreement raster block
  processing.
- `--patch-min-area-m2` filters exported disagreement patches (default 0,
  retaining all sizes). All patches use four-neighbor connectivity; NoData
  separates patches.

At least two rasters are required. Classifications must contain integer class
codes (or whole-number floats), have north-up grids, and be in CRS units
suitable for area and distance calculations. The comparison aligns differing
CRSs to the finest projected input in metres with nearest-neighbor resampling.
Class meanings are **not** inferred or verified: matching numeric codes are
assumed comparable, so supply classifications with a shared code scheme.
Different class-code sets are reported; inputs with no shared codes are
rejected.

### Common footprint and square metrics

The common classification footprint is the intersection of input extents.
For whole-raster metrics, each classification uses its native grid: a native
cell is included when its centroid is in that extent and the containing cell
at the same point is valid in every other classification. The per-input valid
area and exclusions are recorded. Invalid pixels in other rasters never change
classification-only metrics or agreement.

Square centers are randomly selected from finest-grid cell centroids. Each
square is fully inside the common extent, does not overlap another selected
square (touching is allowed), and is accepted only when every classification
cell whose interior intersects it is valid. Square-level landscape metrics
and reflectance summaries retain the square ID. Square class metrics use each
classification's native cells whose centroids lie in that square; west and
north centroid boundaries are included, east and south are excluded. Existing
square metrics retain deterministic majority ties (smallest class code),
Shannon entropy in bits, and complexity (unlike shared edge length among
included adjacent cells divided by geometric square area). Complexity is not
pylandstats edge density.

`*_class_metrics.csv` has one row per input and class (for example 3 inputs
× 6 classes = 18 rows) with whole-raster `area_m2`, `proportion_pct`,
`patch_count`, and `patch_density` (patches/ha). Patch adjacency is
four-neighbor; denominators are the common-footprint native-cell area, and
classes absent from an input get zero values. `*_squares_metrics.csv` holds the
per-square pylandstats metrics (class proportion and patch density, landscape
edge density) with the outer landscape boundary and NoData borders excluded
from edge density (`count_boundary=False`). Native-resolution patch and edge
differences can reflect resolution as well as classification differences.

### Compare reflectance

```bash
plaknit compare reflectance \
  --input planet=planet_class.tif --raster planet:red=planet_red.tif \
  --raster planet:ndvi=planet_ndvi.tif \
  --input sentinel=sentinel_class.tif --raster sentinel:red=sentinel_red.tif \
  --raster sentinel:ndvi=sentinel_ndvi.tif \
  --out-dir ./reflectance_compare/ --seed 42
```

This is a separate command from the spatial and temporal modes, which compare
classifications only. Each `--raster LABEL:NAME=PATH` is a single-band
continuous raster (reflectance band, NDVI, other index) tied to an input label;
a `NAME` must be given for at least two inputs. Rasters must exactly match
their classification grid (CRS, transform, width, height); if a classification
is reprojected for comparison, values are bilinear-resampled with a
nearest-neighbor validity mask. Options: `--out-dir`, `--prefix` (default
`reflectance`), `--seed` (default 42), `--classes`, and `--no-reproject`.

Scaling to 0-1 is automatic. The raster's GDAL scale/offset metadata
(`value * scale + offset`) is used when it is declared and maps all valid
values into 0-1. Otherwise (no metadata, or metadata that does not give 0-1) the
raster's valid pixels are min/max scaled and a warning is issued. The method and
parameters are recorded per raster in the metadata. Min/max scaling makes values
relative to each raster's own range, so outliers compress the range and
indices such as NDVI are distorted; prefer rasters with scale/offset metadata.
A constant raster cannot be min/max scaled and raises an error.

For each input, class, and raster name, a seeded random 95% sample of eligible
pixels is drawn; eligible pixels are in the common classification footprint
and valid in every input supplying that name. The summary reports eligible and
sampled counts with the median and lower/upper quartiles. For each input pair,
shared raster name, and class, an unpaired two-sided Mann-Whitney U test
compares the samples, reported with a rank-biserial effect size
(`2U/(n_a n_b) - 1`, positive when A tends to exceed B), a median difference
(B minus A), and Benjamini-Hochberg-adjusted p-values. Groups with fewer than
5 sampled pixels are marked `insufficient_pixels`. Adjacent pixels are spatially
autocorrelated, so the p-values are overstated; rely on the effect size.
Class masks come from each input's own classification, so the same class code
need not cover the same locations.

Outputs: `*_class_summary.csv`, `*_mann_whitney.csv`, one boxplot PNG per raster
and class, and `*_metadata.json` (seed, scaling, definitions, caveats).

### Agreement and output files

- `*_squares.csv` and `*_squares.gpkg`: square IDs, geometries, centers, and
  existing per-input native-resolution majority, entropy, complexity, and
  cell counts.
- `*_class_metrics.csv`: per-class whole-raster metrics, one row per input and
  class. `*_squares_metrics.csv`: square-level pylandstats outputs.
- `*_agreement.tif` and `*_agreement_legend.csv`: categorical agreement on
  the finest input's native grid and a legend mapping numeric pattern IDs to
  complete canonical grouping sequences. IDs are labels, not sequences.
- `*_agreement_overview.csv`: one row with valid area, full-agreement and
  disagreement area/fraction, distinct pattern count, and disagreement patch
  count. `*_agreement_squares.csv` and `.gpkg` report each
  square's full-agreement fraction, disagreement fraction, and leading
  patterns.
- `*_metadata.json`: mode, ordered inputs, dates, resolutions,
  common valid areas, exclusions, seeds, thresholds, and metric definitions.

This is **inter-input agreement**, not classification accuracy unless
independent ground truth is supplied. Fine-grid cells inheriting a coarser
input's class are not independent observations. Rasters are read into memory.
The former `plaknit landcover` command was removed; migrate to one of the two
`plaknit compare` modes.

## Planning Monthly Planet Composites

`plaknit plan` queries Planet's Data/STAC API, filters PSScene candidates, tiles the AOI, and picks the smallest monthly set that meets both coverage and clear-observation targets. Planning now only writes/prints a plan; ordering is handled separately by `plaknit order`.

```bash
plaknit plan \
  --aoi /path/to/aoi.gpkg \
  --start 2019-08-01 \
  --end 2019-08-31 \
  --cloud-max 0.05 \
  --sun-elev-min 35 \
  --coverage-target 0.99 \
  --min-clear-fraction 0.9 \
  --min-clear-obs 4 \
  --tile-size-m 1000 \
  --instrument-type PS2.SD \
  --out aug_2019_plan.json
```

The summary table shows candidate/selected scenes, achieved coverage, and clear
observation depth.

Planet limits STAC/Data AOI intersections to 1,500 vertices, so the planner
automatically simplifies uploaded AOIs (while preserving topology) until they
fit under that threshold and logs how many vertices were removed. Scenes that
lack clear/cloud metadata are ignored during scoring so plans only rely on
scenes with reliable quality fields.

If you already saved a plan JSON/GeoJSON, submit matching orders later without
recomputing coverage:

```bash
plaknit order \
  --plan aug_2019_plan.json \
  --aoi /path/to/aoi.gpkg \
  --sr-bands 4 \
  --harmonize-to sentinel2 \
  --order-prefix aug2019
```

Order output arguments:
- `--plan`: Plan JSON/GeoJSON that defines which scene IDs (and months) are ordered.
- `--aoi`: Geometry used for clipping; the clip AOI is applied to delivered scenes.
- `--sr-bands`: Chooses 4- or 8-band SR bundle; changes the bands in each scene.
- `--harmonize-to`: `sentinel2` harmonizes to Sentinel-2; `none` keeps native SR.
- `--order-prefix`: Prefix for order name and archive filename; batches append `_2`, `_3`, etc., and ZIPs end with `.zip`.
- `--archive-type`: Delivery archive format; Planet currently supports `zip` only.
- `--single-archive` / `--no-single-archive`: One ZIP per order vs per-scene files.
- `-v` / `-vv`: Verbose logging for submissions and retries; no change to output.

The order subcommand loads the stored plan, clips to the provided AOI, and
submits orders (chunked at 100 scenes max) while reporting the returned order
IDs. If Planet reports any “no access to assets …” errors, `plaknit order`
automatically drops the inaccessible scene IDs and retries so the remaining
items can still be delivered.


## Mosaic

`plaknit mosaic` ships with a CLI that is best run in High-Performance Computing Environments. Install the package into the same
environment that contains GDAL and Orfeo Toolbox, then run the mosaic workflow:

```bash
plaknit mosaic \
  --inputs /path/to/strips \
  --udms /path/to/udms \
  --output /path/to/output/aug_2019.tif \
  --tmpdir /path/to/tmp \
  --jobs 8 \
  --ram 191072
```

You can call the module directly with `python -m plaknit.mosaic`
if you prefer to pin the interpreter. The progress display stays minimal with
four bars: Radiometry (optional) → Mask tiles → Binary mask → Mosaic
(shown when `rich` is installed).

### Required arguments

- `--inputs / -il`: One or more GeoTIFFs or directories. Directories are
  expanded to all `.tif` files.
- `--output / -out`: Destination path for the final mosaic.

### Optional arguments

- `--udms / -udm`: UDM rasters (files or directories). Omit only when
  `--skip-masking` is supplied.
- `--skip-masking`: Use the provided inputs directly without applying the
  gdal-based UDM mask.
- `--harmonize-radiometry`: Enable graph-based radiometric harmonization
  (Harmoni-Planet style) before masking/projection.
- `--metadata-jsons / -meta`: Metadata JSON files/directories used to build the
  overlap/time graph; required only when `--harmonize-radiometry` is enabled.
- `--sr-bands`: Surface reflectance bundle size (4 or 8).
- `--target-crs`: Optional CRS override (for example `EPSG:32735`) applied to
  every tile before mosaicking.
- `--workdir / --tmpdir`: Override the locations used for intermediate strips
  and OTB scratch files. Defaults are automatically managed temp directories.
- `--jobs`: Number of parallel masking workers (defaults to 4).
- `--ram`: RAM hint for OTB in MB (defaults to 131072).
- `-v/ -vv`: Increase logging verbosity.

By default, `plaknit mosaic` now checks every input CRS and reprojects
non-matching rasters to a single target CRS before running OTB. The target is
chosen from the majority CRS across inputs; ties are resolved by choosing the
CRS from the tile nearest the geographic center of the stack.
## Raster/vector distance

Calculate a distance surface from any supported vector source on a projected
raster grid:

```python
from plaknit import distance_to_vector

distance_to_vector("template.tif", "features.gpkg", "distance.tif")
```

The output uses the template grid and stores distances in its CRS map units.
Install `plaknit[gpu]` and pass `backend="gpu"` to use CuPy.


## Random Forest classification

`plaknit rf` trains and applies a Random Forest to multi-band stacks. The CLI
accepts one or more `--image` paths; you can pass multiple aligned GeoTIFFs
directly (or repeat `--image`) or build a VRT first (`gdalbuildvrt stack.vrt band1.tif band2.tif ...`).
Use `--binary-out` during prediction to output individual binary masks for each class.

```bash
# Train (writes a .joblib model)
plaknit rf train \
  --image /data/stack.vrt /data/mosaic.tif \
  --labels /data/training_points.gpkg \
  --label-column class \
  --model-out /data/rf_model.joblib

# Classify (writes a classified GeoTIFF of class IDs)
plaknit rf classify \
  --image /data/stack.vrt /data/mosaic.tif \
  --model /data/rf_model.joblib \
  --output /data/output/classification.tif \
  --block-shape 512 512 \
  --jobs 8 \
  --smooth mrf \
  --beta 1.0 \
  --neighborhood 4 \
  --icm-iters 3 \
  --block-overlap 0
```

Python API (same engine, useful for notebooks):

```python
from plaknit import train_rf, predict_rf

rf = train_rf(
    image_path="planet_stack.tif",
    shapefile_path="training_data.geojson",
    label_column="class_id",
    model_out="planet_rf.joblib",
    n_estimators=600,
    test_fraction=0.3,
    n_jobs=32,
    training_buffer_meters=250.0,
)

predict_rf(
    image_path="planet_stack_2024.tif",
    model_path="planet_rf.joblib",
    output_path="planet_stack_2024_classified.tif",
)
```

Training samples pixels under each polygon window-by-window to keep RAM in
check. `training_buffer_meters` includes surrounding pixels in the training
sample while preserving the source geometry label. Prediction streams over
raster blocks (`--block-shape` overrides block size) so it works on laptops and
HPC nodes alike; add `--jobs` to parallelize block prediction with multiple
worker processes (watch for CPU oversubscription if the model was trained with
`n_jobs` > 1). Classified rasters store numeric
class IDs; inspect `rf.label_decoder` to map each ID back to its label. Training
holds out a fraction of samples for evaluation; prediction logs the holdout
confusion matrix plus band importance from the stored model and writes
`*_metrics.txt` next to the output raster (raw + smoothed confusion matrices and
misclassified validation IDs when available). See `docs/hpcenv.md`
for a Singularity/Apptainer job template that binds the stack,
labels, model, and venv for training + prediction.

## Boosted Regression Tree (BRT) Ensemble classification

`plaknit brt` trains and applies Boosted Regression Tree (BRT) models to multi-band stacks. Training labels use Bernoulli coding: presence is `1` and absence is `0`. Ensemble prediction writes unweighted mean probability and confidence-interval rasters.

### Training

```bash
# Basic ensemble training (5 models, 200 boosting rounds each)
plaknit brt train \
  --image /data/stack.vrt /data/mosaic.tif \
  --labels /data/training_points.gpkg \
  --label-column class \
  --ensemble-dir /data/brt_ensemble/

# Advanced training with hyperparameter tuning
plaknit brt train \
  --image /data/stack.vrt /data/mosaic.tif \
  --labels /data/training_points.gpkg \
  --label-column class \
  --ensemble-dir /data/brt_ensemble/ \
  --n-models 10 \
  --n-estimators 300 \
  --max-depth 8 \
  --learning-rate 0.05 \
  --subsample 0.8 \
  --colsample-bytree 0.9 \
  --test-fraction 0.3 \
  --grid-size 500 \
  --random-state 42 \
  --gpu
```

### Prediction

```bash
# Prediction writes mean, lower, and upper probability rasters
plaknit brt predict \
  --image /data/stack.vrt /data/mosaic.tif \
  --ensemble-dir /data/brt_ensemble/ \
  --output-dir /data/output/ \
  --jobs 8
```

### Python API

```python
from plaknit import BRTEnsemble

# Training phase
ensemble = BRTEnsemble(
    n_models=5,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    gpu=True,
)
ensemble.fit(
    image_path="planet_stack.tif",
    shapefile_path="training_data.geojson",
    label_column="class_id",
    ensemble_dir="./brt_ensemble/",
    grid_size=None,
)

# Prediction phase
ensemble.predict(
    image_path="planet_stack_2024.tif",
    ensemble_dir="./brt_ensemble/",
  output_dir="./probability_outputs/",
  feature_importance_out="predictor_importance.csv",
    jobs=8,
)
```

### Key Features

- **Ensemble Diversity**: Each model in the ensemble is trained with a different random seed, ensuring diverse predictions.
- **Unweighted Summaries**: Class probabilities are averaged equally across all models, with t-based lower and upper confidence bounds.
- **Model Assessment**: Holdout ROC AUC is recorded for each member in ensemble metadata and optional model-summary CSV output.
- **Predictor Ranking**: Optional feature-importance CSVs summarize each input band's unweighted split importance across the ensemble.
- **GPU Support**: Add `--gpu` flag to use XGBoost's GPU acceleration (requires `plaknit[gpu]` installation); gracefully falls back to CPU if unavailable.
- **Probability Output**: `mean_probabilities.tif`, `lower_probabilities.tif`, and `upper_probabilities.tif` are written to the output directory. One-member ensembles write only the mean raster.
- **Spatial Sampling**: Use `--grid-size` to spatially thin training samples for improved generalization.

### Hyperparameter Guide

- `--n-models`: Number of models in the ensemble (5-15 recommended). More models increase ensemble diversity.
- `--n-estimators`: Boosting rounds per model (100-500). More rounds capture complex patterns but risk overfitting.
- `--max-depth`: Tree depth (4-8). Shallower trees prevent overfitting; deeper trees capture interactions.
- `--learning-rate`: Step size for boosting updates (0.01-0.3). Lower rates are more conservative.
- `--subsample`: Fraction of samples per round (0.5-1.0). Lower values add stochasticity and reduce overfitting.
- `--colsample-bytree`: Fraction of features per tree (0.5-1.0). Lower values increase diversity.
- `--test-fraction`: Holdout fraction for evaluation (0.2-0.4). Per-model accuracy is recorded as a diagnostic.

Use `--feature-importance-out predictor_importance.csv` during ensemble prediction to write unweighted per-band summary statistics. `mean_importance` is a relative split-contribution ranking, not a linear coefficient or a causal effect.

### Comparing BRT Ensemble vs Random Forest

| Feature | BRT Ensemble | Random Forest |
|---------|--------------|---------------|
| Model Type | Gradient boosting | Bagging (ensemble) |
| Training Time | Medium | Fast |
| Prediction Accuracy | High | High |
| Interpretability | Low | Medium |
| GPU Support | Yes | No |
| Hyperparameter Tuning | More involved | Less involved |
| Ensemble within Ensemble | Yes (multiple BRTs) | Yes (multiple trees) |

Both classifiers are suitable for PlanetScope data; choose BRT for potentially higher accuracy on complex scenes, or Random Forest for faster training with good accuracy.
