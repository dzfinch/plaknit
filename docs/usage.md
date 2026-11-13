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

Spatial mode does not take dates. Input labels identify sources in tables and
metadata, so choose labels that name the sensor or source.

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
`--group-by date|year` records the temporal grouping in metadata (default:
`date`); the wide square transition table identifies inputs by their order.

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
cell whose interior intersects it is valid. Square outputs retain the square
ID. Existing square statistics use each classification's native cells whose
centroids lie in that square; west and
north centroid boundaries are included, east and south are excluded. Existing
square metrics retain deterministic majority ties (smallest class code), and
use PyLandStats `shannon_diversity_index` (natural logarithm) and
`joint_entropy` (base 2, horizontal and vertical adjacencies). Both are
computed on the selected native cells. PyLandStats returns an undefined result
for single-class landscapes; both metrics are reported as zero in that case.
The installed PyLandStats version determines the exact metric implementation
and is recorded in metadata. In PyLandStats 3.1.0, Shannon diversity is
calculated using adjacency counts, so it can differ from Shannon entropy
calculated directly from class proportions.
These columns replace the former base-2 `entropy` and unlike-edge-length
`complexity` columns, whose values are not interchangeable with the new metrics.

`*_class_metrics.csv` has one row per input and class (for example 3 inputs
× 6 classes = 18 rows) with whole-raster `area_m2`, `proportion_pct`,
`patch_count`, and `patch_density` (patches/ha). Patch adjacency is
four-neighbor; denominators are the common-footprint native-cell area, and
classes absent from an input get zero values. Native-resolution patch differences
can reflect resolution as well as classification differences.

`*_squares_metrics.csv` describes class transitions, with one row per square
and class in the union of input class codes. The reference is the input with
the largest analysis-grid cell area, breaking ties in input order. For each
row, `reference_input_label` identifies that input and
`reference_class_fraction` is the fraction of valid finest-grid cells in
the square mapped to that reference class; `reference_class_cell_count` is
the corresponding count. Each `input_N_LABEL_class_C_fraction` column is the
fraction of those reference-class cells mapped to class C in input N, where N
is the one-based input order and LABEL is its exact input label. The input
number keeps columns distinct even for similar labels; CSV quoting preserves
labels containing punctuation. This replaces the previous unlabeled
`input_N_class_C_fraction` headers. All non-reference inputs are included, even
when their resolutions are equal to the reference. Values for a comparison
input sum to one per present reference class. When the reference class is
absent, its count and square fraction are zero and its conditional fractions
are blank (undefined), not zero. Mapping and validity match the agreement
raster's largest-overlap rule; square inclusion uses finest-grid cell
centroids. Metadata maps the columns to input labels, resolutions, and class
codes. If a square has no valid agreement-grid cells, its agreement fractions
and reference-class square fractions are also blank. These are mapped cell
proportions, not exact intersection areas.

### Compare reflectance

```bash
plaknit compare reflectance \
  --input planet=planet_class.tif --raster planet:red=planet_red.tif \
  --raster planet:ndvi=planet_ndvi.tif \
  --input sentinel=sentinel_class.tif --raster sentinel:red=sentinel_red.tif \
  --raster sentinel:ndvi=sentinel_ndvi.tif \
  --spatial-units ./compare_squares.gpkg \
  --out-dir ./reflectance_compare/ --seed 42
```

For multiband rasters, list one name per band in raster order:

```powershell
plaknit compare reflectance `
  --input "planet=C:\data\planet_class.tif" `
  --raster "planet:blue;green;red;nir=C:\data\planet_stack.tif" `
  --input "sentinel=C:\data\sentinel_class.tif" `
  --raster "sentinel:blue;green;red;nir=C:\data\sentinel_stack.tif" `
  --spatial-units "C:\data\compare_squares.gpkg" `
  --out-dir "C:\data\reflectance_compare"
```

Quote arguments containing semicolons in PowerShell and other shells. Names
map to bands starting at 1; supply exactly as many names as the raster has
bands. Single-band `LABEL:NAME=PATH` arguments remain supported and can be
mixed with stacks. Each name must be unique within an input and supplied for
at least two inputs. Matching names are compared, even if their band orders
differ between inputs.
For Python callers, `ReflectanceInput.rasters` also accepts `ReflectanceBand`
values specifying `path`, one-based `band`, and the raster's `band_count`;
existing single-band path strings remain supported.

This command requires spatial-unit polygons with a unique ID, supplied by
`--spatial-units`. A GeoPackage/vector file must carry its CRS and an ID field
named `unit_id`, `square_id`, or `id`. A CSV must contain that ID and
`xmin,ymin,xmax,ymax`; because CSV has no CRS, also provide
`--spatial-units-crs` (for example `EPSG:32614`). Existing comparison
`*_squares.gpkg`/`*_squares.csv` outputs can be used directly.

Each raster is read on its input classification's native grid. If the grids
differ, values are bilinear-resampled and validity masks nearest-neighbor
resampled in memory within each spatial unit. Declared raster scale/offset
metadata is applied. Without it, native values are preserved; inputs are not
independently min/max stretched. Confirm that matching raster names have
comparable units before interpreting their differences. Source path, band,
scale method, unit IDs, CRS, alignment, and inference settings are recorded in
metadata.

For each input, raster name, class, and spatial unit, the analysis reports the
median and quartiles of valid pixels selected by that input's own classification
mask. Comparisons pair these unit-level medians by unit ID and class. The
estimate is the median paired difference (input B minus input A); the effect
size is paired rank-biserial correlation. Confidence intervals resample whole
coordinate-based blocks, and p-values use paired block sign-flip permutations.
Exact permutation is used for at most 15 blocks; larger tests use seeded Monte
Carlo permutations with a plus-one correction. BH adjustment covers all
comparisons with inferential p-values in the run; these permutation p-values
cannot underflow to a reported zero.

The empirical block-size estimate uses the pooled semivariogram of square-level
paired differences. `--block-size-m` overrides it. If no stable range can be
estimated and no override is supplied, the analysis reports descriptive
estimates but no confidence intervals or p-values. Rows with fewer than
`--min-independent-blocks` (default 10) are also descriptive only. The block
estimate is limited by the number and spacing of supplied spatial units and is
not proof that blocks are independent. `--bootstrap-replicates` (default 2000),
`--permutation-replicates` (default 9999), `--seed`, `--classes`, and block
settings are recorded for reproducibility.

Results describe the supplied raster products, not a sensor-only causal effect.
Acquisition dates, scene IDs, processing histories, and quality-mask provenance
are not inferred; provide and evaluate those separately. Matching class codes
are assumed to have comparable meanings. A class with no paired spatial units
cannot be tested.

Outputs: `*_spatial_unit_summary.csv` (one row per input/unit/class/raster),
`*_class_summary.csv` (summary across unit medians),
`*_spatial_comparisons.csv` (paired estimates, intervals, effects, p-values,
and status), `*_methods_report.md`, and
`*_metadata.json`.

### Agreement and output files

- `*_squares.csv` and `*_squares.gpkg`: square IDs, geometries, centers, and
  per-input native-resolution majority, Shannon diversity index, joint entropy,
  and cell counts, plus overall full-agreement and disagreement fractions.
- `*_class_metrics.csv`: per-class whole-raster metrics, one row per input and
  class. `*_squares_metrics.csv`: per-square conditional class transitions.
- `*_agreement.tif` and `*_agreement_legend.csv`: categorical agreement on
  the finest input's native grid and a legend mapping numeric pattern IDs to
  complete canonical grouping sequences. The legend's `cell_proportion` is
  the pattern's cell count divided by all valid agreement cells (excluding
  NoData); it is blank when there are no valid cells. IDs are labels, not
  sequences.
- `*_metadata.json`: mode, ordered inputs, dates, resolutions,
  common valid areas, exclusions, seeds, transition columns, and metric definitions.

The former agreement overview and separate agreement-square outputs are no
longer written, and `--patch-min-area-m2` and the corresponding Python
`run_analysis` argument have been removed. Older files in an existing output
directory are not deleted automatically.

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
