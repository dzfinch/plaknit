# Usage

## Planning & Ordering Monthly Planet Composites

`plaknit plan` can run on local devices to query Planet's Data/STAC API, filter PSScene candidates, tile the AOI, and pick the smallest monthly set that meets both coverage and clear-observation targets. You can optionally submit Planet orders with clipped surface reflectance scenes (4- or 8-band SR + UDM2) and Sentinel-2 harmonization, chunked into batches of up to 100 scenes with predictable order/ZIP names.

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
  --sr-bands 4 \
  --instrument-type PS2.SD \
  --harmonize-to sentinel2 \
  --out aug_2019_plan.json
```

The summary table shows candidate/selected scenes, achieved coverage, clear
observation depth, and any resulting order IDs. Orders deliver single-archive
ZIPs per order (no pre-mosaicking on Planet's side) so you can hand the outputs
to `plaknit mosaic` or future composite builders on HPC; orders are chunked into
up to 100 scenes each, with names suffixed `_1`, `_2`, etc. when needed.

Planet limits STAC/Data AOI intersections to 1,500 vertices, so the planner
automatically simplifies uploaded AOIs (while preserving topology) until they
fit under that threshold and logs how many vertices were removed. Scenes that
lack clear/cloud metadata are ignored during scoring so plans only rely on
scenes with reliable quality fields.

If you already saved a plan JSON/GeoJSON, submit the matching orders later
without recomputing coverage:

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
three bars: Mask tiles → Binary mask → Mosaic (shown when `rich` is installed).

### Required arguments

- `--inputs / -il`: One or more GeoTIFFs or directories. Directories are
  expanded to all `.tif` files.
- `--output / -out`: Destination path for the final mosaic.

### Optional arguments

- `--udms / -udm`: UDM rasters (files or directories). Omit only when
  `--skip-masking` is supplied.
- `--skip-masking`: Use the provided inputs directly without applying the
  gdal-based UDM mask.
- `--sr-bands`: Surface reflectance bundle size (4 or 8).
- `--ndvi`: Append an NDVI band (NIR/Red uses bands 4/3 for 4-band, 8/6 for 8-band).
- `--workdir / --tmpdir`: Override the locations used for intermediate strips
  and OTB scratch files. Defaults are automatically managed temp directories.
- `--jobs`: Number of parallel masking workers (defaults to 4).
- `--ram`: RAM hint for OTB in MB (defaults to 131072).
- `-v/ -vv`: Increase logging verbosity.


## Random Forest classification

`plaknit classify` trains and applies a Random Forest to multi-band stacks. The CLI
accepts one or more `--image` paths; you can pass multiple aligned GeoTIFFs
directly (or repeat `--image`) or build a VRT first (`gdalbuildvrt stack.vrt band1.tif band2.tif ...`).
Use `--band-indices` (1-based) to select a subset of stacked bands when you want
the same model to run on 4-band and 8-band inputs. Train + predict from the CLI:

```bash
# Train (writes a .joblib model)
plaknit classify train \
  --image /data/stack.vrt /data/mosaic.tif \
  --labels /data/training_points.gpkg \
  --label-column class \
  --model-out /data/rf_model.joblib

# Predict (writes a classified GeoTIFF of class IDs)
plaknit classify predict \
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
)

predict_rf(
    image_path="planet_stack_2024.tif",
    model_path="planet_rf.joblib",
    output_path="planet_stack_2024_classified.tif",
)
```

Training samples pixels under each polygon window-by-window to keep RAM in
check. Prediction streams over raster blocks (`--block-shape` overrides block
size) so it works on laptops and HPC nodes alike; add `--jobs` to parallelize
block prediction with multiple worker processes (watch for CPU oversubscription
if the model was trained with `n_jobs` > 1). Classified rasters store numeric
class IDs; inspect `rf.label_decoder` to map each ID back to its label. Training
holds out a fraction of samples for evaluation; prediction logs the holdout
confusion matrix plus band importance from the stored model and writes
`*_metrics.txt` next to the output raster (raw + smoothed confusion matrices and
misclassified validation IDs when available). See `docs/hpcenv.md`
for a Singularity/Apptainer job template that binds the stack,
labels, model, and venv for training + prediction.
