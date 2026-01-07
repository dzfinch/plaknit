# Usage

`plaknit` ships with a CLI that is best run in High-Performance Computing Environments. Install the package into the same
environment that contains GDAL and Orfeo Toolbox, then run the stitched workflow:

```bash
plaknit stitch \
  --inputs /data/strips/*.tif \
  --udms /data/strips/*.udm2.tif \
  --output /data/mosaics/planet_mosaic.tif \
  --jobs 8 \
  --ram 196608
```

`plaknit stitch` is also available as `plaknit mosaic` for backward
compatibility. You can call the module directly with `python -m plaknit.mosaic`
if you prefer to pin the interpreter. The progress display stays minimal with
three bars: Mask tiles → Binary mask → Mosaic (shown when `rich` is installed).

## Required arguments

- `--inputs / -il`: One or more GeoTIFFs or directories. Directories are
  expanded to all `.tif` files.
- `--output / -out`: Destination path for the final stitched mosaic.

## Optional arguments

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

## Planning & Ordering Monthly Planet Composites (Beta)

`plaknit plan` can run on local devices to query Planet's Data/STAC API, filter PSScene candidates, tile the AOI, and pick the smallest monthly set that meets both coverage and clear-observation targets. You can optionally submit Planet orders with clipped surface reflectance scenes (4- or 8-band SR + UDM2) and Sentinel-2 harmonization, chunked into batches of up to 100 scenes with predictable order/ZIP names.

```bash
plaknit plan \
  --aoi aoi_bounds.gpkg \
  --start 2024-01-01 \
  --end 2024-06-30 \
  --cloud-max 0.15 \
  --sun-elev-min 35 \
  --coverage-target 0.98 \
  --min-clear-fraction 0.8 \
  --min-clear-obs 3 \
  --tile-size-m 1000 \
  --sr-bands 8 \
  --harmonize-to sentinel2 \
  --out my_monthly_plan.json \
  --order \
  --order-prefix plk_demo
```

The summary table shows candidate/selected scenes, achieved coverage, clear
observation depth, and any resulting order IDs. Orders deliver single-archive
ZIPs per order (no pre-mosaicking on Planet's side) so you can hand the outputs
to `plaknit stitch` or future composite builders on HPC; orders are chunked into
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
  --plan my_monthly_plan.json \
  --aoi aoi_bounds.gpkg \
  --sr-bands 4 \
  --harmonize-to sentinel2 \
  --order-prefix plk_demo \
  --archive-type zip
```

The order subcommand loads the stored plan, clips to the provided AOI, and
submits orders (chunked at 100 scenes max) while reporting the returned order
IDs. If Planet reports any “no access to assets …” errors, `plaknit order`
automatically drops the inaccessible scene IDs and retries so the remaining
items can still be delivered.


## Normalized difference analysis

`plaknit` also exposes helpers that wrap `rasterio` so you can build spectral
indices without leaving Python:

```python
from plaknit import normalized_difference_from_raster

ndvi = normalized_difference_from_raster(
    "planet_strip.tif",
    numerator_band=4,      # NIR
    denominator_band=3,    # Red
    dst_path="planet_strip_ndvi.tif",
)
```

For rasters stored in two separate files, call
`plaknit.normalized_difference_from_files("nir.tif", "red.tif")`. Each helper
returns the calculated array so you can continue working with NumPy while
optionally persisting the results back to disk.

## Random Forest classification

`plaknit classify` trains and applies a Random Forest to multi-band stacks. The CLI
expects a single `--image` path; if your bands live in separate TIFFs, build a VRT
first (`gdalbuildvrt stack.vrt band1.tif band2.tif ...`). Train + predict from the
CLI:

```bash
# Train (writes a .joblib model)
plaknit classify train \
  --image /data/stack.vrt \
  --labels /data/train_labels.gpkg \
  --label-column class \
  --model-out /data/rf_model.joblib \
  --n-estimators 500 \
  --jobs 32

# Predict (writes a classified GeoTIFF of class IDs)
plaknit classify predict \
  --image /data/stack.vrt \
  --model /data/rf_model.joblib \
  --output /data/output/prediction.tif \
  --block-shape 512 512 \
  --jobs 8 \
  --smooth none
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
class IDs; inspect `rf.label_decoder` to map each ID back to its label. See
`docs/hpcenv.md` for a Singularity/Apptainer job template that binds the stack,
labels, model, and venv for training + prediction.
