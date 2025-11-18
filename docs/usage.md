# Usage

`plaknit` ships with a CLI that mirrors the standalone `mosaic_planet.py`
script you may have used previously. Install the package into the same
environment that contains GDAL and Orfeo Toolbox, then run:

```bash
plaknit --inputs /data/strips/*.tif \
        --udms /data/strips/*.udm.tif \
        --output /data/mosaics/planet_mosaic.tif \
        --jobs 8 \
        --ram 196608
```

You can also call the module directly with `python -m plaknit.mosaic` if you
prefer to pin the interpreter.

## Planning & Ordering Monthly Planet Composites

`plaknit plan` runs on laptops or login nodes (no GDAL/OTB requirements) to
query Planet's Data/STAC API, filter PSScene candidates, tile the AOI, and pick
the smallest monthly set that meets both coverage and clear-observation
targets. You can optionally submit one Planet order per month with clipped
surface reflectance scenes (4- or 8-band SR + UDM2) and Sentinel-2
harmonization.

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
observation depth, and any resulting order IDs. Orders deliver one ZIP per
scene (no pre-mosaicking on Planet's side) so you can hand the outputs to
`plaknit mosaic` or future composite builders on HPC.

If you already saved a plan JSON/GeoJSON, submit the matching orders later
without recomputing coverage:

```bash
plaknit order \
  --plan my_monthly_plan.json \
  --aoi aoi_bounds.gpkg \
  --sr-bands 4 \
  --harmonize-to none \
  --order-prefix plk_demo \
  --archive-type zip
```

The order subcommand loads the stored plan, clips to the provided AOI, and issues
one order per month while reporting the returned order IDs.

## Required arguments

- `--inputs / -il`: One or more GeoTIFFs or directories. Directories are
  expanded to all `.tif` files.
- `--output / -out`: Destination path for the final mosaic.

## Optional arguments

- `--udms / -udm`: UDM rasters (files or directories). Omit only when
  `--skip-masking` is supplied.
- `--skip-masking`: Use the provided inputs directly without applying the
  gdal-based UDM mask.
- `--workdir / --tmpdir`: Override the locations used for intermediate strips
  and OTB scratch files. Defaults are automatically managed temp directories.
- `--jobs`: Number of parallel masking workers (defaults to 4).
- `--ram`: RAM hint for OTB in MB (defaults to 131072).
- `-v/ -vv`: Increase logging verbosity.

The CLI guarantees the same behavior as the original script, but it now lives
inside the package so you can version and redistribute the workflow alongside
the rest of your tooling.

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

`plaknit.classify` adds scalable training + inference utilities that lean on
`geopandas`, `rasterio`, and scikit-learn. A minimal workflow:

```python
from plaknit import train_rf, predict_rf

# Train using polygons that carry a "class_id" field
rf = train_rf(
    image_path="planet_stack.tif",
    shapefile_path="training_data.geojson",
    label_column="class_id",
    model_out="planet_rf.joblib",
    n_estimators=600,
    n_jobs=32,
)

# Apply the model to another stack (writes a GeoTIFF of class IDs)
predict_rf(
    image_path="planet_stack_2024.tif",
    model_path="planet_rf.joblib",
    output_path="planet_stack_2024_classified.tif",
)
```

Training data are sampled window-by-window beneath each polygon, keeping RAM
usage bounded. Prediction streams over raster blocks (or user-defined tile
sizes) so the same code works on laptops and HPC nodes alike. Classified
rasters store numeric class IDs; inspect `rf.label_decoder` to map each ID back
to its original label.
