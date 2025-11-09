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
