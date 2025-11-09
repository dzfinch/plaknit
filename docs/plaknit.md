# plaknit API reference

This reference is intentionally focused on the three core modules you are most
likely to automate against: `plaknit.mosaic`, `plaknit.analysis`, and
`plaknit.classify`. Each section annotates the public functions/classes so you
can wire them into HPC batch jobs, notebooks, or downstream services.

## Mosaic workflow (`plaknit.mosaic`)

The mosaic module hosts the orchestration objects behind the CLI, so you can
script the same behavior without shell wrappers.

::: plaknit.mosaic.MosaicJob

::: plaknit.mosaic.MosaicWorkflow

::: plaknit.mosaic.run_mosaic

## Spectral indices (`plaknit.analysis`)

Helpers for computing normalized-difference style indices directly from
NumPy arrays or raster files (single dataset or two separate stacks).

::: plaknit.analysis.normalized_difference

::: plaknit.analysis.normalized_difference_from_raster

::: plaknit.analysis.normalized_difference_from_files

## Random Forest classification (`plaknit.classify`)

Train/predict utilities that couple rasterio, geopandas, and scikit-learn.
Use these functions to build reusable models for PlanetScope stacks.

::: plaknit.classify.train_rf

::: plaknit.classify.predict_rf
