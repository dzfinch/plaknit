# plaknit API reference

This reference is intentionally focused on the two core modules you are most
likely to automate against: `plaknit.mosaic` and `plaknit.classify`. Each
section annotates the public functions/classes so you can wire them into HPC
batch jobs, notebooks, or downstream services.

## Mosaic workflow (`plaknit.mosaic`)

The mosaic module hosts the orchestration objects behind the CLI, so you can
script the same behavior without shell wrappers.

::: plaknit.mosaic.MosaicJob

::: plaknit.mosaic.MosaicWorkflow

::: plaknit.mosaic.run_mosaic

## Random Forest classification (`plaknit.classify`)

Train/predict utilities that couple rasterio, geopandas, and scikit-learn.
Use these functions to build reusable models for PlanetScope stacks.

::: plaknit.classify.train_rf

::: plaknit.classify.predict_rf