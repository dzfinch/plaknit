# plaknit API reference

This reference is intentionally focused on the core modules you are most
likely to automate against: `plaknit.acquisition.mosaic`, `plaknit.classify`, and `plaknit.models`. Each
section annotates the public functions/classes so you can wire them into HPC
batch jobs, notebooks, or downstream services.

## Mosaic workflow (`plaknit.acquisition.mosaic`)

The mosaic module hosts the orchestration objects behind the CLI, so you can
script the same behavior without shell wrappers.

::: plaknit.acquisition.mosaic.MosaicJob

::: plaknit.acquisition.mosaic.MosaicWorkflow

::: plaknit.acquisition.mosaic.run_mosaic

## Random Forest classification (`plaknit.classify`)

Train/predict utilities that couple rasterio, geopandas, and scikit-learn.
Use these functions to build reusable models for PlanetScope stacks.

::: plaknit.classify.train_rf

::: plaknit.classify.predict_rf

## Boosted Regression Tree Ensemble (`plaknit.models.ensemble`)

Ensemble of independent BRT models with unweighted probability summaries. This
class provides a scikit-learn-style API for training and applying BRT ensembles
to multi-band raster stacks.

::: plaknit.models.ensemble.BRTEnsemble

### Workflow

1. **Initialize**: Create a `BRTEnsemble` instance with desired hyperparameters
2. **Train**: Call `fit()` with training raster and labeled polygons
3. **Predict**: Call `predict()` to apply the ensemble to new rasters

Each model in the ensemble is trained with a different random seed, and
predictions are summarized with an equal-weight mean and t-based probability
confidence bounds across all models.

Each ensemble member's holdout assessment is reported as binary ROC AUC.
Optional feature-importance output summarizes each predictor band's unweighted
split importance across members; it is a relative ranking, not a coefficient.

`BRTEnsemble` uses `n_models` for the number of ensemble members and
`n_estimators` for the number of XGBoost boosting rounds per
member. Ensemble training also accepts `training_buffer_meters`, which expands
each training geometry before extracting member samples. Ensemble metadata
records the buffer setting.

## Single Boosted Regression Tree model (`plaknit.models.brt`)

If you prefer a single BRT model without ensemble averaging, use these functions
directly.

::: plaknit.models.brt.train_brt

::: plaknit.models.brt.predict_brt

### Bernoulli training and pseudo-absences

BRT training uses binary presence/absence labels. The label column must contain
presence values coded as `1` and may also contain explicit absence values coded
as `0`. By default, `train_brt()` generates pseudo-absence pixels at a `1.0`
ratio, meaning one pseudo-absence per presence sample. Set
`pseudo_absence_ratio=0` to disable generation.

Pseudo-absences are sampled from valid raster pixels outside the training
geometries. `training_buffer_meters` optionally expands those geometries for
both training extraction and candidate selection, preventing samples within the
specified distance of the training data. The buffer applies to the complete
input geometries and uses the raster CRS; a metric transformation is used when
the raster CRS is geographic.
Sampling is deterministic for a given `random_state`. Generated absence
samples are labeled `0` and are included before grid thinning and holdout
splitting.

For example:

```python
from plaknit import train_brt

model = train_brt(
	image_path="planet_stack.tif",
	shapefile_path="presence_sites.gpkg",
	label_column="presence",
	model_out="brt_model.joblib",
	n_estimators=200,
	pseudo_absence_ratio=1.0,
	training_buffer_meters=250.0,
	random_state=42,
)
```

The trained model exposes `pseudo_absence_ratio_`,
`training_buffer_meters_`, and `pseudo_absence_count_` attributes in addition to
the existing training metadata. BRT output classes are the Bernoulli values
`0` and `1`.

## Random Forest training buffers

`train_rf()` accepts the same `training_buffer_meters` parameter. For RF, the
buffer expands each training geometry only for feature extraction; no
pseudo-absence samples are generated. The buffered pixels retain the label of
their source training geometry.

```python
from plaknit import train_rf

model = train_rf(
	image_path="planet_stack.tif",
	shapefile_path="training_data.gpkg",
	label_column="class_id",
	model_out="rf_model.joblib",
	training_buffer_meters=250.0,
)
```

## Raster/vector distance

`distance_to_vector` rasterizes any GeoPandas-supported vector source onto a
projected Rasterio template grid, then writes Euclidean distances in map units.
Use `backend="gpu"` with a CUDA-compatible CuPy installation for GPU execution;
the default `backend="cpu"` uses SciPy.

::: plaknit.geometry.distance_to_vector