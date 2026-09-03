# BRT Ensemble CLI Integration Guide

## Quick Start

### 1. Training an Ensemble

```bash
# Basic training with defaults (5 models, 200 boosting rounds each)
plaknit brt train \
  --image stack.tif \
  --labels training.gpkg \
  --label-column class_id \
  --ensemble-dir ./brt_ensemble/

# Advanced training with custom hyperparameters
plaknit brt train \
  --image band1.tif band2.tif band3.tif \
  --band-indices 1 2 3 \
  --labels training.shp \
  --label-column class \
  --ensemble-dir ./my_ensemble/ \
  --n-models 10 \
  --n-estimators 300 \
  --max-depth 8 \
  --learning-rate 0.05 \
  --test-fraction 0.2 \
  --grid-size 100 \
  --random-state 123 \
  --gpu
```

### 2. Making Predictions

```bash
# Prediction writes mean, lower, and upper probability rasters
plaknit brt predict \
  --image stack.tif \
  --ensemble-dir ./brt_ensemble/ \
  --output-dir ./probability_outputs/ \
  --feature-importance-out predictor_importance.csv \
  --jobs -1
```

## Command Reference

### Training Arguments

**Required:**
- `--image PATH [PATH ...]` - Input raster(s)
- `--labels PATH` - Training labels (Shapefile/GeoPackage)
- `--label-column COLUMN` - Column with class labels
- `--output PATH` - Model file when `--n-models=1`; ensemble directory when `--n-models>1`

**Optional Ensemble Size:**
- `--n-models N` - Number of models; 1 trains a single model (default: 1)
- `--random-state SEED` - Base random seed (default: 42)

**Optional Model Hyperparameters:**
- `--n-estimators N` - Boosting rounds per model (default: 200)
- `--max-depth D` - Tree depth (default: 6)
- `--learning-rate LR` - Boosting learning rate (default: 0.1)
- `--subsample F` - Fraction of samples per tree (default: 1.0)
- `--colsample-bytree F` - Fraction of features per tree (default: 1.0)
- `--jobs N` - Concurrent CPU ensemble model fits (default: 1; values <= 0 use available CPU count; GPU training remains serial)

**Optional Sampling:**
- `--band-indices I [I ...]` - Specific bands to use
- `--test-fraction F` - Holdout fraction (default: 0.3)
- `--grid-size PX` - Grid-based spatial sampling
- `--gpu` - Enable GPU acceleration

### Prediction Arguments

**Required:**
- `--image PATH [PATH ...]` - Input raster(s)
- `--ensemble-dir DIR` - Directory with trained models
- `--output-dir DIR` - Directory for `mean_probabilities.tif` and, for ensembles with at least two models, `lower_probabilities.tif` and `upper_probabilities.tif`
- `--model-summary-out PATH` - Optional CSV with member seed, holdout ROC AUC, and hyperparameters
- `--feature-importance-out PATH` - Optional CSV with unweighted per-band ensemble importance statistics

**Optional Processing:**
- `--band-indices I [I ...]` - Specific bands
- `--block-shape H W` - Memory-efficient block reading
- `--jobs N` - Parallel workers (1=serial, -1=all cores)

- `--ci-level F` - Confidence level for lower and upper probability bounds (default: 0.95)
- `--block-overlap PX` - Overlap used while processing blocks (default: 0)

## Workflow Examples

### Example 1: Simple Classification Pipeline

```bash
# 1. Train ensemble with default settings
plaknit brt train \
  --image landsat_stack.tif \
  --labels labeled_sites.gpkg \
  --label-column land_use \
  --ensemble-dir ./land_use_ensemble/

# 2. Predict on new imagery
plaknit brt predict \
  --image new_scene_stack.tif \
  --ensemble-dir ./land_use_ensemble/ \
  --output-dir ./land_use_probabilities/
```

### Example 2: Production Pipeline with Validation

```bash
# 1. Train ensemble with spatial sampling for robust generalization
plaknit brt train \
  --image scene1.tif scene2.tif scene3.tif \
  --labels validation_polygons.shp \
  --label-column habitat_type \
  --ensemble-dir ./habitat_model/ \
  --n-models 7 \
  --test-fraction 0.25 \
  --grid-size 500 \
  --random-state 42 \
  --gpu

# 2. Predict with probability summaries for uncertainty quantification
plaknit brt predict \
  --image new_scene.tif \
  --ensemble-dir ./habitat_model/ \
  --output-dir ./habitat_probabilities/ \
  --jobs -1
```

### Example 3: Fine-Tuned Ensemble for Difficult Classes

```bash
# Train with more models and stronger regularization
plaknit brt train \
  --image training_stack.tif \
  --labels field_validation.gpkg \
  --label-column category \
  --ensemble-dir ./difficult_classes/ \
  --n-models 15 \
  --n-estimators 400 \
  --max-depth 5 \
  --learning-rate 0.05 \
  --subsample 0.8 \
  --colsample-bytree 0.9 \
  --test-fraction 0.3 \
  --grid-size 250 \
  --random-state 999

# Predict probability summaries
plaknit brt predict \
  --image target_scene.tif \
  --ensemble-dir ./difficult_classes/ \
  --output-dir ./difficult_class_probabilities/ \
  --jobs -1
```

## Output Files

### After Training

```
brt_ensemble/
├── brt_0.joblib              # First model
├── brt_1.joblib              # Second model
├── brt_2.joblib              # ... (N models total)
└── ensemble_metadata.json    # Training config + test accuracies
```

**Metadata structure:**
```json
{
  "n_models": 5,
  "n_estimators": 200,
  "training_buffer_meters": 250.0,
  "max_depth": 6,
  "learning_rate": 0.1,
  "subsample": 1.0,
  "colsample_bytree": 1.0,
  "test_fraction": 0.3,
  "band_indices": [1, 2, 3],
  "grid_size": null,
  "test_auc": [0.92, 0.89, 0.91, 0.88, 0.93]
}
```

### After Prediction

```
output/
├── mean_probabilities.tif             # N bands = N classes
├── lower_probabilities.tif            # Present for 2+ models
└── upper_probabilities.tif            # Present for 2+ models
```

`--feature-importance-out` writes one row per predictor band with
`mean_importance`, `std_importance`, `min_importance`, `max_importance`, and
`models_using_feature`. Importance is an unweighted split-based ranking, not a
linear coefficient or a causal effect.

## Tips & Troubleshooting

### GPU Usage
- Add `--gpu` flag to enable XGBoost GPU acceleration
- Requires `xgboost[gpu]` and CUDA-capable GPU
- Falls back to CPU gracefully if GPU unavailable

### Memory Management
- Use `--block-shape H W` to reduce memory per block
- Increase `--jobs` gradually to find optimal parallelism
- For very large rasters, set conservative block sizes (e.g., 512x512)

### Model Accuracy
- Increase `--n-models` for better ensemble diversity (5-15 recommended)
- Increase `--n-estimators` for more boosting rounds per model (100-500 typical)
- Reduce `--max-depth` to prevent overfitting (4-6 typical)
- Lower `--learning-rate` for finer gradual improvements (0.05-0.1)
- Use `--grid-size` for spatially diverse training samples

### Reproducibility
- Always set `--random-state` explicitly
- Ensemble with seeds `[random_state, random_state+1, ...]` ensures diversity

## Integration with Other plaknit Commands

```bash
# Classification-only workflow (RF or BRT)
plaknit classify train --image stack.tif --labels train.gpkg ...
plaknit brt train --image stack.tif --labels train.gpkg ...

# Planning (Planet data acquisition)
plaknit plan --start 2024-01-01 --end 2024-12-31 ...
plaknit mosaic --input images/ --output mosaic.tif
plaknit brt train --image mosaic.tif --labels labels.gpkg ...

# Ordering (Planet data)
plaknit order --plan plan_id ...
```

## Performance Notes

- **Training**: O(N * T) where N = models, T = single model training time
- **Prediction**: O(N * P) where P = single model prediction time
- **Memory**: ~100-300MB per model depending on raster size
- **Typical ensemble (5 models)**: 5-10 minute training, 2-5 minute prediction on 10k×10k raster

## See Also

- `plaknit classify` - Random Forest classification (same CLI structure)
- `plaknit classify smooth` - Post-process probabilities with MRF/Bayes
- See package documentation: `plaknit --help`
