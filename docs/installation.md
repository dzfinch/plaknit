# Installation

## Stable release

To install plaknit, run this command in your terminal:

```
pip install plaknit
```

This is the preferred method to install plaknit, as it will always install the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process.

## Optional Dependencies

### GPU Acceleration

To enable GPU acceleration for BRT/Random Forest model training and prediction, install the GPU extra:

```
pip install plaknit[gpu]
```

This installs `xgboost[gpu]` and `cupy` for CUDA-accelerated computation. Requires a CUDA-capable GPU and CUDA Toolkit installation. The package gracefully falls back to CPU if GPU is unavailable.

### From sources

To install plaknit from sources, run this command in your terminal:

```
pip install git+https://github.com/dzfinch/plaknit
```
