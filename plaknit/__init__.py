"""Top-level package for plaknit."""

from .analysis import (
    normalized_difference,
    normalized_difference_from_files,
    normalized_difference_from_raster,
)
from .classify import predict_rf, train_rf

__author__ = """Dryver Finch"""
__email__ = "dryver2206@gmail.com"
__version__ = "0.0.2"

__all__ = [
    "normalized_difference",
    "normalized_difference_from_raster",
    "normalized_difference_from_files",
    "train_rf",
    "predict_rf",
]
