"""Public classification API for plaknit.

This module is the stable import path for random-forest (RF) and boosted regression tree (BRT)
training and prediction. The ``plaknit.compat`` path remains only as a deprecated compatibility alias.
"""

from .models.brt import predict_brt, train_brt
from .models.rf import _open_raster_stack, predict_rf, smooth_probs, train_rf

__all__ = [
    "_open_raster_stack",
    "predict_brt",
    "predict_rf",
    "smooth_probs",
    "train_brt",
    "train_rf",
]
