"""Compatibility shim for the migrated RF implementation.

This preserves the legacy import path used by the existing tests and downstream code.
"""

from .models.rf import _open_raster_stack, predict_rf, smooth_probs, train_rf

__all__ = ["_open_raster_stack", "train_rf", "predict_rf", "smooth_probs"]
