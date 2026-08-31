"""Compatibility shims for the migrated RF module."""

from __future__ import annotations

from ..models.rf import predict_rf, smooth_probs, train_rf

__all__ = ["train_rf", "predict_rf", "smooth_probs"]
