"""Top-level package for plaknit."""

from .analysis import (
    normalized_difference,
    normalized_difference_from_files,
    normalized_difference_from_raster,
)
from .classify import predict_rf, train_rf
from .orders import submit_orders_for_plan
from .planner import plan_monthly_composites, write_plan

__author__ = """Dryver Finch"""
__email__ = "dryver2206@gmail.com"
__version__ = "0.1.0"

__all__ = [
    "normalized_difference",
    "normalized_difference_from_raster",
    "normalized_difference_from_files",
    "train_rf",
    "predict_rf",
    "plan_monthly_composites",
    "write_plan",
    "submit_orders_for_plan",
]
