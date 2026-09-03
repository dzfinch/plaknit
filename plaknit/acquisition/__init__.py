"""Acquisition subpackage for plaknit.

This package groups geometry, planning, ordering, and mosaic workflow modules under
one import path so callers can use ``from plaknit.acquisition import ...`` without
relying on implicit namespace package behavior.
"""

from . import geometry, mosaic, orders, planner

__all__ = ["geometry", "mosaic", "orders", "planner"]
