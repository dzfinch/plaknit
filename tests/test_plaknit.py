#!/usr/bin/env python

"""Tests for the `plaknit` package."""

import unittest

import plaknit


class TestPlaknit(unittest.TestCase):
    """Tests for the installed package surface."""

    def test_package_exports(self):
        """The package root should expose the public API."""
        self.assertTrue(hasattr(plaknit, "__version__"))
        self.assertTrue(hasattr(plaknit, "train_rf"))
        self.assertTrue(hasattr(plaknit, "predict_rf"))
        self.assertTrue(hasattr(plaknit, "train_brt"))
        self.assertTrue(hasattr(plaknit, "predict_brt"))
