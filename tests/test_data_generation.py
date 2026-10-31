"""
Tests for Synthetic Data Generation Module
"""

import unittest
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

# Ensure project root and src are in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from data_generation import SyntheticDataGenerator


class TestSyntheticDataGenerator(unittest.TestCase):
    """Test suite for SyntheticDataGenerator."""

    def setUp(self):
        """Set up small generator instance for fast testing."""
        self.generator = SyntheticDataGenerator(random_seed=42, n_years=2)

    def test_climate_data_structure_and_types(self):
        """Verify climate dataframe columns, record counts, and types."""
        df = self.generator.generate_climate_data()

        expected_columns = {"Date", "District", "Avg_Temp", "Max_Temp", "Min_Temp", "Precipitation_mm"}
        self.assertTrue(expected_columns.issubset(set(df.columns)))
        self.assertFalse(df.empty)

        # 3 districts across (2 years * 365 + leap day) days
        self.assertGreater(len(df), 2000)
        self.assertSetEqual(set(df["District"].unique()), {"Shopian", "Baramulla", "Pulwama"})

        # Physical sanity checks
        self.assertTrue((df["Max_Temp"] >= df["Avg_Temp"]).all())
        self.assertTrue((df["Avg_Temp"] >= df["Min_Temp"]).all())
        self.assertTrue((df["Precipitation_mm"] >= 0).all())

    def test_geospatial_data_ranges(self):
        """Verify NDVI is bounded [0, 1] and altitude matches district boundaries."""
        df = self.generator.generate_geospatial_data()

        self.assertIn("NDVI_score", df.columns)
        self.assertIn("Arable_Land_Altitude", df.columns)

        # NDVI bounds
        self.assertTrue((df["NDVI_score"] >= 0.0).all())
        self.assertTrue((df["NDVI_score"] <= 1.0).all())

        # District altitude boundaries
        shopian = df[df["District"] == "Shopian"]["Arable_Land_Altitude"]
        self.assertTrue((shopian >= 1500).all() and (shopian <= 2500).all())

        baramulla = df[df["District"] == "Baramulla"]["Arable_Land_Altitude"]
        self.assertTrue((baramulla >= 1200).all() and (baramulla <= 2200).all())

        pulwama = df[df["District"] == "Pulwama"]["Arable_Land_Altitude"]
        self.assertTrue((pulwama >= 1400).all() and (pulwama <= 2300).all())

    def test_economic_data_calculations(self):
        """Verify economic data has valid crops, positive revenues, and no NaNs."""
        df = self.generator.generate_economic_data()

        expected_crops = {"Apples", "Walnuts", "Saffron"}
        self.assertSetEqual(set(df["Crop_Type"].unique()), expected_crops)
        self.assertTrue((df["Revenue_per_Hectare"] >= 0).all())
        self.assertTrue((df["Yield_per_Hectare"] > 0).all())
        self.assertTrue((df["Farm_Gate_Price"] > 0).all())
        self.assertEqual(df.isnull().sum().sum(), 0)

    def test_reproducibility(self):
        """Verify random seed generates identical records."""
        gen1 = SyntheticDataGenerator(random_seed=123, n_years=1)
        df1 = gen1.generate_climate_data()

        gen2 = SyntheticDataGenerator(random_seed=123, n_years=1)
        df2 = gen2.generate_climate_data()

        pd.testing.assert_frame_equal(df1, df2)


if __name__ == "__main__":
    unittest.main()

