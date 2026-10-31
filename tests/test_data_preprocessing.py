"""
Tests for Data Preprocessing and Feature Engineering Module
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from data_generation import SyntheticDataGenerator
from data_preprocessing import DataPreprocessor


class TestDataPreprocessing(unittest.TestCase):
    """Test suite for DataPreprocessor."""

    @classmethod
    def setUpClass(cls):
        """Generate test data once for all test methods."""
        generator = SyntheticDataGenerator(random_seed=42, n_years=3)
        cls.climate_df = generator.generate_climate_data()
        cls.geo_df = generator.generate_geospatial_data()
        cls.econ_df = generator.generate_economic_data()

    def setUp(self):
        self.preprocessor = DataPreprocessor(
            self.climate_df, self.geo_df, self.econ_df
        )

    def test_merge_and_aggregate_shape_and_columns(self):
        """Verify merged features contain expected engineered anomaly columns."""
        merged = self.preprocessor.merge_and_aggregate()

        expected_features = [
            "Year", "District", "Crop_Type", "Yield_per_Hectare", "Farm_Gate_Price",
            "Revenue_per_Hectare", "Yearly_Avg_Temp", "Yearly_Precip",
            "Frost_Days_March", "Hot_Days_JulAug", "Max_Temp_PreHarvest",
            "Precip_Growing_Season", "NDVI_Mean", "NDVI_Deviation", "Altitude_Mean"
        ]
        for feat in expected_features:
            self.assertIn(feat, merged.columns, f"Expected column {feat} missing in merged dataset")

        self.assertGreater(len(merged), 0)

    def test_handle_missing_values(self):
        """Verify missing values are filled according to selected strategy."""
        sample_df = pd.DataFrame({
            "District": ["Shopian", "Shopian", "Baramulla", "Baramulla"],
            "Year": [2020, 2021, 2020, 2021],
            "Value": [10.0, np.nan, np.nan, 20.0]
        })

        filled_ffill = self.preprocessor.handle_missing_values(sample_df.copy(), strategy="forward_fill")
        self.assertEqual(filled_ffill["Value"].isnull().sum(), 0)

        filled_mean = self.preprocessor.handle_missing_values(sample_df.copy(), strategy="mean")
        self.assertEqual(filled_mean["Value"].isnull().sum(), 0)

    def test_remove_outliers_iqr(self):
        """Verify IQR outlier removal detects and drops extreme values."""
        df_with_outlier = pd.DataFrame({
            "District": ["Shopian"] * 10,
            "Year": list(range(2010, 2020)),
            "Value": [10.0, 10.2, 9.8, 10.1, 10.0, 9.9, 10.3, 10.1, 10.0, 9999.0]
        })

        cleaned = self.preprocessor.remove_outliers(df_with_outlier.copy(), method="iqr")
        self.assertLess(len(cleaned), len(df_with_outlier))
        self.assertNotIn(9999.0, cleaned["Value"].values)

    def test_ndvi_deviation_calculation(self):
        """Verify NDVI rolling deviation aligns with expectations."""
        geo_yearly = self.preprocessor._prepare_geospatial_features()
        geo_with_ndvi = self.preprocessor._engineer_ndvi_deviation(geo_yearly)

        self.assertIn("NDVI_Deviation", geo_with_ndvi.columns)
        self.assertEqual(geo_with_ndvi["NDVI_Deviation"].isnull().sum(), 0)


if __name__ == "__main__":
    unittest.main()

