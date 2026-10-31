"""
Tests for Model Training, Evaluation, and Vulnerability Index Calculations
"""

import unittest
import os
import sys
import tempfile
import pandas as pd
import numpy as np
import xgboost as xgb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from model_training import XGBoostTrainer
from model_evaluation import ModelEvaluator, FinancialVulnerabilityCalculator


class TestModelPipeline(unittest.TestCase):
    """Test suite for XGBoost training and vulnerability calculation."""

    @classmethod
    def setUpClass(cls):
        """Prepare minimal synthetic features dataframe for pipeline testing."""
        np.random.seed(42)
        n_samples = 45
        years = np.repeat([2020, 2021, 2022, 2023, 2024], 9)
        districts = np.tile(["Shopian", "Baramulla", "Pulwama"], 15)
        crops = np.tile(["Apples", "Walnuts", "Saffron"], 15)

        cls.df = pd.DataFrame({
            "Year": years,
            "District": districts,
            "Crop_Type": crops,
            "Frost_Days_March": np.random.randint(0, 8, n_samples),
            "Hot_Days_JulAug": np.random.randint(0, 15, n_samples),
            "Max_Temp_PreHarvest": np.random.uniform(25, 36, n_samples),
            "Precip_Growing_Season": np.random.uniform(150, 400, n_samples),
            "NDVI_Mean": np.random.uniform(0.4, 0.8, n_samples),
            "NDVI_Std": np.random.uniform(0.02, 0.08, n_samples),
            "NDVI_Min": np.random.uniform(0.2, 0.4, n_samples),
            "NDVI_Deviation": np.random.uniform(-0.1, 0.1, n_samples),
            "Altitude_Mean": np.random.uniform(1400, 2400, n_samples),
            "Yearly_Avg_Temp": np.random.uniform(12, 18, n_samples),
            "Yearly_Temp_Variability": np.random.uniform(4, 9, n_samples),
            "Yearly_Max_Temp": np.random.uniform(28, 38, n_samples),
            "Yearly_Min_Temp": np.random.uniform(-3, 3, n_samples),
            "Yearly_Precip": np.random.uniform(400, 900, n_samples),
            "Yield_per_Hectare": np.random.uniform(1, 30, n_samples),
            "Farm_Gate_Price": np.random.uniform(30, 2000, n_samples),
            "Revenue_per_Hectare": np.random.uniform(50000, 800000, n_samples)
        })

    def test_feature_preparation_and_temporal_split(self):
        """Verify feature extraction and ensure temporal split has zero future leakage."""
        trainer = XGBoostTrainer(random_seed=42)
        X, y = trainer.prepare_features(self.df)

        self.assertEqual(len(X), len(self.df))
        self.assertEqual(len(y), len(self.df))
        self.assertNotIn("Revenue_per_Hectare", X.columns)

        X_train, X_test, y_train, y_test = trainer.temporal_train_test_split(
            X, y, self.df, test_years=2
        )

        train_indices = X_train.index
        test_indices = X_test.index

        # Verify no overlap
        self.assertTrue(set(train_indices).isdisjoint(set(test_indices)))
        # Verify years separation (cutoff year is years[-2] = 2023)
        self.assertTrue((self.df.loc[train_indices, "Year"] <= 2023).all())
        self.assertTrue((self.df.loc[test_indices, "Year"] > 2023).all())

    def test_model_training_and_evaluation(self):
        """Train quick XGBoost model and verify metric structure."""
        trainer = XGBoostTrainer(random_seed=42)
        X, y = trainer.prepare_features(self.df)
        X_train, X_test, y_train, y_test = trainer.temporal_train_test_split(
            X, y, self.df, test_years=2
        )

        # Train with small estimators for speed
        best_params = {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1}
        model = trainer.train_model(X_train, y_train, best_params=best_params)

        self.assertIsInstance(model, xgb.XGBRegressor)
        metrics = trainer.evaluate_model(X_train, X_test, y_train, y_test)

        for key in ["train_rmse", "test_rmse", "train_mae", "test_mae", "train_r2", "test_r2"]:
            self.assertIn(key, metrics)
            self.assertFalse(np.isnan(metrics[key]))

    def test_vulnerability_index_bounds_and_structure(self):
        """Verify Financial Vulnerability Index outputs are within [0, 100] and properly formatted."""
        trainer = XGBoostTrainer(random_seed=42)
        X, y = trainer.prepare_features(self.df)
        trainer.train_model(X, y, best_params={"n_estimators": 20, "max_depth": 3})

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
            self.df.to_csv(tmp_path, index=False)

        try:
            calc = FinancialVulnerabilityCalculator(
                model=trainer.model,
                processed_data_path=tmp_path,
                feature_names=trainer.feature_names
            )
            vuln_df = calc.calculate_vulnerability_index(temp_increase=1.5, method="percentile")

            self.assertFalse(vuln_df.empty)
            self.assertIn("Vulnerability_Index", vuln_df.columns)
            self.assertIn("Revenue_Loss_Percentage", vuln_df.columns)

            self.assertTrue((vuln_df["Vulnerability_Index"] >= 0.0).all())
            self.assertTrue((vuln_df["Vulnerability_Index"] <= 100.0).all())
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
