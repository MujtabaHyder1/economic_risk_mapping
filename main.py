"""
Main Pipeline Orchestration Script
End-to-end execution of Economic Risk Mapping ML pipeline
"""

import sys
import time
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generation import SyntheticDataGenerator
from data_preprocessing import preprocess_data
from model_training import train_pipeline
from model_evaluation import generate_evaluation_report
from utils import (
    create_project_directories,
    setup_logging,
    print_project_banner,
    print_section,
    log_pipeline_step,
    generate_summary_report
)
from config import DATA_GENERATION, MODEL_TRAINING, LOGGING
import pandas as pd
import joblib


def main():
    """
    Execute complete Economic Risk Mapping pipeline.

    Pipeline stages:
    1. Generate synthetic datasets (climate, geospatial, economic)
    2. Preprocess and engineer features
    3. Train and tune XGBoost model
    4. Evaluate model and generate visualizations
    5. Calculate Financial Vulnerability Index
    """

    start_time = time.time()

    # Print header
    print_project_banner("ECONOMIC RISK MAPPING OF HORTICULTURE", 70)
    print("Shifting Climatic Zones Impact Analysis")
    print("Focus: Kashmir Valley (Shopian, Baramulla, Pulwama)")
    print("="*70 + "\n")

    # Create project directories
    print("Initializing project structure...")
    create_project_directories(base_dir=".")
    print("[OK] Directory structure created\n")

    # Initialize file + console logging
    setup_logging(
        log_file=LOGGING['file'],
        level=LOGGING['level']
    )

    # =========================================================================
    # STAGE 1: DATA GENERATION
    # =========================================================================
    step_start = time.time()
    log_pipeline_step("STAGE 1: Synthetic Data Generation", "STARTED")

    generator = SyntheticDataGenerator(
        random_seed=DATA_GENERATION['random_seed'],
        n_years=DATA_GENERATION['n_years']
    )

    try:
        climate_df, geo_df, econ_df = generator.save_datasets(
            output_dir="data/synthetic/"
        )
        step_duration = time.time() - step_start
        log_pipeline_step("STAGE 1: Synthetic Data Generation", "COMPLETED", step_duration)
    except Exception as e:
        print(f"[FAIL] STAGE 1 FAILED: {e}")
        return 1

    # =========================================================================
    # STAGE 2: DATA PREPROCESSING & FEATURE ENGINEERING
    # =========================================================================
    step_start = time.time()
    log_pipeline_step("STAGE 2: Data Preprocessing", "STARTED")

    try:
        merged_df = preprocess_data(
            climate_path="data/synthetic/climate_data.csv",
            geo_path="data/synthetic/geospatial_data.csv",
            econ_path="data/synthetic/economic_data.csv",
            output_path="data/processed/merged_features.csv"
        )
        step_duration = time.time() - step_start
        log_pipeline_step("STAGE 2: Data Preprocessing", "COMPLETED", step_duration)
    except Exception as e:
        print(f"[FAIL] STAGE 2 FAILED: {e}")
        return 1

    # =========================================================================
    # STAGE 3: MODEL TRAINING & TUNING
    # =========================================================================
    step_start = time.time()
    log_pipeline_step("STAGE 3: Model Training", "STARTED")

    try:
        model, metrics = train_pipeline(
            processed_data_path="data/processed/merged_features.csv",
            model_output_path="models/xgboost_model.pkl",
            hyperparameter_tune=True,
            cv_folds=MODEL_TRAINING['cv_folds'],
            test_years=MODEL_TRAINING['test_years']
        )
        step_duration = time.time() - step_start
        log_pipeline_step("STAGE 3: Model Training", "COMPLETED", step_duration)
    except Exception as e:
        print(f"[FAIL] STAGE 3 FAILED: {e}")
        return 1

    # =========================================================================
    # STAGE 4: MODEL EVALUATION & VISUALIZATION
    # =========================================================================
    step_start = time.time()
    log_pipeline_step("STAGE 4: Evaluation & Visualization", "STARTED")

    try:
        # Prepare test data for evaluation
        from model_training import XGBoostTrainer
        from data_preprocessing import DataPreprocessor

        df = pd.read_csv("data/processed/merged_features.csv")
        trainer = XGBoostTrainer()
        X, y = trainer.prepare_features(df)
        X_train, X_test, y_train, y_test = trainer.temporal_train_test_split(X, y, df, test_years=2)

        # Generate comprehensive evaluation report
        generate_evaluation_report(
            model=model,
            X_test=X_test,
            y_test=y_test,
            metrics=metrics,
            feature_names=trainer.feature_names,
            processed_data_path="data/processed/merged_features.csv",
            output_dir="results/"
        )

        step_duration = time.time() - step_start
        log_pipeline_step("STAGE 4: Evaluation & Visualization", "COMPLETED", step_duration)
    except Exception as e:
        print(f"[FAIL] STAGE 4 FAILED: {e}")
        return 1

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_duration = time.time() - start_time

    print_section("PIPELINE EXECUTION SUMMARY", 70)

    print("\n[OK] All pipeline stages completed successfully!\n")

    print(f"Total Execution Time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)\n")

    print("Generated Artifacts:")
    print("\n  Data:")
    print("    - data/synthetic/climate_data.csv")
    print("    - data/synthetic/geospatial_data.csv")
    print("    - data/synthetic/economic_data.csv")
    print("    - data/processed/merged_features.csv")

    print("\n  Model:")
    print("    - models/xgboost_model.pkl")

    print("\n  Visualizations:")
    print("    - results/feature_importance.png")
    print("    - results/predictions_vs_actual.png")
    print("    - results/error_distribution.png")
    print("    - results/vulnerability_heatmap.png")
    print("    - results/revenue_impact.png")

    print("\n  Reports & Metrics:")
    print("    - results/evaluation_metrics.json")
    print("    - results/vulnerability_index_1_5C.csv")
    print("    - results/vulnerability_index_2_0C.csv")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)

    print("""
1. INTEGRATE REAL DATA
   Replace synthetic data with actual climate, geospatial, and economic datasets
   Update data_generation.py with real data loaders

2. REFINE FEATURES
   - Validate engineered features with agricultural domain experts
   - Add crop-specific features (phenology, irrigation patterns)
   - Consider spatial autocorrelation for neighboring districts

3. ADVANCED ANALYSIS
   - Perform spatial clustering analysis (K-means by altitude zones)
   - Implement ensemble models (XGBoost + LightGBM + Random Forest)
   - Add time-series forecasting for multi-year projections

4. DEPLOYMENT
   - Wrap model in REST API (Flask/FastAPI)
   - Create web dashboard for risk visualization
   - Implement real-time predictions for current season

5. MONITORING
   - Track model drift with new data
   - Schedule monthly retraining cycles
   - Log predictions for validation against actual outcomes

6. STAKEHOLDER ENGAGEMENT
   - Present findings to agricultural policy makers
   - Develop user-friendly interface for farmer access
   - Create actionable adaptation strategies based on risk profiles

For questions or improvements, refer to:
  - README.md - Project overview and setup
  - src/data_generation.py - Synthetic data configuration
  - src/data_preprocessing.py - Feature engineering logic
  - src/model_training.py - Model architecture and tuning
  - src/model_evaluation.py - Evaluation metrics and visualizations
""")

    print("="*70 + "\n")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
