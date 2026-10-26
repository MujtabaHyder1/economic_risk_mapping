# Setup & Installation Guide

## Prerequisites

- **Python**: 3.9 or higher
- **pip**: Latest version
- **Git**: For version control (optional)

## Installation Steps

### 1. Clone or Navigate to Project Directory

```bash
cd economic_risk_mapping
```

### 2. Create Virtual Environment (Recommended)

**On Windows (CMD/PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- pandas (data manipulation)
- numpy (numerical computing)
- scikit-learn (preprocessing & metrics)
- xgboost (gradient boosting model)
- matplotlib & seaborn (visualization)
- scipy (statistical functions)
- joblib (model persistence)
- pyyaml (configuration)

### 4. Verify Installation

```bash
python -c "import xgboost; print(f'XGBoost version: {xgboost.__version__}')"
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"
```

## Running the Pipeline

### Quick Start (Entire Pipeline)

```bash
python main.py
```

This will execute all stages:
1. Generate synthetic datasets
2. Preprocess and engineer features
3. Train and tune XGBoost model
4. Evaluate performance
5. Calculate Financial Vulnerability Index
6. Generate visualizations

**Expected Runtime**: 3-5 minutes (depending on system specs)

### Running Individual Stages

#### Stage 1: Generate Synthetic Data

```bash
python -c "from src.data_generation import SyntheticDataGenerator; \
gen = SyntheticDataGenerator(n_years=12); \
gen.save_datasets('data/synthetic/')"
```

#### Stage 2: Preprocess Data

```bash
python -c "from src.data_preprocessing import preprocess_data; \
df = preprocess_data('data/synthetic/climate_data.csv', \
                     'data/synthetic/geospatial_data.csv', \
                     'data/synthetic/economic_data.csv')"
```

#### Stage 3: Train Model

```bash
python -c "from src.model_training import train_pipeline; \
model, metrics = train_pipeline('data/processed/merged_features.csv', \
                                hyperparameter_tune=True)"
```

#### Stage 4: Generate Evaluation Report

```bash
python -c "from src.model_evaluation import generate_evaluation_report; \
import joblib; \
model = joblib.load('models/xgboost_model.pkl'); \
# ... (see main.py for full integration)"
```

## Configuration

All pipeline parameters are defined in `src/config.py`:

```python
# Data generation parameters
DATA_GENERATION['n_years'] = 15  # Generate 15 years instead of 12

# Model training parameters
MODEL_TRAINING['cv_folds'] = 10  # Use 10-fold CV instead of 5

# Vulnerability analysis parameters
MODEL_EVALUATION['vulnerability_scenarios'] = [1.5, 2.0, 2.5]  # Add 2.5°C scenario
```

Edit `src/config.py` to customize the pipeline before running.

## Project Structure

```
economic_risk_mapping/
├── README.md                          # Project overview
├── requirements.txt                   # Python dependencies
├── main.py                            # Main pipeline orchestration
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # Configuration parameters
│   ├── data_generation.py             # Synthetic data creation
│   ├── data_preprocessing.py          # Feature engineering
│   ├── model_training.py              # XGBoost training & tuning
│   ├── model_evaluation.py            # Evaluation & visualizations
│   └── utils.py                       # Utility functions
│
├── data/
│   ├── raw/                           # Raw datasets (for real data)
│   ├── synthetic/                     # Generated synthetic data
│   │   ├── climate_data.csv
│   │   ├── geospatial_data.csv
│   │   └── economic_data.csv
│   └── processed/
│       └── merged_features.csv        # Final preprocessed dataset
│
├── models/
│   └── xgboost_model.pkl              # Trained model
│
├── results/
│   ├── feature_importance.png         # Feature importance plot
│   ├── predictions_vs_actual.png      # Prediction accuracy visualization
│   ├── error_distribution.png         # Error analysis
│   ├── vulnerability_heatmap.png      # Risk heatmap by district/crop
│   ├── revenue_impact.png             # Climate scenario impact
│   ├── evaluation_metrics.json        # Model performance metrics
│   ├── vulnerability_index_1_5C.csv   # Risk index (+1.5°C scenario)
│   ├── vulnerability_index_2_0C.csv   # Risk index (+2.0°C scenario)
│   └── summary_report.txt             # Executive summary
│
├── notebooks/
│   └── eda_and_analysis.ipynb         # Exploratory analysis (future)
│
└── logs/
    └── pipeline.log                   # Execution logs
```

## Output Interpretation

### 1. Feature Importance Plot
Shows which climate anomalies and geospatial factors most strongly influence revenue loss.
- **Top features**: Focus adaptation strategies here
- **Red/Hot colors**: High importance

### 2. Predictions vs Actual
Scatter plot with perfect prediction line (45°).
- **Points near line**: Good predictions
- **Points far from line**: Systematic errors or outliers
- **R² value**: Overall model fit (closer to 1.0 = better)

### 3. Vulnerability Heatmap
Financial Vulnerability Index (0-100) by district and crop.
- **Green (0-20)**: Low Risk
- **Yellow (20-40)**: Moderate Risk
- **Orange (40-60)**: High Risk
- **Red (60-80)**: Very High Risk
- **Dark Red (80-100)**: Critical Risk

### 4. Revenue Impact Chart
Shows baseline (current climate) vs scenario (+1.5°C) revenue.
- **Height difference**: Expected revenue loss under climate change
- **Used for**: Identifying most vulnerable districts

### 5. Vulnerability Index CSV Files
Detailed district-crop combinations with:
- Baseline & scenario revenue
- Revenue loss percentage
- Vulnerability Index (0-100)

## Integrating Real Data

### Step 1: Prepare Real Datasets

Create three CSV files with columns:
- **climate_data.csv**: Date, District, Avg_Temp, Max_Temp, Min_Temp, Precipitation_mm
- **geospatial_data.csv**: Date, District, NDVI_score, Arable_Land_Altitude
- **economic_data.csv**: Year, District, Crop_Type, Yield_per_Hectare, Farm_Gate_Price, Revenue_per_Hectare

### Step 2: Place in Data Directory

```bash
cp your_climate_data.csv data/raw/
cp your_geospatial_data.csv data/raw/
cp your_economic_data.csv data/raw/
```

### Step 3: Modify main.py

Replace data loading in Stage 1:
```python
# Instead of synthetic data generation
# climate_df = pd.read_csv("data/raw/climate_data.csv")
# geo_df = pd.read_csv("data/raw/geospatial_data.csv")
# econ_df = pd.read_csv("data/raw/economic_data.csv")
```

### Step 4: Re-run Pipeline

```bash
python main.py
```

## Troubleshooting

### Issue: "No module named 'xgboost'"

**Solution**: 
```bash
pip install xgboost==2.0.3
```

### Issue: "Memory Error" during training

**Solution**: Reduce `cv_folds` or `n_years` in `src/config.py`

### Issue: Model predictions all zeros

**Solution**: Check feature scaling and target variable distribution in preprocessed data

### Issue: Feature importance plot is blank

**Solution**: Ensure model is trained before evaluation. Check `models/xgboost_model.pkl` exists.

## Performance Tips

1. **Reduce hyperparameter tuning time**: Set `hyperparameter_tune=False` in `main.py`
2. **Use fewer years**: Change `n_years=5` in `SyntheticDataGenerator`
3. **Disable visualizations**: Comment out plotting functions temporarily
4. **Use GPU acceleration**: Install XGBoost with CUDA support

## Next Steps

1. **Real Data Integration**: Replace synthetic data with actual climate/economic records
2. **Feature Refinement**: Add crop-specific phenological features
3. **Spatial Analysis**: Implement spatial clustering by altitude zones
4. **Model Ensembling**: Combine XGBoost with other algorithms
5. **REST API**: Wrap model for real-time predictions
6. **Dashboard**: Create web interface for risk visualization

## Documentation

- **Data Generation**: See docstrings in `src/data_generation.py`
- **Feature Engineering**: See `src/data_preprocessing.py` for anomaly calculations
- **Model Architecture**: See `src/model_training.py` for XGBoost configuration
- **Evaluation Metrics**: See `src/model_evaluation.py` for metric definitions

## Support & Contact

For questions or improvements, refer to README.md or individual module docstrings.

---

**Created**: 2026-10-25  
**Version**: 1.0.0  
**Project**: Economic Risk Mapping of Horticulture Under Shifting Climatic Zones
