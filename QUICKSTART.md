
## Author

**Mujtaba Hyder**

## Installation & Execution (2 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
python main.py
```
**Output**: All results generated in ~55 seconds

### 3. Launch Interactive Web Dashboard
```bash
streamlit run app.py
```
Opens interactive climate scenario laboratory at `http://localhost:8501`.

### 4. Run Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## What Was Built

### ✅ Complete ML Pipeline & Platform

**Core Modules & Apps:**
1. **app.py** - Interactive Streamlit web dashboard for live scenario simulation
2. **data_generation.py** - Generates realistic synthetic datasets (14k+ rows)
3. **data_preprocessing.py** - Merges data & engineers 16 climate/geospatial features
4. **model_training.py** - Trains XGBoost with hyperparameter tuning (1,080 fits)
5. **model_evaluation.py** - Generates visualizations & Financial Vulnerability Index
6. **notebooks/eda_and_analysis.ipynb** - Exploratory data analysis & workflow notebook
7. **tests/** - Comprehensive unit and integration test suite (11 tests)

---

## Key Results

### Model Performance
- **Test R²**: 0.9772 (explains 97.72% of revenue variance)
- **Test RMSE**: ₹62,994 per hectare
- **Test MAPE**: 18.95%

### Financial Vulnerability Index (1.5°C warming)

| District | Risk Level | Avg Loss |
|----------|-----------|----------|
| Baramulla | HIGH (40.5) | -12.1% |
| Shopian | MODERATE (38.0) | -11.3% |
| Pulwama | LOW (18.8) | -5.6% |

**Most Vulnerable**: Saffron in Shopian (-29.7% revenue loss)  
**Most Resilient**: Apples in Pulwama (-4.7% revenue loss)

---

## Generated Outputs

### Data
- `data/synthetic/climate_data.csv` - 14,247 daily climate records
- `data/synthetic/geospatial_data.csv` - 468 NDVI/altitude records
- `data/synthetic/economic_data.csv` - 117 yearly crop records
- `data/processed/merged_features.csv` - Final dataset (108 samples, 16 features)

### Model
- `models/xgboost_model.pkl` - Trained model (production-ready)

### Visualizations (PNG, 300 DPI)
- `results/feature_importance.png` - Top 15 features driving risk
- `results/predictions_vs_actual.png` - Model accuracy scatter plot
- `results/error_distribution.png` - Prediction error analysis
- `results/vulnerability_heatmap.png` - Risk by district & crop
- `results/revenue_impact.png` - Baseline vs climate scenario

### Reports & Metrics
- `results/evaluation_metrics.json` - Detailed performance metrics
- `results/vulnerability_index_1_5C.csv` - Risk analysis (+1.5°C scenario)
- `results/vulnerability_index_2_0C.csv` - Risk analysis (+2.0°C scenario)

---

## Feature Engineering Highlights

### Climate Anomalies
- **Frost_Days_March** - Critical for blooming (apples/walnuts)
- **Hot_Days_JulAug** - Pre-harvest heatwave impact
- **NDVI_Deviation** - Vegetation stress indicator
- **Precip_Growing_Season** - Water availability (Apr-Oct)

### Key Finding
Yield (53%) + Price (15%) = 68% of revenue variance  
Climate anomalies = 25%+ of remaining variance

---

## Customization

### Change Number of Years
Edit `main.py` line ~62:
```python
generator = SyntheticDataGenerator(n_years=20)  # Generate 20 years instead of 12
```

### Disable Hyperparameter Tuning (faster execution)
Edit `main.py` line ~75:
```python
model, metrics = train_pipeline(..., hyperparameter_tune=False)
```

### Adjust Temperature Scenarios
Edit `src/model_evaluation.py`:
```python
vuln_df = calculator.calculate_vulnerability_index(temp_increase=2.5)  # 2.5°C instead of 1.5°C
```

### Modify Feature Engineering
Edit `src/config.py` to adjust thresholds:
```python
'frost_threshold': -2.0,  # Instead of 0°C
'heatwave_threshold': 35.0,  # Instead of 30°C
```

---

## Integration with Real Data

### Step 1: Prepare Your Data
Create three CSV files:
- `climate_data.csv`: Date, District, Avg_Temp, Max_Temp, Min_Temp, Precipitation_mm
- `geospatial_data.csv`: Date, District, NDVI_score, Arable_Land_Altitude
- `economic_data.csv`: Year, District, Crop_Type, Yield_per_Hectare, Farm_Gate_Price, Revenue_per_Hectare

### Step 2: Place in Data Directory
```bash
cp your_climate_data.csv data/raw/
cp your_geospatial_data.csv data/raw/
cp your_economic_data.csv data/raw/
```

### Step 3: Modify main.py
Replace synthetic data generation with:
```python
climate_df = pd.read_csv("data/raw/climate_data.csv")
geo_df = pd.read_csv("data/raw/geospatial_data.csv")
econ_df = pd.read_csv("data/raw/economic_data.csv")
```

### Step 4: Run Pipeline
```bash
python main.py
```

---

## Code Structure Overview

```
economic_risk_mapping/
├── main.py                    # Run this to execute everything
├── README.md                  # Project overview
├── SETUP.md                   # Detailed installation guide
├── IMPLEMENTATION_SUMMARY.md  # Complete technical report
├── requirements.txt           # Dependencies
│
├── src/
│   ├── config.py              # All tunable parameters
│   ├── data_generation.py     # Synthetic data creation
│   ├── data_preprocessing.py  # Feature engineering
│   ├── model_training.py      # XGBoost training
│   ├── model_evaluation.py    # Visualizations & metrics
│   └── utils.py               # Helper functions
│
├── data/
│   ├── synthetic/             # Generated data
│   └── processed/             # Merged & engineered features
│
├── models/
│   └── xgboost_model.pkl      # Trained model
│
└── results/
    ├── *.png                  # Visualizations
    ├── *.csv                  # Risk indices
    └── *.json                 # Metrics
```

---

## Performance Tips

| Task | Time | How to Speed Up |
|------|------|-----------------|
| Full pipeline | ~60s | Set `hyperparameter_tune=False` |
| Hyperparameter tuning | ~55s | Reduce `cv_folds=3` or `n_years=5` |
| Data generation | <1s | Adjust in config |
| Model training | ~3-5s | Reduce `n_estimators` in config |
| Visualization | ~3s | Skip plots if not needed |

---

## Interpretation Guide

### Feature Importance Chart
- **Height of bar** = Importance score (0-1)
- **Longer bars** = Feature more strongly predicts revenue
- **Top features** = Focus point for climate adaptation

### Predictions vs Actual Plot
- **Points near red diagonal** = Good predictions
- **Points far from diagonal** = Model uncertainty
- **R² = 0.98** = Excellent fit (close to line)

### Vulnerability Heatmap
- **Green** (0-20) = Low Risk - crop resilient to warming
- **Yellow** (20-40) = Moderate Risk - adaptations needed
- **Orange** (40-60) = High Risk - significant mitigation required
- **Red** (60-100) = Critical Risk - urgent intervention needed

### Revenue Impact Chart
- **Height difference between bars** = Expected revenue loss
- **Larger difference** = Higher climate vulnerability
- **Used for** = Prioritizing support to affected districts

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "ModuleNotFoundError: xgboost" | Run `pip install -r requirements.txt` |
| "Memory Error" | Reduce `n_years` or `cv_folds` in config |
| Low R² on real data | Check feature distributions; validate with domain experts |
| Predictions all zeros | Check target variable distribution in processed data |
| Visualizations not saved | Ensure `results/` directory exists |

---

## Documentation Files

| File | Purpose |
|------|---------|
| **README.md** | Project overview & references |
| **SETUP.md** | Installation, environment setup, detailed usage |
| **IMPLEMENTATION_SUMMARY.md** | Complete technical documentation & findings |
| **This file** | Quick start & quick reference |

---

## Next: What to Do With Results

### For Policy Makers
- Review vulnerability_index*.csv files
- Focus adaptive strategies on highest-risk crops/districts
- Plan irrigation investments for water availability

### For Farmers
- Export risk predictions for your crop type
- Plan crop varieties based on climate vulnerability
- Schedule irrigation/frost protection accordingly

### For Researchers
- Validate feature engineering with field experiments
- Refine climate thresholds with actual damage records
- Develop ensemble models for higher accuracy

---

## Support

For issues or questions:
1. Check SETUP.md for common problems
2. Review IMPLEMENTATION_SUMMARY.md for technical details
3. Examine docstrings in source code (e.g., `help(SyntheticDataGenerator)`)
4. Check configuration in `src/config.py` to adjust parameters

---

**Status**: ✅ Production-Ready  
**Last Updated**: 2026-10-25  
**Version**: 1.0.0
