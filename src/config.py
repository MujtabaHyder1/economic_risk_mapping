"""
Configuration Module
Project settings and parameters for Economic Risk Mapping pipeline
"""

# ============================================================================
# DATA GENERATION SETTINGS
# ============================================================================

# Synthetic data generation parameters
DATA_GENERATION = {
    'random_seed': 42,
    'n_years': 12,  # Years of synthetic data to generate (2012-2024)
    'districts': ['Shopian', 'Baramulla', 'Pulwama'],
    'crops': ['Apples', 'Walnuts', 'Saffron'],

    # Climate data ranges (realistic for Kashmir Valley)
    'climate': {
        'temp_range': (-5, 35),  # Celsius
        'precip_max': 500,  # mm
        'seasonal_pattern': True  # Use seasonal variation
    },

    # Geospatial data ranges
    'geospatial': {
        'ndvi_range': (0, 1),
        'altitude_ranges': {
            'Shopian': (1500, 2500),
            'Baramulla': (1200, 2200),
            'Pulwama': (1400, 2300)
        }
    },

    # Crop yield ranges (metric tons/hectare)
    'crop_yields': {
        'Apples': (20, 40),
        'Walnuts': (2, 4),
        'Saffron': (0.005, 0.015)
    },

    # Farm-gate prices (INR/kg)
    'farm_gate_prices': {
        'Apples': (20, 50),
        'Walnuts': (200, 400),
        'Saffron': (2000, 5000)
    }
}

# ============================================================================
# DATA PREPROCESSING SETTINGS
# ============================================================================

PREPROCESSING = {
    'missing_value_strategy': 'forward_fill',  # 'forward_fill', 'backward_fill', 'mean'
    'outlier_method': 'iqr',  # 'iqr' or 'zscore'
    'outlier_threshold': 3.0,  # Z-score threshold

    # Feature engineering parameters
    'frost_threshold': 0.0,  # Celsius (critical for blooming)
    'heatwave_threshold': 30.0,  # Celsius
    'harvest_months': [7, 8],  # July-August
    'growing_season': list(range(4, 11)),  # April-October
    'ndvi_rolling_window': 10  # Years for rolling average
}

# ============================================================================
# MODEL TRAINING SETTINGS
# ============================================================================

MODEL_TRAINING = {
    'random_seed': 42,
    'test_years': 2,  # Years to hold for testing (final year, 2024)
    'cv_folds': 5,  # Cross-validation folds

    # XGBoost hyperparameters (GridSearchCV space)
    'hyperparameter_grid': {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.85],
        'colsample_bytree': [0.7, 0.85],
        'gamma': [0, 1]
    },

    # Default parameters (if tuning is skipped)
    'default_params': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'gamma': 0,
        'objective': 'reg:squarederror'
    }
}

# ============================================================================
# MODEL EVALUATION SETTINGS
# ============================================================================

MODEL_EVALUATION = {
    'top_features': 15,  # Number of top features to display
    'vulnerability_scenarios': [1.5, 2.0],  # Temperature increase in degC
    'vulnerability_normalization': 'percentile',  # 'percentile' or 'zscore'

    # Risk classification thresholds
    'risk_levels': {
        'Low': (0, 20),
        'Moderate': (20, 40),
        'High': (40, 60),
        'Very High': (60, 80),
        'Critical': (80, 100)
    }
}

# ============================================================================
# PATHS & DIRECTORIES
# ============================================================================

PATHS = {
    'data_raw': 'data/raw/',
    'data_synthetic': 'data/synthetic/',
    'data_processed': 'data/processed/',
    'models': 'models/',
    'results': 'results/',
    'notebooks': 'notebooks/',
    'logs': 'logs/'
}

# ============================================================================
# OUTPUT FILES
# ============================================================================

OUTPUT_FILES = {
    # Data
    'climate_data': 'data/synthetic/climate_data.csv',
    'geospatial_data': 'data/synthetic/geospatial_data.csv',
    'economic_data': 'data/synthetic/economic_data.csv',
    'merged_features': 'data/processed/merged_features.csv',

    # Model
    'model_pkl': 'models/xgboost_model.pkl',

    # Results
    'feature_importance': 'results/feature_importance.png',
    'predictions_vs_actual': 'results/predictions_vs_actual.png',
    'error_distribution': 'results/error_distribution.png',
    'vulnerability_heatmap': 'results/vulnerability_heatmap.png',
    'revenue_impact': 'results/revenue_impact.png',
    'evaluation_metrics': 'results/evaluation_metrics.json',
    'vulnerability_1_5C': 'results/vulnerability_index_1_5C.csv',
    'vulnerability_2_0C': 'results/vulnerability_index_2_0C.csv',
    'summary_report': 'results/summary_report.txt'
}

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

LOGGING = {
    'level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'format': '[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    'file': 'logs/pipeline.log'
}

# ============================================================================
# FEATURE ENGINEERING FEATURES
# ============================================================================

FEATURE_COLUMNS = {
    'climate_anomalies': [
        'Frost_Days_March',
        'Hot_Days_JulAug',
        'Max_Temp_PreHarvest',
        'Precip_Growing_Season'
    ],
    'geospatial': [
        'NDVI_Mean',
        'NDVI_Std',
        'NDVI_Min',
        'NDVI_Deviation',
        'Altitude_Mean'
    ],
    'annual_climate': [
        'Yearly_Avg_Temp',
        'Yearly_Temp_Variability',
        'Yearly_Max_Temp',
        'Yearly_Min_Temp',
        'Yearly_Precip'
    ],
    'economic': [
        'Yield_per_Hectare',
        'Farm_Gate_Price'
    ]
}

# Target variable
TARGET_VARIABLE = 'Revenue_per_Hectare'

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

VISUALIZATION = {
    'dpi': 300,
    'figsize_default': (12, 6),
    'figsize_large': (14, 8),
    'style': 'seaborn-v0_8-darkgrid',
    'colormap': 'viridis',
    'font_size': 11
}

# ============================================================================
# PROJECT METADATA
# ============================================================================

PROJECT = {
    'name': 'Economic Risk Mapping of Horticulture Under Shifting Climatic Zones',
    'version': '1.0.0',
    'description': 'Spatial ML model for predicting per-hectare financial loss of cash crops',
    'focus_region': 'Kashmir Valley',
    'districts': ['Shopian', 'Baramulla', 'Pulwama'],
    'crops': ['Apples', 'Walnuts', 'Saffron'],
    'target_variable': 'Revenue per Hectare (INR)',
    'created': '2026-10-25'
}


def print_config():
    """Print current configuration."""
    import json

    config = {
        'DATA_GENERATION': DATA_GENERATION,
        'PREPROCESSING': PREPROCESSING,
        'MODEL_TRAINING': MODEL_TRAINING,
        'MODEL_EVALUATION': MODEL_EVALUATION,
        'PATHS': PATHS,
        'FEATURE_COLUMNS': FEATURE_COLUMNS,
        'PROJECT': PROJECT
    }

    print(json.dumps(config, indent=2, default=str))


if __name__ == "__main__":
    print_config()
