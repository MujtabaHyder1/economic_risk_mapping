"""
Model Training Module
XGBoost regression with hyperparameter tuning and temporal validation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class XGBoostTrainer:
    """Train and tune XGBoost model for economic risk prediction."""

    def __init__(self, random_seed: int = 42):
        """
        Initialize the trainer.

        Args:
            random_seed: For reproducibility
        """
        self.random_seed = random_seed
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.best_params = None
        self.cv_results = None

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variable.

        Selects relevant climate, geospatial, and economic features.
        Excludes non-predictive columns (Date identifiers).

        Args:
            df: Merged dataset with all features

        Returns:
            Tuple of (X features, y target)
        """
        print("Preparing features for modeling...")

        # Define features: climate anomalies, geospatial, and annual metrics
        feature_columns = [
            # Climate Anomalies
            'Frost_Days_March',
            'Hot_Days_JulAug',
            'Max_Temp_PreHarvest',
            'Precip_Growing_Season',

            # Geospatial Features
            'NDVI_Mean',
            'NDVI_Std',
            'NDVI_Min',
            'NDVI_Deviation',
            'Altitude_Mean',

            # Annual Climate Metrics
            'Yearly_Avg_Temp',
            'Yearly_Temp_Variability',
            'Yearly_Max_Temp',
            'Yearly_Min_Temp',
            'Yearly_Precip',

            # Economic Features
            'Yield_per_Hectare',
            'Farm_Gate_Price'
        ]

        # Verify all features exist
        missing_features = [f for f in feature_columns if f not in df.columns]
        if missing_features:
            print(f"  Warning: Missing features {missing_features}")
            feature_columns = [f for f in feature_columns if f in df.columns]

        X = df[feature_columns].copy()
        y = df['Revenue_per_Hectare'].copy()

        self.feature_names = feature_columns

        print(f"  [OK] Features: {len(feature_columns)} predictors")
        print(f"  [OK] Target: Revenue_per_Hectare")
        print(f"  [OK] Shape: X={X.shape}, y={y.shape}")

        return X, y

    def temporal_train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        df_source: pd.DataFrame,
        test_years: int = 2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data temporally (no data leakage).

        Train on historical data, test on most recent years.
        This prevents the model from having access to future information.

        Args:
            X: Feature matrix
            y: Target vector
            df_source: Source dataframe with Year column for temporal awareness
            test_years: Number of most recent years to hold for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print(f"\nPerforming temporal train-test split...")

        # Get unique years and sort
        years = sorted(df_source['Year'].unique())
        cutoff_year = years[-test_years]

        # Create train/test masks
        train_mask = df_source['Year'] <= cutoff_year
        test_mask = df_source['Year'] > cutoff_year

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        print(f"  [OK] Train set: {len(X_train):,} samples ({years[0]}-{cutoff_year})")
        print(f"  [OK] Test set: {len(X_test):,} samples ({cutoff_year+1}-{years[-1]})")

        return X_train, X_test, y_train, y_test

    def hyperparameter_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int = 5,
        param_grid: Dict = None
    ) -> Dict:
        """
        Perform GridSearchCV for XGBoost hyperparameter tuning.

        Tests combinations of key parameters to optimize RMSE.

        Args:
            X_train: Training features
            y_train: Training target
            cv_folds: Number of cross-validation folds
            param_grid: Parameter grid for GridSearchCV (if None, uses defaults)

        Returns:
            Dictionary of best parameters
        """
        print(f"\nPerforming hyperparameter tuning (GridSearchCV, {cv_folds}-fold CV)...")
        print("  Testing parameter combinations...")

        # Use provided grid or defaults
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.85],
                'colsample_bytree': [0.7, 0.85],
                'gamma': [0, 1]
            }

        # Base XGBoost model
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=self.random_seed,
            verbosity=0,
            n_jobs=-1
        )

        # GridSearchCV
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        self.best_params = grid_search.best_params_
        self.cv_results = grid_search.cv_results_

        print(f"\n  [OK] Best CV Score (RMSE): {np.sqrt(-grid_search.best_score_):.4f}")
        print(f"  [OK] Best Parameters:")
        for param, value in self.best_params.items():
            print(f"     - {param}: {value}")

        return self.best_params

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        best_params: Dict = None
    ) -> xgb.XGBRegressor:
        """
        Train XGBoost model with tuned hyperparameters.

        Args:
            X_train: Training features
            y_train: Training target
            best_params: Dictionary of hyperparameters (if None, use defaults)

        Returns:
            Trained XGBoost model
        """
        print("\nTraining XGBoost model...")

        if best_params is None:
            best_params = {
                'n_estimators': 150,
                'max_depth': 3,
                'learning_rate': 0.05,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'gamma': 1,
                'reg_alpha': 0.5,  # L1 regularization
                'reg_lambda': 1.0  # L2 regularization
            }
        else:
            # Add regularization to tuned params if not present
            if 'reg_alpha' not in best_params:
                best_params['reg_alpha'] = 0.5
            if 'reg_lambda' not in best_params:
                best_params['reg_lambda'] = 1.0

        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=self.random_seed,
            verbosity=0,
            **best_params
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train)],
            verbose=False
        )

        print(f"  [OK] Model trained: {self.model.n_estimators} boosting rounds")

        return self.model

    def evaluate_model(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict:
        """
        Evaluate model on train and test sets.

        Calculates: RMSE, MAE, R², MAPE

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target

        Returns:
            Dictionary of evaluation metrics
        """
        print("\nEvaluating model performance...")

        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
        }

        # MAPE (Mean Absolute Percentage Error)
        metrics['train_mape'] = np.mean(np.abs((y_train - y_train_pred) / np.where(y_train != 0, y_train, np.finfo(float).eps))) * 100
        metrics['test_mape'] = np.mean(np.abs((y_test - y_test_pred) / np.where(y_test != 0, y_test, np.finfo(float).eps))) * 100

        # Print results
        print("\n" + "-"*60)
        print("MODEL EVALUATION METRICS")
        print("-"*60)
        print(f"\nTraining Set:")
        print(f"  - RMSE: {metrics['train_rmse']:.4f}")
        print(f"  - MAE:  {metrics['train_mae']:.4f}")
        print(f"  - R²:   {metrics['train_r2']:.4f}")
        print(f"  - MAPE: {metrics['train_mape']:.2f}%")

        print(f"\nTest Set:")
        print(f"  - RMSE: {metrics['test_rmse']:.4f}")
        print(f"  - MAE:  {metrics['test_mae']:.4f}")
        print(f"  - R²:   {metrics['test_r2']:.4f}")
        print(f"  - MAPE: {metrics['test_mape']:.2f}%")

        # Check for overfitting
        rmse_ratio = metrics['test_rmse'] / metrics['train_rmse']
        if rmse_ratio > 1.3:
            print(f"\n  [!] Warning: Possible overfitting (Test/Train RMSE ratio: {rmse_ratio:.2f})")
        else:
            print(f"\n  [OK] Good generalization (Test/Train RMSE ratio: {rmse_ratio:.2f})")

        print("-"*60)

        return metrics

    def save_model(self, model_path: str = "models/xgboost_model.pkl") -> None:
        """
        Save trained model to disk.

        Args:
            model_path: Path to save model
        """
        import os
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"\n[OK] Model saved: {model_path}")

    def load_model(self, model_path: str) -> xgb.XGBRegressor:
        """
        Load trained model from disk.

        Args:
            model_path: Path to model file

        Returns:
            Loaded XGBoost model
        """
        self.model = joblib.load(model_path)
        print(f"[OK] Model loaded: {model_path}")
        return self.model

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Extract and rank feature importance.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame of feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        print(f"\nTop {top_n} Feature Importance:")
        print("-"*60)
        for idx, row in importance_df.head(top_n).iterrows():
            bar = "=" * int(row['Importance'] * 50)
            print(f"  {row['Feature']:30s} {bar} {row['Importance']:.4f}")

        return importance_df


def train_pipeline(
    processed_data_path: str,
    model_output_path: str = "models/xgboost_model.pkl",
    hyperparameter_tune: bool = True,
    cv_folds: int = 5,
    test_years: int = 5
) -> Tuple[xgb.XGBRegressor, Dict]:
    """
    End-to-end training pipeline.

    Args:
        processed_data_path: Path to processed dataset
        model_output_path: Path to save trained model
        hyperparameter_tune: Whether to perform GridSearchCV
        cv_folds: Number of CV folds
        test_years: Number of years to hold for testing

    Returns:
        Tuple of (trained model, evaluation metrics)
    """
    print("\n" + "="*60)
    print("MODEL TRAINING PIPELINE")
    print("="*60 + "\n")

    # Load processed data
    print("Loading processed dataset...")
    df = pd.read_csv(processed_data_path)
    print(f"  [OK] Loaded: {len(df):,} samples, {len(df.columns)} features")

    # Initialize trainer
    trainer = XGBoostTrainer(random_seed=42)

    # Prepare features and target
    X, y = trainer.prepare_features(df)

    # Temporal train-test split
    X_train, X_test, y_train, y_test = trainer.temporal_train_test_split(X, y, df, test_years=test_years)

    # Hyperparameter tuning (optional)
    if hyperparameter_tune:
        # Import config for hyperparameter grid
        from config import MODEL_TRAINING
        best_params = trainer.hyperparameter_tuning(
            X_train,
            y_train,
            cv_folds=cv_folds,
            param_grid=MODEL_TRAINING.get('hyperparameter_grid')
        )
    else:
        best_params = None
        print("\nSkipping hyperparameter tuning (using defaults)")

    # Train model
    trainer.train_model(X_train, y_train, best_params=best_params)

    # Evaluate
    metrics = trainer.evaluate_model(X_train, X_test, y_train, y_test)

    # Feature importance
    trainer.get_feature_importance(top_n=12)

    # Save model
    trainer.save_model(model_output_path)

    return trainer.model, metrics


if __name__ == "__main__":
    model, metrics = train_pipeline(
        "data/processed/merged_features.csv",
        hyperparameter_tune=True
    )
    print("\n[OK] Training pipeline complete!")
