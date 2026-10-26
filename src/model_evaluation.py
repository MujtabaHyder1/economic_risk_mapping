"""
Model Evaluation and Visualization Module
Generates metrics, visualizations, and Financial Vulnerability Index
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Generate comprehensive evaluations and visualizations."""

    def __init__(self, model, X_test: pd.DataFrame, y_test: pd.Series, feature_names: list):
        """
        Initialize evaluator.

        Args:
            model: Trained XGBoost model
            X_test: Test features
            y_test: Test target
            feature_names: List of feature names
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.y_pred = model.predict(X_test)

    def plot_feature_importance(
        self,
        output_path: str = "results/feature_importance.png",
        top_n: int = 15
    ) -> None:
        """
        Plot top N features by importance.

        Args:
            output_path: Path to save plot
            top_n: Number of top features to display
        """
        print("Generating feature importance plot...")

        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True).tail(top_n)

        plt.figure(figsize=(10, 8))
        colors = sns.color_palette("viridis", len(importance_df))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.ylabel('Feature', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Feature Importance - Economic Risk Model',
                  fontsize=14, fontweight='bold', pad=20)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] Saved: {output_path}")

    def plot_predictions_vs_actual(
        self,
        output_path: str = "results/predictions_vs_actual.png"
    ) -> None:
        """
        Plot predicted vs actual revenue.

        Args:
            output_path: Path to save plot
        """
        print("Generating predictions vs actual plot...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Scatter plot
        ax = axes[0]
        ax.scatter(self.y_test, self.y_pred, alpha=0.6, s=50, color='steelblue')
        min_val = min(self.y_test.min(), self.y_pred.min())
        max_val = max(self.y_test.max(), self.y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Revenue per Hectare (INR)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Predicted Revenue per Hectare (INR)', fontsize=11, fontweight='bold')
        ax.set_title('Predictions vs Actual (Test Set)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Residuals plot
        ax = axes[1]
        residuals = self.y_test - self.y_pred
        ax.scatter(self.y_pred, residuals, alpha=0.6, s=50, color='coral')
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Revenue per Hectare (INR)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Residuals (INR)', fontsize=11, fontweight='bold')
        ax.set_title('Residual Plot (Test Set)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

        plt.tight_layout()

        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] Saved: {output_path}")

    def plot_error_distribution(
        self,
        output_path: str = "results/error_distribution.png"
    ) -> None:
        """
        Plot distribution of prediction errors.

        Args:
            output_path: Path to save plot
        """
        print("Generating error distribution plot...")

        errors = self.y_test - self.y_pred
        mape = np.abs((self.y_test - self.y_pred) / np.where(self.y_test != 0, self.y_test, np.finfo(float).eps)) * 100

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Absolute error distribution
        ax = axes[0]
        ax.hist(np.abs(errors), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(np.abs(errors)), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(np.abs(errors)):.2f}')
        ax.set_xlabel('Absolute Error (INR)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Distribution of Absolute Prediction Errors', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # MAPE distribution
        ax = axes[1]
        ax.hist(mape, bins=30, color='coral', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(mape), color='red', linestyle='--',
                   linewidth=2, label=f'Mean MAPE: {np.mean(mape):.2f}%')
        ax.set_xlabel('Mean Absolute Percentage Error (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Distribution of MAPE (%)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] Saved: {output_path}")


class FinancialVulnerabilityCalculator:
    """Calculate Financial Vulnerability Index for climate scenarios."""

    def __init__(self, model, processed_data_path: str, feature_names: list):
        """
        Initialize calculator.

        Args:
            model: Trained XGBoost model
            processed_data_path: Path to processed dataset
            feature_names: List of feature names used in model
        """
        self.model = model
        self.feature_names = feature_names
        self.df = pd.read_csv(processed_data_path)
        self.districts = self.df['District'].unique()
        self.crops = self.df['Crop_Type'].unique()

    def calculate_vulnerability_index(
        self,
        temp_increase: float = 1.5,
        method: str = 'percentile'
    ) -> pd.DataFrame:
        """
        Calculate Financial Vulnerability Index (0-100 scale).

        Simulates climate scenario with temperature increase and predicts revenue loss.

        Args:
            temp_increase: Degrees Celsius increase in average temperature
            method: 'percentile' (default) or 'zscore' for normalization

        Returns:
            DataFrame with vulnerability indices by district and crop
        """
        print(f"\nCalculating Financial Vulnerability Index (Scenario: +{temp_increase}degC)...")

        vulnerability_results = []

        for district in self.districts:
            for crop in self.crops:
                # Filter data for this district-crop combination
                subset = self.df[
                    (self.df['District'] == district) &
                    (self.df['Crop_Type'] == crop)
                ].copy()

                if len(subset) == 0:
                    continue

                # Prepare baseline features
                X_baseline = subset[self.feature_names].copy()
                baseline_pred = self.model.predict(X_baseline).mean()

                # Create climate scenario: increase temperature features
                X_scenario = X_baseline.copy()
                temp_cols = [col for col in self.feature_names if 'Temp' in col or 'Heat' in col]
                X_scenario[temp_cols] = X_scenario[temp_cols] + temp_increase

                # Adjust frost days (fewer with warming)
                if 'Frost_Days_March' in X_scenario.columns:
                    X_scenario['Frost_Days_March'] = np.maximum(
                        0,
                        X_scenario['Frost_Days_March'] - (temp_increase * 5)
                    )

                # Adjust NDVI (potentially stressed vegetation)
                if 'NDVI_Deviation' in X_scenario.columns:
                    X_scenario['NDVI_Deviation'] = X_scenario['NDVI_Deviation'] - (temp_increase * 0.05)

                scenario_pred = self.model.predict(X_scenario).mean()

                # Calculate revenue loss
                revenue_loss_pct = max(0, (baseline_pred - scenario_pred) / np.where(baseline_pred != 0, baseline_pred, np.finfo(float).eps) * 100)

                vulnerability_results.append({
                    'District': district,
                    'Crop_Type': crop,
                    'Baseline_Revenue': round(baseline_pred, 2),
                    'Scenario_Revenue': round(scenario_pred, 2),
                    'Revenue_Loss_Percentage': round(revenue_loss_pct, 2),
                    'Climate_Scenario': f'+{temp_increase}degC'
                })

        vuln_df = pd.DataFrame(vulnerability_results)

        # Normalize to 0-100 scale (Financial Vulnerability Index)
        if method == 'percentile':
            max_loss = vuln_df['Revenue_Loss_Percentage'].max()
            if max_loss > 0:
                vuln_df['Vulnerability_Index'] = (
                    vuln_df['Revenue_Loss_Percentage'] / max_loss * 100
                ).round(1)
            else:
                vuln_df['Vulnerability_Index'] = 0.0

        elif method == 'zscore':
            mean_loss = vuln_df['Revenue_Loss_Percentage'].mean()
            std_loss = vuln_df['Revenue_Loss_Percentage'].std()
            if std_loss > 0:
                vuln_df['Vulnerability_Index'] = (
                    ((vuln_df['Revenue_Loss_Percentage'] - mean_loss) / std_loss) * 15 + 50
                ).clip(0, 100).round(1)
            else:
                vuln_df['Vulnerability_Index'] = 50.0

        print("\n" + "-"*80)
        print("FINANCIAL VULNERABILITY INDEX BY DISTRICT & CROP")
        print("-"*80)

        # Summary by district
        district_summary = vuln_df.groupby('District').agg({
            'Vulnerability_Index': 'mean',
            'Revenue_Loss_Percentage': 'mean'
        }).round(1).sort_values('Vulnerability_Index', ascending=False)

        for district, row in district_summary.iterrows():
            risk_level = self._get_risk_level(row['Vulnerability_Index'])
            print(f"\n{district}:")
            print(f"  - Vulnerability Index: {row['Vulnerability_Index']:.1f}/100 ({risk_level})")
            print(f"  - Expected Revenue Loss: {row['Revenue_Loss_Percentage']:.1f}%")

            # Crop breakdown
            crop_data = vuln_df[vuln_df['District'] == district].sort_values(
                'Vulnerability_Index', ascending=False
            )
            for _, crop_row in crop_data.iterrows():
                crop_risk = self._get_risk_level(crop_row['Vulnerability_Index'])
                print(f"     - {crop_row['Crop_Type']:12s}: {crop_row['Vulnerability_Index']:5.1f} ({crop_risk})")

        print("\n" + "-"*80)

        return vuln_df

    def _get_risk_level(self, index: float) -> str:
        """
        Classify risk level based on vulnerability index.

        Args:
            index: Vulnerability Index (0-100)

        Returns:
            Risk level string
        """
        if index < 20:
            return "Low Risk"
        elif index < 40:
            return "Moderate Risk"
        elif index < 60:
            return "High Risk"
        elif index < 80:
            return "Very High Risk"
        else:
            return "Critical Risk"

    def plot_vulnerability_heatmap(
        self,
        vuln_df: pd.DataFrame,
        output_path: str = "results/vulnerability_heatmap.png"
    ) -> None:
        """
        Plot vulnerability index as heatmap by district and crop.

        Args:
            vuln_df: Vulnerability results dataframe
            output_path: Path to save plot
        """
        print("Generating vulnerability heatmap...")

        # Pivot for heatmap
        pivot_data = vuln_df.pivot_table(
            index='District',
            columns='Crop_Type',
            values='Vulnerability_Index',
            aggfunc='mean'
        )

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn_r',
            cbar_kws={'label': 'Vulnerability Index (0-100)'},
            vmin=0,
            vmax=100,
            linewidths=1,
            linecolor='gray'
        )
        plt.title('Financial Vulnerability Index by District & Crop\n(+1.5degC Temperature Scenario)',
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Crop Type', fontsize=12, fontweight='bold')
        plt.ylabel('District', fontsize=12, fontweight='bold')
        plt.tight_layout()

        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] Saved: {output_path}")

    def plot_revenue_impact(
        self,
        vuln_df: pd.DataFrame,
        output_path: str = "results/revenue_impact.png"
    ) -> None:
        """
        Plot baseline vs scenario revenue by district.

        Args:
            vuln_df: Vulnerability results dataframe
            output_path: Path to save plot
        """
        print("Generating revenue impact plot...")

        district_summary = vuln_df.groupby('District').agg({
            'Baseline_Revenue': 'mean',
            'Scenario_Revenue': 'mean'
        }).reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(district_summary))
        width = 0.35

        bars1 = ax.bar(
            x - width/2,
            district_summary['Baseline_Revenue'],
            width,
            label='Baseline (Current Climate)',
            color='steelblue',
            alpha=0.8
        )
        bars2 = ax.bar(
            x + width/2,
            district_summary['Scenario_Revenue'],
            width,
            label='Scenario (+1.5degC)',
            color='coral',
            alpha=0.8
        )

        ax.set_xlabel('District', fontsize=12, fontweight='bold')
        ax.set_ylabel('Revenue per Hectare (INR)', fontsize=12, fontweight='bold')
        ax.set_title('Revenue Impact: Baseline vs Climate Scenario (+1.5degC)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(district_summary['District'])
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'₹{height:,.0f}',
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] Saved: {output_path}")


def generate_evaluation_report(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics: Dict,
    feature_names: list,
    processed_data_path: str,
    output_dir: str = "results/"
) -> None:
    """
    Generate complete evaluation report with visualizations.

    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test target
        metrics: Dictionary of evaluation metrics
        feature_names: List of feature names
        processed_data_path: Path to processed dataset
        output_dir: Output directory for results
    """
    import os

    print("\n" + "="*60)
    print("GENERATING EVALUATION REPORT & VISUALIZATIONS")
    print("="*60 + "\n")

    os.makedirs(output_dir, exist_ok=True)

    # Generate visualizations
    evaluator = ModelEvaluator(model, X_test, y_test, feature_names)
    evaluator.plot_feature_importance(f"{output_dir}feature_importance.png", top_n=15)
    evaluator.plot_predictions_vs_actual(f"{output_dir}predictions_vs_actual.png")
    evaluator.plot_error_distribution(f"{output_dir}error_distribution.png")

    # Calculate vulnerability index
    calculator = FinancialVulnerabilityCalculator(model, processed_data_path, feature_names)

    # 1.5degC scenario
    vuln_df_1_5 = calculator.calculate_vulnerability_index(temp_increase=1.5)
    vuln_df_1_5.to_csv(f"{output_dir}vulnerability_index_1_5C.csv", index=False)

    # 2.0degC scenario
    vuln_df_2_0 = calculator.calculate_vulnerability_index(temp_increase=2.0)
    vuln_df_2_0.to_csv(f"{output_dir}vulnerability_index_2_0C.csv", index=False)

    calculator.plot_vulnerability_heatmap(vuln_df_1_5, f"{output_dir}vulnerability_heatmap.png")
    calculator.plot_revenue_impact(vuln_df_1_5, f"{output_dir}revenue_impact.png")

    # Save evaluation metrics
    metrics_json = {k: float(v) for k, v in metrics.items()}
    with open(f"{output_dir}evaluation_metrics.json", 'w') as f:
        json.dump(metrics_json, f, indent=2)

    print(f"\n[OK] All evaluation results saved to {output_dir}")
    print("\nGenerated files:")
    print(f"  - feature_importance.png")
    print(f"  - predictions_vs_actual.png")
    print(f"  - error_distribution.png")
    print(f"  - vulnerability_heatmap.png")
    print(f"  - revenue_impact.png")
    print(f"  - vulnerability_index_1_5C.csv")
    print(f"  - vulnerability_index_2_0C.csv")
    print(f"  - evaluation_metrics.json")


if __name__ == "__main__":
    print("Evaluation module loaded. Run via main.py or import in your pipeline.")
