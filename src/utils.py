"""
Utility Functions Module
Common utilities for the entire pipeline
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import os
import logging


def create_project_directories(base_dir: str = ".") -> None:
    """
    Create project directory structure.

    Args:
        base_dir: Base directory for the project
    """
    dirs = [
        "data/raw",
        "data/synthetic",
        "data/processed",
        "models",
        "results",
        "notebooks",
        "logs"
    ]

    for dir_path in dirs:
        full_path = os.path.join(base_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)

    print("[OK] Project directory structure created")


def setup_logging(log_file: str = "logs/pipeline.log", level: str = "INFO") -> logging.Logger:
    """
    Configure file + console logging for the pipeline.

    Args:
        log_file: Path to the log file
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')

    Returns:
        Configured logger instance
    """
    # Ensure the log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("economic_risk_mapping")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()  # Avoid duplicate handlers on re-import

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized (level={level}, file={log_file})")
    return logger


def print_project_banner(title: str, width: int = 70) -> None:
    """
    Print formatted project banner.

    Args:
        title: Title text
        width: Banner width
    """
    print("\n" + "="*width)
    print(f"  {title.center(width-4)}")
    print("="*width + "\n")


def print_section(title: str, width: int = 60) -> None:
    """
    Print section header.

    Args:
        title: Section title
        width: Header width
    """
    print("\n" + "-"*width)
    print(f"  {title}")
    print("-"*width)


def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """
    Validate that dataframe contains all required columns.

    Args:
        df: DataFrame to validate
        required_cols: List of required column names

    Returns:
        True if all columns present, False otherwise
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"[FAIL] Missing columns: {missing}")
        return False
    return True


def get_data_summary(df: pd.DataFrame) -> None:
    """
    Print summary statistics for a dataframe.

    Args:
        df: DataFrame to summarize
    """
    print(f"\nDataFrame Summary:")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}"
          if 'Date' in df.columns else "")
    print(f"  Districts: {', '.join(df['District'].unique())}"
          if 'District' in df.columns else "")
    print(f"  Null values: {df.isnull().sum().sum():,}")


def log_pipeline_step(step_name: str, status: str = "STARTED", duration: float = None) -> None:
    """
    Log pipeline execution step.

    Args:
        step_name: Name of the pipeline step
        status: Status ('STARTED', 'COMPLETED', 'FAILED')
        duration: Execution time in seconds (optional)
    """
    duration_str = f" ({duration:.2f}s)" if duration else ""
    message = f"{status:10s} | {step_name}{duration_str}"

    # Write to the file logger if it has been configured
    logger = logging.getLogger("economic_risk_mapping")
    if logger.handlers:
        level_map = {
            "STARTED": logging.INFO,
            "COMPLETED": logging.INFO,
            "FAILED": logging.ERROR,
        }
        logger.log(level_map.get(status.upper(), logging.INFO), message)
    else:
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")


def compare_model_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    district_col: pd.Series = None
) -> pd.DataFrame:
    """
    Create comparison dataframe for predictions.

    Args:
        y_true: Actual target values
        y_pred: Predicted values
        district_col: Optional district information

    Returns:
        Comparison dataframe
    """
    comparison = pd.DataFrame({
        'Actual': y_true.values,
        'Predicted': y_pred,
        'Error': y_true.values - y_pred,
        'Error_Pct': np.abs(y_true.values - y_pred) / y_true.values * 100
    })

    if district_col is not None:
        comparison['District'] = district_col.values

    return comparison.sort_values('Error_Pct', ascending=False)


def generate_summary_report(metrics: dict, vuln_df: pd.DataFrame, output_file: str) -> None:
    """
    Generate text summary report.

    Args:
        metrics: Model evaluation metrics dictionary
        vuln_df: Vulnerability index dataframe
        output_file: Path to save report
    """
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ECONOMIC RISK MAPPING - SUMMARY REPORT\n")
        f.write("Kashmir Valley Horticulture (Shopian, Baramulla, Pulwama)\n")
        f.write("="*70 + "\n\n")

        f.write("MODEL PERFORMANCE\n")
        f.write("-"*70 + "\n")
        f.write(f"Test RMSE:        {metrics.get('test_rmse', 'N/A'):.4f}\n")
        f.write(f"Test MAE:         {metrics.get('test_mae', 'N/A'):.4f}\n")
        f.write(f"Test R²:          {metrics.get('test_r2', 'N/A'):.4f}\n")
        f.write(f"Test MAPE:        {metrics.get('test_mape', 'N/A'):.2f}%\n\n")

        f.write("FINANCIAL VULNERABILITY INDEX (1.5degC Scenario)\n")
        f.write("-"*70 + "\n")

        for district in vuln_df['District'].unique():
            district_data = vuln_df[vuln_df['District'] == district]
            avg_vuln = district_data['Vulnerability_Index'].mean()
            avg_loss = district_data['Revenue_Loss_Percentage'].mean()
            f.write(f"\n{district}:\n")
            f.write(f"  Vulnerability Index: {avg_vuln:.1f}/100\n")
            f.write(f"  Avg Revenue Loss:    {avg_loss:.1f}%\n")

            for _, row in district_data.iterrows():
                f.write(f"    - {row['Crop_Type']:12s}: {row['Vulnerability_Index']:5.1f} " +
                       f"({row['Revenue_Loss_Percentage']:5.1f}% loss)\n")

    print(f"[OK] Summary report saved: {output_file}")


if __name__ == "__main__":
    print("Utility module loaded. Import functions for use in pipeline.")
