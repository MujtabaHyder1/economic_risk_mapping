"""
Data Preprocessing and Feature Engineering Module
Merges climate, geospatial, and economic data with domain-specific feature engineering
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Preprocess and engineer features for horticulture economic risk modeling."""

    def __init__(self, climate_df: pd.DataFrame, geo_df: pd.DataFrame, econ_df: pd.DataFrame):
        """
        Initialize preprocessor with raw datasets.

        Args:
            climate_df: Daily climate data
            geo_df: Monthly geospatial data
            econ_df: Yearly economic data
        """
        self.climate_df = climate_df.copy()
        self.geo_df = geo_df.copy()
        self.econ_df = econ_df.copy()
        self.districts = ["Shopian", "Baramulla", "Pulwama"]
        self.crops = ["Apples", "Walnuts", "Saffron"]

    def _prepare_climate_features(self) -> pd.DataFrame:
        """
        Aggregate daily climate data into monthly/seasonal features.

        Returns:
            DataFrame with Date (month start), District, and climate aggregates
        """
        print("Aggregating climate features...")

        self.climate_df['Date'] = pd.to_datetime(self.climate_df['Date'])
        self.climate_df['Year'] = self.climate_df['Date'].dt.year
        self.climate_df['Month'] = self.climate_df['Date'].dt.month
        self.climate_df['YearMonth'] = self.climate_df['Date'].dt.to_period('M')

        # Monthly aggregates
        monthly_climate = self.climate_df.groupby(['YearMonth', 'District']).agg({
            'Avg_Temp': ['mean', 'std'],
            'Max_Temp': 'max',
            'Min_Temp': 'min',
            'Precipitation_mm': 'sum'
        }).reset_index()

        # Flatten multi-level columns
        monthly_climate.columns = [
            'YearMonth', 'District', 'Avg_Temp_Mean', 'Avg_Temp_Std',
            'Max_Temp_Monthly', 'Min_Temp_Monthly', 'Precip_Monthly'
        ]

        monthly_climate['Date'] = monthly_climate['YearMonth'].dt.to_timestamp()
        monthly_climate['Year'] = monthly_climate['Date'].dt.year
        monthly_climate['Month'] = monthly_climate['Date'].dt.month

        return monthly_climate

    def _engineer_climate_anomalies(self, climate_monthly: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer domain-specific climate anomaly features.

        Critical for horticulture:
        - Frost Days in March: Critical blooming period, frost damages flowers
        - Pre-Harvest Heatwave: Excessive heat before harvest degrades crop quality
        - Temperature Variability: Unpredictable swings stress plants

        Args:
            climate_monthly: Monthly aggregated climate data

        Returns:
            DataFrame with climate anomaly features
        """
        print("Engineering climate anomaly features...")

        # 1. Frost Days in March (critical blooming month for apples/walnuts)
        frost_data = []
        for (year, district), group in self.climate_df.groupby(
            [self.climate_df['Date'].dt.year, 'District']
        ):
            march_data = group[group['Date'].dt.month == 3]
            frost_days = (march_data['Min_Temp'] < 0).sum()
            frost_data.append({
                'Year': year,
                'District': district,
                'Frost_Days_March': frost_days
            })

        frost_df = pd.DataFrame(frost_data)

        # 2. Pre-Harvest Heatwave (July-August for most crops)
        heatwave_data = []
        for (year, district), group in self.climate_df.groupby(
            [self.climate_df['Date'].dt.year, 'District']
        ):
            harvest_period = group[group['Date'].dt.month.isin([7, 8])]
            if len(harvest_period) > 0:
                # Consecutive days > 30degC
                hot_days = (harvest_period['Max_Temp'] > 30).sum()
                heatwave_data.append({
                    'Year': year,
                    'District': district,
                    'Hot_Days_JulAug': hot_days,
                    'Max_Temp_PreHarvest': harvest_period['Max_Temp'].max()
                })

        heatwave_df = pd.DataFrame(heatwave_data)

        # 3. Growing Season Precipitation (April-October)
        precip_data = []
        for (year, district), group in self.climate_df.groupby(
            [self.climate_df['Date'].dt.year, 'District']
        ):
            growing_season = group[group['Date'].dt.month.isin(range(4, 11))]
            total_precip = growing_season['Precipitation_mm'].sum()
            precip_data.append({
                'Year': year,
                'District': district,
                'Precip_Growing_Season': total_precip
            })

        precip_df = pd.DataFrame(precip_data)

        # Merge anomaly features
        anomalies = frost_df.merge(heatwave_df, on=['Year', 'District'])
        anomalies = anomalies.merge(precip_df, on=['Year', 'District'])

        return anomalies

    def _prepare_geospatial_features(self) -> pd.DataFrame:
        """
        Aggregate monthly geospatial data into yearly features.

        Returns:
            DataFrame with Year, District, and geospatial aggregates
        """
        print("Aggregating geospatial features...")

        self.geo_df['Date'] = pd.to_datetime(self.geo_df['Date'])
        self.geo_df['Year'] = self.geo_df['Date'].dt.year

        yearly_geo = self.geo_df.groupby(['Year', 'District']).agg({
            'NDVI_score': ['mean', 'std', 'min'],
            'Arable_Land_Altitude': 'mean'
        }).reset_index()

        # Flatten multi-level columns
        yearly_geo.columns = [
            'Year', 'District', 'NDVI_Mean', 'NDVI_Std', 'NDVI_Min', 'Altitude_Mean'
        ]

        return yearly_geo

    def _engineer_ndvi_deviation(self, geo_yearly: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate NDVI deviation from 10-year rolling average.

        NDVI deviation indicates vegetation stress due to climate anomalies.

        Args:
            geo_yearly: Yearly aggregated geospatial data

        Returns:
            DataFrame with NDVI deviation feature
        """
        print("Engineering NDVI deviation feature...")

        ndvi_features = []
        for district in self.districts:
            district_data = geo_yearly[geo_yearly['District'] == district].sort_values('Year')

            # 10-year rolling average (using available data, minimum 3 years)
            district_data['NDVI_Rolling_Avg'] = district_data['NDVI_Mean'].rolling(
                window=10, min_periods=3, center=False
            ).mean()

            # NDVI Deviation: current year vs rolling average
            district_data['NDVI_Deviation'] = (
                district_data['NDVI_Mean'] - district_data['NDVI_Rolling_Avg']
            ).fillna(0)

            ndvi_features.append(district_data)

        return pd.concat(ndvi_features, ignore_index=True)

    def merge_and_aggregate(self) -> pd.DataFrame:
        """
        Merge all datasets and aggregate to yearly level for modeling.

        Returns:
            Merged DataFrame ready for model training
        """
        print("\n" + "="*60)
        print("DATA PREPROCESSING & FEATURE ENGINEERING")
        print("="*60 + "\n")

        # Prepare base datasets
        climate_monthly = self._prepare_climate_features()
        climate_anomalies = self._engineer_climate_anomalies(climate_monthly)

        geo_yearly = self._prepare_geospatial_features()
        geo_with_ndvi = self._engineer_ndvi_deviation(geo_yearly)

        # Aggregate climate to yearly
        yearly_climate = climate_monthly.groupby(['Year', 'District']).agg({
            'Avg_Temp_Mean': 'mean',
            'Avg_Temp_Std': 'mean',
            'Max_Temp_Monthly': 'max',
            'Min_Temp_Monthly': 'min',
            'Precip_Monthly': 'sum'
        }).reset_index()

        yearly_climate.columns = [
            'Year', 'District', 'Yearly_Avg_Temp', 'Yearly_Temp_Variability',
            'Yearly_Max_Temp', 'Yearly_Min_Temp', 'Yearly_Precip'
        ]

        # Merge all features
        merged = self.econ_df.merge(yearly_climate, on=['Year', 'District'], how='left')
        merged = merged.merge(climate_anomalies, on=['Year', 'District'], how='left')
        merged = merged.merge(geo_with_ndvi, on=['Year', 'District'], how='left')

        print(f"[OK] Merged dataset shape: {merged.shape}")
        print(f"  Rows: {len(merged):,} | Columns: {len(merged.columns)}")

        return merged

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing values appropriately.

        Args:
            df: DataFrame with potential missing values
            strategy: 'forward_fill', 'backward_fill', or 'mean'

        Returns:
            DataFrame with missing values handled
        """
        print(f"\nHandling missing values using '{strategy}' strategy...")
        initial_nulls = df.isnull().sum().sum()

        if strategy == 'forward_fill':
            df = df.sort_values(['District', 'Year']).ffill().bfill()
        elif strategy == 'backward_fill':
            df = df.sort_values(['District', 'Year']).bfill().ffill()
        elif strategy == 'mean':
            df = df.fillna(df.mean(numeric_only=True))

        final_nulls = df.isnull().sum().sum()
        print(f"  Null values: {initial_nulls} -> {final_nulls}")

        return df

    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers using IQR or Z-score method.

        Args:
            df: DataFrame with potential outliers
            method: 'iqr' or 'zscore'
            threshold: Z-score threshold (default 3.0 = ~0.3% outliers)

        Returns:
            DataFrame with outliers removed
        """
        print(f"\nRemoving outliers using '{method}' method...")
        initial_rows = len(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if method == 'iqr':
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[numeric_cols], nan_policy='omit'))
            df = df[(z_scores < threshold).all(axis=1)]

        final_rows = len(df)
        print(f"  Rows: {initial_rows:,} -> {final_rows:,} ({initial_rows-final_rows} removed)")

        return df

    def get_feature_summary(self, df: pd.DataFrame) -> None:
        """Print summary of engineered features."""
        print("\n" + "-"*60)
        print("ENGINEERED FEATURES SUMMARY")
        print("-"*60)
        print("\nClimate Anomalies:")
        print("  - Frost_Days_March: Days below 0degC during critical blooming")
        print("  - Hot_Days_JulAug: Consecutive days > 30degC before harvest")
        print("  - Max_Temp_PreHarvest: Peak temperature during harvest season")
        print("  - Precip_Growing_Season: Total rainfall during growing season (Apr-Oct)")
        print("\nGeospatial Features:")
        print("  - NDVI_Mean: Average vegetation index (0-1 scale)")
        print("  - NDVI_Deviation: Deviation from 10-year rolling average")
        print("  - Altitude_Mean: Mean elevation of arable land")
        print("\nAnnual Climate Metrics:")
        print("  - Yearly_Avg_Temp: Annual mean temperature")
        print("  - Yearly_Temp_Variability: Temperature std deviation")
        print("  - Yearly_Max_Temp & Yearly_Min_Temp: Extremes")
        print("  - Yearly_Precip: Annual precipitation total")
        print("\nEconomic Target:")
        print("  - Revenue_per_Hectare: Target variable for prediction")
        print("-"*60)


def preprocess_data(
    climate_path: str,
    geo_path: str,
    econ_path: str,
    output_path: str = "data/processed/merged_features.csv"
) -> pd.DataFrame:
    """
    End-to-end preprocessing pipeline.

    Args:
        climate_path: Path to climate data CSV
        geo_path: Path to geospatial data CSV
        econ_path: Path to economic data CSV
        output_path: Path to save merged dataset

    Returns:
        Processed DataFrame
    """
    import os

    # Load raw datasets
    print("Loading raw datasets...")
    climate_df = pd.read_csv(climate_path)
    geo_df = pd.read_csv(geo_path)
    econ_df = pd.read_csv(econ_path)

    print(f"  - Climate: {len(climate_df):,} records")
    print(f"  - Geospatial: {len(geo_df):,} records")
    print(f"  - Economic: {len(econ_df):,} records\n")

    # Preprocess and engineer features
    preprocessor = DataPreprocessor(climate_df, geo_df, econ_df)
    merged_df = preprocessor.merge_and_aggregate()

    # Handle missing values
    merged_df = preprocessor.handle_missing_values(merged_df, strategy='forward_fill')

    # Remove outliers
    merged_df = preprocessor.remove_outliers(merged_df, method='iqr')

    # Print feature summary
    preprocessor.get_feature_summary(merged_df)

    # Save processed dataset
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"\n[OK] Processed dataset saved: {output_path}")
    print(f"  Final shape: {merged_df.shape}")

    return merged_df


if __name__ == "__main__":
    # Example usage
    df = preprocess_data(
        "data/synthetic/climate_data.csv",
        "data/synthetic/geospatial_data.csv",
        "data/synthetic/economic_data.csv"
    )
    print("\n" + df.head().to_string())
