"""
Data Generation Module
Synthetic dataset creation for Economic Risk Mapping project
Generates realistic climate, geospatial, and economic data for Kashmir Valley
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple


class SyntheticDataGenerator:
    """Generate realistic synthetic datasets for horticulture economic risk modeling."""

    def __init__(self, random_seed: int = 42, n_years: int = 12):
        """
        Initialize the data generator.

        Args:
            random_seed: For reproducibility
            n_years: Number of years to generate (default 12 years of data)
        """
        np.random.seed(random_seed)
        self.random_seed = random_seed
        self.n_years = n_years
        self.districts = ["Shopian", "Baramulla", "Pulwama"]
        self.crops = ["Apples", "Walnuts", "Saffron"]
        self.start_date = datetime(2012, 1, 1)
        self.end_date = datetime(2012 + n_years, 12, 31)

    def generate_climate_data(self) -> pd.DataFrame:
        """
        Generate daily climate data with realistic seasonal patterns.

        Returns:
            DataFrame with columns: Date, District, Avg_Temp, Max_Temp,
                                   Min_Temp, Precipitation_mm
        """
        print("Generating climate data...")
        data = []
        current_date = self.start_date

        while current_date <= self.end_date:
            day_of_year = current_date.timetuple().tm_yday

            # Seasonal temperature variation (winter: 0-5degC, summer: 25-35degC)
            base_temp = 15 + 15 * np.sin(2 * np.pi * day_of_year / 365)

            for district in self.districts:
                # District-specific micro-climate variation
                district_offset = {"Shopian": -2, "Baramulla": 0, "Pulwama": 1}[district]

                # Add realistic daily variation
                avg_temp = base_temp + district_offset + np.random.normal(0, 2)
                max_temp = avg_temp + np.random.uniform(3, 8)
                min_temp = avg_temp - np.random.uniform(3, 8)

                # Precipitation: higher in spring/fall, lower in summer/winter
                precip_base = 5 * np.sin(2 * np.pi * day_of_year / 365) + 3
                precipitation = max(0, precip_base + np.random.exponential(scale=2))

                data.append({
                    "Date": current_date,
                    "District": district,
                    "Avg_Temp": round(avg_temp, 2),
                    "Max_Temp": round(max_temp, 2),
                    "Min_Temp": round(min_temp, 2),
                    "Precipitation_mm": round(precipitation, 2)
                })

            current_date += timedelta(days=1)

        df = pd.DataFrame(data)
        print(f"  [OK] Generated {len(df):,} climate records ({df['Date'].min().date()} to {df['Date'].max().date()})")
        return df

    def generate_geospatial_data(self) -> pd.DataFrame:
        """
        Generate monthly geospatial data (NDVI, altitude, arable land).

        Returns:
            DataFrame with columns: Date, District, NDVI_score, Arable_Land_Altitude
        """
        print("Generating geospatial data...")
        data = []
        current_date = self.start_date

        while current_date <= self.end_date:
            month = current_date.month
            day_of_year = current_date.timetuple().tm_yday

            for district in self.districts:
                # NDVI: seasonal vegetation index (0-1), peaks in summer
                base_ndvi = 0.5 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
                ndvi = np.clip(base_ndvi + np.random.normal(0, 0.05), 0, 1)

                # Altitude ranges for Kashmir districts (meters)
                altitude_ranges = {
                    "Shopian": (1500, 2500),
                    "Baramulla": (1200, 2200),
                    "Pulwama": (1400, 2300)
                }
                altitude = np.random.uniform(*altitude_ranges[district])

                data.append({
                    "Date": current_date,
                    "District": district,
                    "NDVI_score": round(ndvi, 3),
                    "Arable_Land_Altitude": round(altitude, 1)
                })

            # Move to next month
            if month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=month + 1)

        df = pd.DataFrame(data)
        print(f"  [OK] Generated {len(df):,} geospatial records")
        return df

    def generate_economic_data(self) -> pd.DataFrame:
        """
        Generate yearly economic data (yield, price, revenue per hectare).

        Returns:
            DataFrame with columns: Year, District, Crop_Type, Yield_per_Hectare,
                                   Farm_Gate_Price, Revenue_per_Hectare
        """
        print("Generating economic data...")
        data = []

        for year in range(self.start_date.year, self.end_date.year + 1):
            for district in self.districts:
                for crop in self.crops:
                    # Realistic yield ranges (metric tons per hectare)
                    yield_ranges = {
                        "Apples": (20, 40),
                        "Walnuts": (2, 4),
                        "Saffron": (0.005, 0.015)
                    }
                    yield_per_ha = np.random.uniform(*yield_ranges[crop])

                    # Farm-gate prices (per kg, in INR)
                    price_ranges = {
                        "Apples": (20, 50),
                        "Walnuts": (200, 400),
                        "Saffron": (2000, 5000)
                    }
                    farm_gate_price = np.random.uniform(*price_ranges[crop])

                    # Convert yield to kg and calculate revenue
                    yield_kg = yield_per_ha * 1000
                    revenue = yield_kg * farm_gate_price

                    # Add climate impact (higher temp anomalies reduce yield)
                    climate_impact = np.random.uniform(-0.2, 0.05)  # -20% to +5%
                    revenue *= (1 + climate_impact)

                    data.append({
                        "Year": year,
                        "District": district,
                        "Crop_Type": crop,
                        "Yield_per_Hectare": round(yield_per_ha, 3),
                        "Farm_Gate_Price": round(farm_gate_price, 2),
                        "Revenue_per_Hectare": round(max(0, revenue), 2)
                    })

        df = pd.DataFrame(data)
        print(f"  [OK] Generated {len(df):,} economic records ({len(df)//len(self.crops)//len(self.districts)} years)")
        return df

    def save_datasets(self, output_dir: str = "data/synthetic/") -> None:
        """
        Generate and save all synthetic datasets to CSV.

        Args:
            output_dir: Directory to save datasets
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*60)
        print("SYNTHETIC DATA GENERATION")
        print("="*60 + "\n")

        # Generate and save datasets
        climate_df = self.generate_climate_data()
        climate_df.to_csv(f"{output_dir}climate_data.csv", index=False)

        geo_df = self.generate_geospatial_data()
        geo_df.to_csv(f"{output_dir}geospatial_data.csv", index=False)

        econ_df = self.generate_economic_data()
        econ_df.to_csv(f"{output_dir}economic_data.csv", index=False)

        print(f"\n[OK] All datasets saved to {output_dir}")
        print(f"  - climate_data.csv: {len(climate_df):,} records")
        print(f"  - geospatial_data.csv: {len(geo_df):,} records")
        print(f"  - economic_data.csv: {len(econ_df):,} records")

        return climate_df, geo_df, econ_df


if __name__ == "__main__":
    generator = SyntheticDataGenerator(n_years=12)
    generator.save_datasets()
