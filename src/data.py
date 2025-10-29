"""
Data Loading Module
Handles CSV loading, schema mapping, and validation for NexGen Logistics data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class DataLoader:
    """Load and validate all CSV files with flexible schema mapping."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.datasets = {}

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all available CSVs and return as dictionary."""
        csv_files = {
            'orders': 'orders.csv',
            'customers': 'customers.csv',
            'warehouses': 'warehouses.csv',
            'fleet': 'fleet.csv',
            'tracking': 'tracking.csv',
            'costs': 'costs.csv',
            'carriers': 'carriers.csv'
        }

        for name, filename in csv_files.items():
            file_path = self.data_dir / filename
            if file_path.exists():
                try:
                    self.datasets[name] = pd.read_csv(file_path)
                    print(f"✓ Loaded {name}: {len(self.datasets[name])} rows")
                except Exception as e:
                    print(f"✗ Error loading {name}: {str(e)}")
            else:
                print(f"⚠ File not found: {filename}")

        return self.datasets

    def validate_orders(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean orders data."""
        required_cols = ['order_id']

        # Check for required columns
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in orders: {missing}")

        # Convert dates
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass

        # Create delay flag if delivery dates exist
        if 'promised_date' in df.columns and 'actual_delivery' in df.columns:
            df['is_delayed'] = (df['actual_delivery'] > df['promised_date']).astype(int)
            df['delay_hours'] = (df['actual_delivery'] - df['promised_date']).dt.total_seconds() / 3600
            df['delay_hours'] = df['delay_hours'].fillna(0).clip(lower=0)

        return df

    def validate_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean customer data."""
        if 'customer_id' not in df.columns:
            raise ValueError("customers.csv must have 'customer_id' column")

        # Fill missing segments
        if 'segment' in df.columns:
            df['segment'] = df['segment'].fillna('Standard')

        return df

    def validate_warehouses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean warehouse data."""
        if 'warehouse_id' not in df.columns:
            raise ValueError("warehouses.csv must have 'warehouse_id' column")

        # Calculate warehouse utilization if capacity info exists
        if 'capacity' in df.columns and 'current_load' in df.columns:
            df['utilization_pct'] = (df['current_load'] / df['capacity'] * 100).fillna(0)

        return df

    def validate_carriers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean carrier data."""
        if 'carrier_id' not in df.columns:
            raise ValueError("carriers.csv must have 'carrier_id' column")

        # Ensure on-time percentage is valid
        if 'on_time_pct' in df.columns:
            df['on_time_pct'] = df['on_time_pct'].clip(0, 100)

        return df

    def validate_fleet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean fleet data."""
        if 'vehicle_id' not in df.columns:
            raise ValueError("fleet.csv must have 'vehicle_id' column")

        return df

    def validate_tracking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean tracking data."""
        if 'order_id' not in df.columns:
            raise ValueError("tracking.csv must have 'order_id' column")

        # Convert scan times
        if 'scan_time' in df.columns:
            df['scan_time'] = pd.to_datetime(df['scan_time'], errors='coerce')

        return df

    def validate_costs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean cost data."""
        if 'order_id' not in df.columns:
            raise ValueError("costs.csv must have 'order_id' column")

        # Calculate total cost if components exist
        cost_cols = [col for col in df.columns if 'cost' in col.lower()]
        if len(cost_cols) > 1:
            df['total_cost'] = df[cost_cols].sum(axis=1)

        return df

    def get_data_summary(self) -> pd.DataFrame:
        """Return summary statistics of loaded datasets."""
        summary = []
        for name, df in self.datasets.items():
            summary.append({
                'Dataset': name,
                'Rows': len(df),
                'Columns': len(df.columns),
                'Memory (MB)': df.memory_usage(deep=True).sum() / 1024**2
            })

        return pd.DataFrame(summary)

    def validate_all(self) -> Dict[str, pd.DataFrame]:
        """Run validation on all loaded datasets."""
        validators = {
            'orders': self.validate_orders,
            'customers': self.validate_customers,
            'warehouses': self.validate_warehouses,
            'carriers': self.validate_carriers,
            'fleet': self.validate_fleet,
            'tracking': self.validate_tracking,
            'costs': self.validate_costs
        }

        for name, df in self.datasets.items():
            if name in validators:
                try:
                    self.datasets[name] = validators[name](df)
                    print(f"✓ Validated {name}")
                except Exception as e:
                    print(f"✗ Validation error in {name}: {str(e)}")

        return self.datasets


def load_and_prepare_data(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """
    Main entry point: load and validate all data.

    Returns:
        Dictionary of cleaned DataFrames
    """
    loader = DataLoader(data_dir)
    loader.load_all()
    loader.validate_all()

    print("\n" + "="*50)
    print("Data Summary:")
    print("="*50)
    print(loader.get_data_summary().to_string(index=False))

    return loader.datasets


if __name__ == "__main__":
    # Test the data loader
    datasets = load_and_prepare_data()

    if 'orders' in datasets:
        print("\n" + "="*50)
        print("Sample Orders Data:")
        print("="*50)
        print(datasets['orders'].head())
