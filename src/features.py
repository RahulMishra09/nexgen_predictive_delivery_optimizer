"""
Feature Engineering Module
Creates predictive features from raw logistics data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


class FeatureEngineer:
    """Build ML-ready features from raw logistics data."""

    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        self.datasets = datasets
        self.feature_df = None

    def build_base_features(self) -> pd.DataFrame:
        """Create base feature set from orders data."""
        if 'orders' not in self.datasets:
            raise ValueError("orders dataset is required")

        df = self.datasets['orders'].copy()

        # Date features
        if 'ship_date' in df.columns:
            df['ship_date'] = pd.to_datetime(df['ship_date'], errors='coerce')
            df['ship_day_of_week'] = df['ship_date'].dt.dayofweek
            df['ship_month'] = df['ship_date'].dt.month
            df['ship_day'] = df['ship_date'].dt.day
            df['is_weekend'] = (df['ship_day_of_week'] >= 5).astype(int)

            # Season
            df['season'] = df['ship_month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })

        # Promised delivery window
        if 'promised_date' in df.columns and 'ship_date' in df.columns:
            df['promised_date'] = pd.to_datetime(df['promised_date'], errors='coerce')
            df['promised_window_days'] = (df['promised_date'] - df['ship_date']).dt.days
            df['promised_window_days'] = df['promised_window_days'].clip(lower=0)

        # Distance features
        if 'distance_km' in df.columns:
            df['distance_km'] = pd.to_numeric(df['distance_km'], errors='coerce').fillna(0)
            df['distance_bin'] = pd.cut(df['distance_km'],
                                        bins=[0, 50, 200, 500, 1000, 10000],
                                        labels=['Local', 'Regional', 'Inter-State', 'Long-Haul', 'Express-Air'])

        # Priority encoding
        if 'priority' in df.columns:
            priority_map = {'Express': 3, 'Standard': 2, 'Economy': 1}
            df['priority_code'] = df['priority'].map(priority_map).fillna(2)

        return df

    def add_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich with customer segment and behavior data."""
        if 'customers' not in self.datasets:
            return df

        customers = self.datasets['customers'].copy()

        # Merge customer info
        if 'customer_id' in df.columns and 'customer_id' in customers.columns:
            # Keep only useful columns
            cols_to_merge = ['customer_id']
            if 'segment' in customers.columns:
                cols_to_merge.append('segment')
            if 'region' in customers.columns:
                cols_to_merge.append('region')
            if 'lifetime_value' in customers.columns:
                cols_to_merge.append('lifetime_value')

            df = df.merge(customers[cols_to_merge], on='customer_id', how='left')

            # Segment encoding
            if 'segment' in df.columns:
                segment_map = {'Premium': 3, 'Gold': 2, 'Standard': 1, 'Economy': 0}
                df['segment_code'] = df['segment'].map(segment_map).fillna(1)

        return df

    def add_warehouse_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich with warehouse capacity and load data."""
        if 'warehouses' not in self.datasets:
            return df

        warehouses = self.datasets['warehouses'].copy()

        if 'warehouse_id' in df.columns and 'warehouse_id' in warehouses.columns:
            cols_to_merge = ['warehouse_id']
            if 'utilization_pct' in warehouses.columns:
                cols_to_merge.append('utilization_pct')
            if 'city' in warehouses.columns:
                cols_to_merge.append('city')
            if 'capacity' in warehouses.columns:
                cols_to_merge.append('capacity')

            df = df.merge(warehouses[cols_to_merge], on='warehouse_id', how='left', suffixes=('', '_wh'))

            # High load indicator
            if 'utilization_pct' in df.columns:
                df['warehouse_high_load'] = (df['utilization_pct'] > 80).astype(int)

        return df

    def add_carrier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich with carrier performance metrics."""
        if 'carriers' not in self.datasets:
            return df

        carriers = self.datasets['carriers'].copy()

        if 'carrier_id' in df.columns and 'carrier_id' in carriers.columns:
            cols_to_merge = ['carrier_id']
            if 'on_time_pct' in carriers.columns:
                cols_to_merge.append('on_time_pct')
            if 'coverage' in carriers.columns:
                cols_to_merge.append('coverage')
            if 'emissions_g_per_km' in carriers.columns:
                cols_to_merge.append('emissions_g_per_km')

            df = df.merge(carriers[cols_to_merge], on='carrier_id', how='left')

            # Carrier reliability flag
            if 'on_time_pct' in df.columns:
                df['carrier_reliable'] = (df['on_time_pct'] >= 90).astype(int)

        return df

    def add_fleet_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich with vehicle/fleet data."""
        if 'fleet' not in self.datasets:
            return df

        fleet = self.datasets['fleet'].copy()

        if 'vehicle_id' in df.columns and 'vehicle_id' in fleet.columns:
            cols_to_merge = ['vehicle_id']
            if 'type' in fleet.columns:
                cols_to_merge.append('type')
            if 'avg_speed_kmph' in fleet.columns:
                cols_to_merge.append('avg_speed_kmph')
            if 'refrigeration' in fleet.columns:
                cols_to_merge.append('refrigeration')

            df = df.merge(fleet[cols_to_merge], on='vehicle_id', how='left', suffixes=('', '_fleet'))

            # Calculate estimated travel time
            if 'distance_km' in df.columns and 'avg_speed_kmph' in df.columns:
                df['est_travel_hours'] = (df['distance_km'] / df['avg_speed_kmph'].replace(0, 50)).fillna(0)

        return df

    def add_tracking_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich with tracking event aggregations."""
        if 'tracking' not in self.datasets:
            return df

        tracking = self.datasets['tracking'].copy()

        if 'order_id' in tracking.columns:
            # Aggregate tracking events per order
            tracking_agg = tracking.groupby('order_id').agg({
                'delay_minutes': ['sum', 'max', 'mean'],
                'status': 'count'
            }).reset_index()

            tracking_agg.columns = ['order_id', 'total_delay_min', 'max_delay_min', 'avg_delay_min', 'scan_count']

            df = df.merge(tracking_agg, on='order_id', how='left')

            # Fill missing with 0
            for col in ['total_delay_min', 'max_delay_min', 'avg_delay_min', 'scan_count']:
                if col in df.columns:
                    df[col] = df[col].fillna(0)

        return df

    def add_cost_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich with cost information."""
        if 'costs' not in self.datasets:
            return df

        costs = self.datasets['costs'].copy()

        if 'order_id' in df.columns and 'order_id' in costs.columns:
            cols_to_merge = ['order_id']
            if 'total_cost' in costs.columns:
                cols_to_merge.append('total_cost')
            if 'linehaul_cost' in costs.columns:
                cols_to_merge.append('linehaul_cost')
            if 'last_mile_cost' in costs.columns:
                cols_to_merge.append('last_mile_cost')

            df = df.merge(costs[cols_to_merge], on='order_id', how='left')

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables."""
        # Distance × Priority
        if 'distance_km' in df.columns and 'priority_code' in df.columns:
            df['distance_priority_interaction'] = df['distance_km'] * df['priority_code']

        # Warehouse load × Weekend
        if 'utilization_pct' in df.columns and 'is_weekend' in df.columns:
            df['warehouse_weekend_interaction'] = df['utilization_pct'] * df['is_weekend']

        # Carrier reliability × Distance
        if 'on_time_pct' in df.columns and 'distance_km' in df.columns:
            df['carrier_distance_risk'] = (100 - df['on_time_pct']) * df['distance_km'] / 100

        return df

    def build_all_features(self) -> pd.DataFrame:
        """Execute full feature pipeline."""
        print("Building base features...")
        df = self.build_base_features()

        print("Adding customer features...")
        df = self.add_customer_features(df)

        print("Adding warehouse features...")
        df = self.add_warehouse_features(df)

        print("Adding carrier features...")
        df = self.add_carrier_features(df)

        print("Adding fleet features...")
        df = self.add_fleet_features(df)

        print("Adding tracking features...")
        df = self.add_tracking_features(df)

        print("Adding cost features...")
        df = self.add_cost_features(df)

        print("Creating interaction features...")
        df = self.create_interaction_features(df)

        self.feature_df = df
        print(f"\n✓ Feature engineering complete: {len(df)} rows, {len(df.columns)} features")

        return df

    def get_feature_columns(self) -> List[str]:
        """Return list of engineered feature columns for modeling."""
        if self.feature_df is None:
            raise ValueError("Features not built yet. Call build_all_features() first.")

        # Exclude ID columns and target variables
        exclude = ['order_id', 'customer_id', 'warehouse_id', 'carrier_id', 'vehicle_id',
                   'is_delayed', 'delay_hours', 'actual_delivery', 'promised_date', 'ship_date']

        feature_cols = [col for col in self.feature_df.columns if col not in exclude]

        # Keep only numeric and categorical encoded features
        numeric_cols = self.feature_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        return numeric_cols


def engineer_features(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Main entry point: build all features.

    Returns:
        DataFrame with engineered features
    """
    engineer = FeatureEngineer(datasets)
    df = engineer.build_all_features()

    return df


if __name__ == "__main__":
    from data import load_and_prepare_data

    # Test feature engineering
    datasets = load_and_prepare_data()
    features_df = engineer_features(datasets)

    print("\n" + "="*50)
    print("Feature Columns:")
    print("="*50)
    engineer = FeatureEngineer(datasets)
    engineer.feature_df = features_df
    print(engineer.get_feature_columns())
