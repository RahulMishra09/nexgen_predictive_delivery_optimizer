"""
Prescription Rules Engine
Translates delay risk predictions into actionable recommendations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


class PrescriptionEngine:
    """Generate actionable prescriptions based on delay risk."""

    def __init__(self, datasets: Dict[str, pd.DataFrame] = None):
        self.datasets = datasets or {}
        self.actions = []

    def prescribe_carrier_swap(self, row: pd.Series) -> Dict:
        """Recommend alternative carrier if current one is unreliable."""
        if row.get('carrier_reliable', 1) == 0 and row.get('on_time_pct', 100) < 85:
            # Find better carrier from carriers dataset
            better_carrier = None
            if 'carriers' in self.datasets:
                carriers = self.datasets['carriers']
                better = carriers[carriers['on_time_pct'] >= 90].sort_values('on_time_pct', ascending=False)
                if len(better) > 0:
                    better_carrier = better.iloc[0]['carrier_id']

            return {
                'action': 'Carrier Swap',
                'priority': 'High',
                'details': f"Current carrier has {row.get('on_time_pct', 0):.1f}% on-time rate. "
                          f"Recommend switching to carrier {better_carrier} with better reliability.",
                'estimated_impact': '15-20% delay reduction',
                'cost_impact': 'Medium'
            }
        return None

    def prescribe_route_optimization(self, row: pd.Series) -> Dict:
        """Suggest route changes for long-distance or high-delay orders."""
        if row.get('distance_km', 0) > 500 and row.get('total_delay_min', 0) > 60:
            return {
                'action': 'Route Optimization',
                'priority': 'Medium',
                'details': f"Order traveling {row.get('distance_km', 0):.0f} km with accumulated delays. "
                          "Consider direct route or alternative hub.",
                'estimated_impact': '10-15% time reduction',
                'cost_impact': 'Low'
            }
        return None

    def prescribe_vehicle_reassignment(self, row: pd.Series) -> Dict:
        """Recommend faster vehicle for high-priority delayed orders."""
        if (row.get('priority_code', 1) >= 3 and
            row.get('avg_speed_kmph', 60) < 70 and
            row.get('delay_risk_score', 0) > 0.6):

            return {
                'action': 'Vehicle Reassignment',
                'priority': 'High',
                'details': f"Express order with high delay risk ({row.get('delay_risk_score', 0):.0%}). "
                          "Assign faster vehicle or air freight.",
                'estimated_impact': '30-40% time reduction',
                'cost_impact': 'High'
            }
        return None

    def prescribe_priority_bump(self, row: pd.Series) -> Dict:
        """Escalate priority for valuable customers or orders."""
        if (row.get('segment_code', 1) >= 2 and
            row.get('delay_risk_score', 0) > 0.5 and
            row.get('priority_code', 2) < 3):

            return {
                'action': 'Priority Upgrade',
                'priority': 'Medium',
                'details': f"Premium customer (segment {row.get('segment', 'Unknown')}) with delay risk. "
                          "Upgrade to Express handling.",
                'estimated_impact': '20-25% delay reduction',
                'cost_impact': 'Medium'
            }
        return None

    def prescribe_warehouse_reroute(self, row: pd.Series) -> Dict:
        """Suggest alternative warehouse if current one is overloaded."""
        if row.get('warehouse_high_load', 0) == 1 and row.get('utilization_pct', 0) > 85:
            return {
                'action': 'Warehouse Reroute',
                'priority': 'Medium',
                'details': f"Origin warehouse at {row.get('utilization_pct', 0):.0f}% capacity. "
                          "Consider fulfilling from nearby warehouse with lower load.",
                'estimated_impact': '12-18% delay reduction',
                'cost_impact': 'Low'
            }
        return None

    def prescribe_proactive_communication(self, row: pd.Series) -> Dict:
        """Recommend customer notification for high-risk orders."""
        if row.get('delay_risk_score', 0) > 0.7:
            return {
                'action': 'Proactive Customer Alert',
                'priority': 'Low',
                'details': f"Order has {row.get('delay_risk_score', 0):.0%} delay probability. "
                          "Send proactive notification with revised ETA.",
                'estimated_impact': 'Improved CSAT',
                'cost_impact': 'Minimal'
            }
        return None

    def prescribe_weekend_prep(self, row: pd.Series) -> Dict:
        """Special handling for weekend shipments."""
        if row.get('is_weekend', 0) == 1 and row.get('delay_risk_score', 0) > 0.5:
            return {
                'action': 'Weekend Surge Planning',
                'priority': 'Medium',
                'details': "Weekend shipment with delay risk. Ensure adequate staffing and carrier availability.",
                'estimated_impact': '10-15% delay reduction',
                'cost_impact': 'Low'
            }
        return None

    def generate_prescriptions(self, row: pd.Series) -> List[Dict]:
        """
        Generate all applicable prescriptions for a single order.

        Returns:
            List of prescription dictionaries
        """
        prescriptions = []

        # Run all prescription rules
        rules = [
            self.prescribe_carrier_swap,
            self.prescribe_vehicle_reassignment,
            self.prescribe_priority_bump,
            self.prescribe_route_optimization,
            self.prescribe_warehouse_reroute,
            self.prescribe_proactive_communication,
            self.prescribe_weekend_prep
        ]

        for rule in rules:
            result = rule(row)
            if result:
                prescriptions.append(result)

        return prescriptions

    def generate_action_plan(self, df: pd.DataFrame, min_risk: float = 0.5) -> pd.DataFrame:
        """
        Generate action plan for all at-risk orders.

        Args:
            df: DataFrame with risk scores and features
            min_risk: Minimum risk threshold to generate actions

        Returns:
            DataFrame with order_id, risk_score, and recommended actions
        """
        # Filter high-risk orders
        at_risk = df[df['delay_risk_score'] >= min_risk].copy()

        if len(at_risk) == 0:
            print(f"No orders found with risk >= {min_risk}")
            return pd.DataFrame()

        # Generate prescriptions for each order
        action_plan = []

        for idx, row in at_risk.iterrows():
            prescriptions = self.generate_prescriptions(row)

            if prescriptions:
                for presc in prescriptions:
                    action_plan.append({
                        'order_id': row.get('order_id', idx),
                        'risk_score': row.get('delay_risk_score', 0),
                        'risk_category': row.get('risk_category', 'Unknown'),
                        'priority': row.get('priority', 'Standard'),
                        'action': presc['action'],
                        'action_priority': presc['priority'],
                        'details': presc['details'],
                        'estimated_impact': presc['estimated_impact'],
                        'cost_impact': presc['cost_impact']
                    })

        action_df = pd.DataFrame(action_plan)

        if len(action_df) > 0:
            # Sort by risk score and action priority
            priority_order = {'High': 3, 'Medium': 2, 'Low': 1}
            action_df['priority_score'] = action_df['action_priority'].map(priority_order)
            action_df = action_df.sort_values(['risk_score', 'priority_score'], ascending=[False, False])
            action_df = action_df.drop('priority_score', axis=1)

        return action_df

    def summarize_actions(self, action_df: pd.DataFrame) -> Dict:
        """
        Summarize the action plan for executive reporting.

        Returns:
            Dictionary with summary statistics
        """
        if len(action_df) == 0:
            return {
                'total_at_risk_orders': 0,
                'total_actions': 0,
                'high_priority_actions': 0,
                'estimated_monthly_savings': 0
            }

        summary = {
            'total_at_risk_orders': action_df['order_id'].nunique(),
            'total_actions': len(action_df),
            'high_priority_actions': len(action_df[action_df['action_priority'] == 'High']),
            'actions_by_type': action_df['action'].value_counts().to_dict(),
            'avg_risk_score': action_df['risk_score'].mean(),
            'max_risk_score': action_df['risk_score'].max()
        }

        # Estimate cost savings (simplified calculation)
        # Assume avg delay cost = ₹500, avg recovery rate = 60%
        avg_delay_cost = 500
        recovery_rate = 0.6
        summary['estimated_monthly_savings'] = int(
            summary['total_at_risk_orders'] * avg_delay_cost * recovery_rate
        )

        return summary


def generate_prescriptions(df: pd.DataFrame, datasets: Dict[str, pd.DataFrame] = None,
                           min_risk: float = 0.5) -> Tuple[pd.DataFrame, Dict]:
    """
    Main entry point: generate action plan with prescriptions.

    Args:
        df: DataFrame with predictions and features
        datasets: Raw datasets for reference
        min_risk: Minimum risk threshold

    Returns:
        (action_plan_df, summary_dict)
    """
    engine = PrescriptionEngine(datasets)

    print(f"\nGenerating prescriptions for orders with risk >= {min_risk}...")
    action_plan = engine.generate_action_plan(df, min_risk=min_risk)

    summary = engine.summarize_actions(action_plan)

    print("\n" + "="*50)
    print("Action Plan Summary:")
    print("="*50)
    print(f"At-risk orders: {summary['total_at_risk_orders']}")
    print(f"Total actions:  {summary['total_actions']}")
    print(f"High priority:  {summary['high_priority_actions']}")
    print(f"\nEstimated monthly savings: ₹{summary['estimated_monthly_savings']:,}")

    if 'actions_by_type' in summary:
        print("\nActions by type:")
        for action, count in summary['actions_by_type'].items():
            print(f"  {action}: {count}")

    return action_plan, summary


if __name__ == "__main__":
    from data import load_and_prepare_data
    from features import engineer_features
    from model import train_and_evaluate_model, predict_new_orders

    # Test prescription generation
    datasets = load_and_prepare_data()
    features_df = engineer_features(datasets)

    if 'is_delayed' in features_df.columns:
        predictor, metrics = train_and_evaluate_model(features_df)
        predictions_df = predict_new_orders(predictor, features_df)

        action_plan, summary = generate_prescriptions(
            predictions_df,
            datasets=datasets,
            min_risk=0.5
        )

        print("\n" + "="*50)
        print("Sample Actions:")
        print("="*50)
        print(action_plan.head(10).to_string(index=False))
