"""
Utility Functions
Helper functions for visualization, metrics, and reporting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve
from typing import Dict, Tuple, List
import io
import base64


def create_metrics_summary(metrics: Dict) -> str:
    """Format metrics dictionary as readable string."""
    summary = f"""
    **Model Performance Metrics**

    - **AUC-ROC**: {metrics['auc']:.3f}
    - **Precision**: {metrics['precision']:.3f}
    - **Recall**: {metrics['recall']:.3f}
    - **F1-Score**: {metrics['f1']:.3f}
    - **Accuracy**: {metrics['accuracy']:.3f}

    **Confusion Matrix**:
    - True Positives (TP): {metrics['true_positives']}
    - False Positives (FP): {metrics['false_positives']}
    - True Negatives (TN): {metrics['true_negatives']}
    - False Negatives (FN): {metrics['false_negatives']}
    """
    return summary


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15):
    """
    Create horizontal bar chart of feature importance.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display

    Returns:
        Plotly figure
    """
    data = importance_df.head(top_n).sort_values('importance')

    fig = go.Figure(go.Bar(
        x=data['importance'],
        y=data['feature'],
        orientation='h',
        marker_color='#1f77b4'
    ))

    fig.update_layout(
        title=f'Top {top_n} Most Important Features',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=max(400, top_n * 25),
        margin=dict(l=150, r=20, t=50, b=50)
    )

    return fig


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, auc_score: float):
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        auc_score: AUC score

    Returns:
        Plotly figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_score:.3f})',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', width=1, dash='dash')
    ))

    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        showlegend=True
    )

    return fig


def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray):
    """
    Plot Precision-Recall curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities

    Returns:
        Plotly figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name='Precision-Recall Curve',
        line=dict(color='#2ca02c', width=2)
    ))

    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        height=500
    )

    return fig


def plot_risk_distribution(df: pd.DataFrame, risk_col: str = 'delay_risk_score'):
    """
    Plot distribution of risk scores.

    Args:
        df: DataFrame with risk scores
        risk_col: Name of risk score column

    Returns:
        Plotly figure
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[risk_col],
        nbinsx=50,
        marker_color='#ff7f0e',
        opacity=0.7
    ))

    # Add vertical lines for risk thresholds
    fig.add_vline(x=0.3, line_dash="dash", line_color="green",
                  annotation_text="Low Risk", annotation_position="top")
    fig.add_vline(x=0.6, line_dash="dash", line_color="orange",
                  annotation_text="Medium Risk", annotation_position="top")

    fig.update_layout(
        title='Distribution of Delay Risk Scores',
        xaxis_title='Risk Score',
        yaxis_title='Count',
        height=400
    )

    return fig


def plot_confusion_matrix(metrics: Dict):
    """
    Plot confusion matrix heatmap.

    Args:
        metrics: Dictionary with TP, FP, TN, FN

    Returns:
        Plotly figure
    """
    cm = np.array([
        [metrics['true_negatives'], metrics['false_positives']],
        [metrics['false_negatives'], metrics['true_positives']]
    ])

    labels = np.array([
        [f"TN<br>{cm[0,0]}", f"FP<br>{cm[0,1]}"],
        [f"FN<br>{cm[1,0]}", f"TP<br>{cm[1,1]}"]
    ])

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted No Delay', 'Predicted Delay'],
        y=['Actual No Delay', 'Actual Delay'],
        text=labels,
        texttemplate='%{text}',
        textfont={"size": 16},
        colorscale='Blues',
        showscale=True
    ))

    fig.update_layout(
        title='Confusion Matrix',
        height=400,
        xaxis_title='Predicted',
        yaxis_title='Actual'
    )

    return fig


def plot_actions_by_type(action_df: pd.DataFrame):
    """
    Plot bar chart of actions by type.

    Args:
        action_df: Action plan DataFrame

    Returns:
        Plotly figure
    """
    action_counts = action_df['action'].value_counts().reset_index()
    action_counts.columns = ['Action', 'Count']

    fig = go.Figure(go.Bar(
        x=action_counts['Count'],
        y=action_counts['Action'],
        orientation='h',
        marker_color='#9467bd'
    ))

    fig.update_layout(
        title='Recommended Actions by Type',
        xaxis_title='Number of Orders',
        yaxis_title='Action Type',
        height=400
    )

    return fig


def plot_risk_by_priority(df: pd.DataFrame):
    """
    Plot risk distribution by order priority.

    Args:
        df: DataFrame with risk scores and priority

    Returns:
        Plotly figure
    """
    if 'priority' not in df.columns or 'delay_risk_score' not in df.columns:
        return None

    fig = go.Figure()

    for priority in df['priority'].unique():
        data = df[df['priority'] == priority]['delay_risk_score']
        fig.add_trace(go.Box(
            y=data,
            name=str(priority),
            boxmean='sd'
        ))

    fig.update_layout(
        title='Delay Risk Distribution by Order Priority',
        xaxis_title='Priority',
        yaxis_title='Risk Score',
        height=400
    )

    return fig


def calculate_business_impact(action_summary: Dict) -> Dict:
    """
    Calculate estimated business impact metrics.

    Args:
        action_summary: Summary from prescription engine

    Returns:
        Dictionary with business metrics
    """
    # Assumptions (can be configured)
    avg_delay_cost = 500  # ₹ per delayed order
    avg_order_value = 2000  # ₹
    customer_churn_cost = 10000  # ₹ per churned customer
    delay_churn_rate = 0.05  # 5% of delayed customers churn

    at_risk_orders = action_summary.get('total_at_risk_orders', 0)
    high_priority = action_summary.get('high_priority_actions', 0)

    # Calculate impacts
    baseline_delay_cost = at_risk_orders * avg_delay_cost
    intervention_recovery = baseline_delay_cost * 0.6  # 60% recovery with interventions
    prevented_delays = int(at_risk_orders * 0.6)
    prevented_churn = int(prevented_delays * delay_churn_rate)

    impact = {
        'at_risk_orders': at_risk_orders,
        'baseline_delay_cost': baseline_delay_cost,
        'estimated_savings': int(intervention_recovery),
        'prevented_delays': prevented_delays,
        'prevented_churn_customers': prevented_churn,
        'churn_cost_avoided': prevented_churn * customer_churn_cost,
        'total_business_value': int(intervention_recovery + prevented_churn * customer_churn_cost),
        'roi_multiplier': 5.0  # Estimated ROI on intervention costs
    }

    return impact


def format_currency(amount: float, currency: str = '₹') -> str:
    """Format number as currency string."""
    return f"{currency}{amount:,.0f}"


def export_action_plan_csv(action_df: pd.DataFrame) -> str:
    """
    Convert action plan to CSV for download.

    Args:
        action_df: Action plan DataFrame

    Returns:
        CSV string
    """
    return action_df.to_csv(index=False)


def create_executive_summary(metrics: Dict, action_summary: Dict, impact: Dict) -> str:
    """
    Create markdown-formatted executive summary.

    Args:
        metrics: Model performance metrics
        action_summary: Action plan summary
        impact: Business impact calculations

    Returns:
        Markdown string
    """
    summary = f"""
# Executive Summary: Predictive Delivery Optimizer

## Model Performance
- **Prediction Accuracy**: {metrics['accuracy']:.1%}
- **Precision**: {metrics['precision']:.1%} (of predicted delays, {metrics['precision']:.1%} are actual delays)
- **Recall**: {metrics['recall']:.1%} (catching {metrics['recall']:.1%} of all delays)
- **AUC Score**: {metrics['auc']:.3f}

## At-Risk Orders Identified
- **Total at-risk orders**: {impact['at_risk_orders']}
- **High-priority interventions**: {action_summary.get('high_priority_actions', 0)}
- **Average risk score**: {action_summary.get('avg_risk_score', 0):.1%}

## Business Impact (Monthly Estimates)
- **Baseline delay cost**: {format_currency(impact['baseline_delay_cost'])}
- **Estimated savings**: {format_currency(impact['estimated_savings'])}
- **Prevented delays**: {impact['prevented_delays']} orders
- **Customer churn prevented**: {impact['prevented_churn_customers']} customers
- **Total business value**: {format_currency(impact['total_business_value'])}

## Recommended Actions
{_format_action_list(action_summary.get('actions_by_type', {}))}

## Next Steps
1. Review high-priority actions for immediate implementation
2. Assign action items to operations team
3. Monitor intervention effectiveness
4. Retrain model monthly with new data
"""
    return summary


def _format_action_list(actions_by_type: Dict) -> str:
    """Helper to format action list for summary."""
    if not actions_by_type:
        return "- No actions recommended"

    lines = []
    for action, count in actions_by_type.items():
        lines.append(f"- **{action}**: {count} orders")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test utilities
    sample_metrics = {
        'auc': 0.85,
        'precision': 0.78,
        'recall': 0.82,
        'f1': 0.80,
        'accuracy': 0.88,
        'true_positives': 120,
        'false_positives': 30,
        'true_negatives': 800,
        'false_negatives': 50
    }

    print(create_metrics_summary(sample_metrics))
