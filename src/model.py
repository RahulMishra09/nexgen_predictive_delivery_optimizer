"""
ML Model Module
Train, evaluate, and deploy delay prediction models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_curve
)
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


class DelayPredictor:
    """Train and predict shipment delay risk."""

    def __init__(self, model_type: str = 'logistic'):
        """
        Initialize predictor.

        Args:
            model_type: 'logistic' or 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.trained = False

    def prepare_data(self, df: pd.DataFrame, target_col: str = 'is_delayed',
                     test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare train/test split with proper scaling.

        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        # Ensure target exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")

        # Get features
        exclude_cols = [target_col, 'order_id', 'customer_id', 'warehouse_id',
                        'carrier_id', 'vehicle_id', 'actual_delivery',
                        'promised_date', 'ship_date', 'delay_hours']

        # Select numeric features only
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        feature_df = df[feature_cols].select_dtypes(include=[np.number])

        # Handle missing values
        feature_df = feature_df.fillna(feature_df.median())

        # Handle infinite values
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        self.feature_cols = feature_df.columns.tolist()

        X = feature_df.values
        y = df[target_col].values

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test, self.feature_cols

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """Train the model."""
        print(f"Training {self.model_type} model...")

        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                **kwargs
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 20),
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.fit(X_train, y_train)
        self.trained = True
        print("✓ Model training complete")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance.

        Returns:
            Dictionary of metrics
        """
        if not self.trained:
            raise ValueError("Model not trained yet")

        # Predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # Metrics
        auc = roc_auc_score(y_test, y_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        metrics = {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return delay probabilities."""
        if not self.trained:
            raise ValueError("Model not trained yet")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N most important features.

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.trained:
            raise ValueError("Model not trained yet")

        if self.model_type == 'logistic':
            # Use absolute coefficients for logistic regression
            importance = np.abs(self.model.coef_[0])
        elif self.model_type == 'random_forest':
            importance = self.model.feature_importances_
        else:
            return pd.DataFrame()

        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)

        return feature_importance

    def save(self, filepath: str = "models/model.joblib"):
        """Save model and scaler to disk."""
        if not self.trained:
            raise ValueError("Model not trained yet")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'model_type': self.model_type
        }

        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")

    def load(self, filepath: str = "models/model.joblib"):
        """Load model and scaler from disk."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_cols = model_data['feature_cols']
        self.model_type = model_data['model_type']
        self.trained = True

        print(f"✓ Model loaded from {filepath}")


def train_and_evaluate_model(df: pd.DataFrame, model_type: str = 'logistic',
                              target_col: str = 'is_delayed',
                              test_size: float = 0.2) -> Tuple[DelayPredictor, Dict]:
    """
    Main training pipeline.

    Args:
        df: Feature dataframe
        model_type: 'logistic' or 'random_forest'
        target_col: Target column name
        test_size: Test split ratio

    Returns:
        (trained_model, metrics_dict)
    """
    # Initialize predictor
    predictor = DelayPredictor(model_type=model_type)

    # Prepare data
    print("\nPreparing train/test split...")
    X_train, X_test, y_train, y_test, feature_names = predictor.prepare_data(
        df, target_col=target_col, test_size=test_size
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {len(feature_names)}")
    print(f"Positive class rate (train): {y_train.mean():.2%}")

    # Train
    predictor.train(X_train, y_train)

    # Evaluate
    print("\nEvaluating model...")
    metrics = predictor.evaluate(X_test, y_test)

    print("\n" + "="*50)
    print("Model Performance:")
    print("="*50)
    print(f"AUC-ROC:   {metrics['auc']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1-Score:  {metrics['f1']:.3f}")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print("\nConfusion Matrix:")
    print(f"  TP: {metrics['true_positives']:>5}  FP: {metrics['false_positives']:>5}")
    print(f"  FN: {metrics['false_negatives']:>5}  TN: {metrics['true_negatives']:>5}")

    return predictor, metrics


def predict_new_orders(predictor: DelayPredictor, df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate risk scores for new/future orders.

    Args:
        predictor: Trained DelayPredictor
        df: DataFrame with same features as training

    Returns:
        DataFrame with risk scores and predictions
    """
    if not predictor.trained:
        raise ValueError("Model not trained yet")

    # Prepare features
    feature_df = df[predictor.feature_cols].fillna(0)
    feature_df = feature_df.replace([np.inf, -np.inf], 0)

    X = feature_df.values

    # Predict
    risk_scores = predictor.predict_proba(X)
    predictions = predictor.predict(X)

    # Add to dataframe
    result = df.copy()
    result['delay_risk_score'] = risk_scores
    result['predicted_delay'] = predictions
    result['risk_category'] = pd.cut(
        risk_scores,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )

    return result


if __name__ == "__main__":
    from data import load_and_prepare_data
    from features import engineer_features

    # Test model training
    datasets = load_and_prepare_data()
    features_df = engineer_features(datasets)

    if 'is_delayed' in features_df.columns:
        predictor, metrics = train_and_evaluate_model(
            features_df,
            model_type='random_forest'
        )

        # Show feature importance
        print("\n" + "="*50)
        print("Top 10 Features:")
        print("="*50)
        print(predictor.get_feature_importance(10).to_string(index=False))

        # Save model
        predictor.save()
