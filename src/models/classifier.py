"""Classification model implementation."""

from typing import Dict, Optional, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from .base_model import BaseModel


class ClassifierModel(BaseModel):
    """Classification model with multiple algorithm support."""

    SUPPORTED_ALGORITHMS = {
        'random_forest': RandomForestClassifier,
        'xgboost': XGBClassifier,
        'lightgbm': LGBMClassifier
    }

    def __init__(
        self,
        algorithm: str = 'xgboost',
        model_name: Optional[str] = None,
        **model_params
    ):
        """Initialize classifier.

        Args:
            algorithm: Algorithm to use (random_forest, xgboost, lightgbm)
            model_name: Optional model name
            **model_params: Parameters to pass to the model
        """
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Algorithm {algorithm} not supported. "
                f"Choose from: {list(self.SUPPORTED_ALGORITHMS.keys())}"
            )

        model_name = model_name or f"{algorithm}_classifier"
        super().__init__(model_name)

        self.algorithm = algorithm
        self.model = self.SUPPORTED_ALGORITHMS[algorithm](**model_params)
        self.metadata['algorithm'] = algorithm
        self.metadata['model_params'] = model_params

    def train(
        self,
        X,
        y,
        eval_set: Optional[tuple] = None,
        verbose: bool = True,
        **kwargs
    ) -> None:
        """Train the classification model.

        Args:
            X: Training features
            y: Training labels
            eval_set: Optional evaluation set (X_val, y_val)
            verbose: Whether to print training progress
            **kwargs: Additional training parameters
        """
        if verbose:
            print(f"Training {self.algorithm} classifier...")

        # XGBoost and LightGBM support eval_set
        if eval_set is not None and self.algorithm in ['xgboost', 'lightgbm']:
            kwargs['eval_set'] = [eval_set]

        self.model.fit(X, y, **kwargs)
        self.is_trained = True

        if verbose:
            print("Training completed!")

    def predict(self, X) -> np.ndarray:
        """Make class predictions.

        Args:
            X: Features to predict on

        Returns:
            Predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Features to predict on

        Returns:
            Predicted probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict_proba(X)

    def evaluate(
        self,
        X,
        y,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            X: Test features
            y: True labels
            verbose: Whether to print evaluation results

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0)
        }

        # Add ROC AUC for binary classification
        if len(np.unique(y)) == 2:
            metrics['roc_auc'] = roc_auc_score(y, y_proba[:, 1])

        if verbose:
            print("\nModel Evaluation Results:")
            print("=" * 50)
            for metric, value in metrics.items():
                print(f"{metric.capitalize()}: {value:.4f}")
            print("=" * 50)
            print("\nClassification Report:")
            print(classification_report(y, y_pred))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y, y_pred))

        self.metadata['evaluation_metrics'] = metrics
        return metrics

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names/indices to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")

        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError(f"{self.algorithm} does not support feature importance")

        importances = self.model.feature_importances_
        return {f"feature_{i}": imp for i, imp in enumerate(importances)}
