"""Base model class for all machine learning models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import json


class BaseModel(ABC):
    """Abstract base class for machine learning models."""

    def __init__(self, model_name: str):
        """Initialize base model.

        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.metadata = {}

    @abstractmethod
    def train(self, X, y, **kwargs):
        """Train the model.

        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional training parameters
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Predictions
        """
        pass

    @abstractmethod
    def evaluate(self, X, y) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            X: Test features
            y: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        pass

    def save_model(self, filepath: Path) -> None:
        """Save model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = filepath.with_suffix('.joblib')
        joblib.dump(self.model, model_path)

        # Save metadata
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"Model saved to {model_path}")

    def load_model(self, filepath: Path) -> None:
        """Load model from disk.

        Args:
            filepath: Path to load the model from
        """
        filepath = Path(filepath)

        # Load model
        model_path = filepath.with_suffix('.joblib')
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(model_path)
        self.is_trained = True

        # Load metadata
        metadata_path = filepath.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

        print(f"Model loaded from {model_path}")

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        if self.model is None:
            return {}
        return self.model.get_params()

    def set_params(self, **params) -> None:
        """Set model parameters.

        Args:
            **params: Parameters to set
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        self.model.set_params(**params)
