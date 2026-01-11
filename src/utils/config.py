"""Configuration management."""

from pathlib import Path
from typing import Any, Dict, Optional
import json
from pydantic import BaseModel


class Config(BaseModel):
    """Application configuration."""

    # Project paths
    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    models_dir: Path = project_root / "models"
    notebooks_dir: Path = project_root / "notebooks"

    # Model settings
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5

    # Data processing
    missing_threshold: float = 0.5
    outlier_threshold: float = 1.5

    @classmethod
    def from_json(cls, filepath: Path) -> "Config":
        """Load configuration from JSON file.

        Args:
            filepath: Path to JSON config file

        Returns:
            Config instance
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_json(self, filepath: Path) -> None:
        """Save configuration to JSON file.

        Args:
            filepath: Path to save config file
        """
        with open(filepath, 'w') as f:
            json.dump(self.model_dump(), f, indent=2, default=str)

    def ensure_dirs(self) -> None:
        """Create all necessary directories."""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
