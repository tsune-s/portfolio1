"""Tests for data loader module."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
from src.data.loader import DataLoader


@pytest.fixture
def data_loader(tmp_path):
    """Create a data loader with temporary directory."""
    return DataLoader(data_dir=tmp_path)


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35]
    })
    filepath = tmp_path / "raw" / "sample.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    return filepath


def test_data_loader_initialization(data_loader, tmp_path):
    """Test DataLoader initialization."""
    assert data_loader.data_dir == tmp_path
    assert data_loader.raw_dir.exists()
    assert data_loader.processed_dir.exists()


def test_load_csv(data_loader, sample_csv):
    """Test loading CSV file."""
    df = data_loader.load_csv("sample.csv")
    assert len(df) == 3
    assert list(df.columns) == ['id', 'name', 'age']


def test_load_csv_file_not_found(data_loader):
    """Test loading non-existent file."""
    with pytest.raises(FileNotFoundError):
        data_loader.load_csv("nonexistent.csv")


def test_save_processed(data_loader):
    """Test saving processed data."""
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })

    filepath = data_loader.save_processed(df, "test_output", format="csv")
    assert filepath.exists()

    # Verify saved data
    loaded_df = pd.read_csv(filepath)
    assert len(loaded_df) == 3
