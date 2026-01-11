"""Tests for data preprocessor module."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessor import DataPreprocessor


@pytest.fixture
def sample_df():
    """Create a sample DataFrame."""
    return pd.DataFrame({
        'age': [25, 30, np.nan, 40, 45],
        'salary': [50000, 60000, 70000, np.nan, 90000],
        'category': ['A', 'B', 'A', 'C', 'B']
    })


@pytest.fixture
def preprocessor():
    """Create a preprocessor instance."""
    return DataPreprocessor()


def test_handle_missing_values(preprocessor, sample_df):
    """Test handling missing values."""
    df_cleaned = preprocessor.handle_missing_values(sample_df, strategy='mean')
    # Numeric columns should have no missing values after imputation
    assert df_cleaned[['age', 'salary']].isnull().sum().sum() == 0


def test_encode_categorical(preprocessor, sample_df):
    """Test categorical encoding."""
    df_encoded = preprocessor.encode_categorical(
        sample_df,
        columns=['category'],
        method='label'
    )
    assert df_encoded['category'].dtype in [np.int32, np.int64]


def test_scale_features(preprocessor):
    """Test feature scaling."""
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50]
    })

    df_scaled = preprocessor.scale_features(df, ['feature1', 'feature2'], method='standard')

    # Check that mean is close to 0 and std is approximately 1 (within reasonable tolerance)
    assert abs(df_scaled['feature1'].mean()) < 0.01
    assert abs(df_scaled['feature1'].std() - 1.0) < 0.2  # Allow for sample vs population std difference


def test_create_time_features(preprocessor):
    """Test time feature creation."""
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=5)
    })

    df_with_features = preprocessor.create_time_features(df, 'date')

    assert 'date_year' in df_with_features.columns
    assert 'date_month' in df_with_features.columns
    assert 'date_day' in df_with_features.columns
    assert 'date_dayofweek' in df_with_features.columns


def test_create_interaction_features(preprocessor):
    """Test interaction feature creation."""
    df = pd.DataFrame({
        'feat1': [1, 2, 3],
        'feat2': [4, 5, 6]
    })

    df_with_interactions = preprocessor.create_interaction_features(
        df,
        [('feat1', 'feat2')]
    )

    assert 'feat1_x_feat2' in df_with_interactions.columns
    assert 'feat1_div_feat2' in df_with_interactions.columns
