"""Data preprocessing utilities."""

from typing import List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """Comprehensive data preprocessing pipeline."""

    def __init__(self):
        """Initialize preprocessor with default settings."""
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "mean",
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Handle missing values in DataFrame.

        Args:
            df: Input DataFrame
            strategy: Imputation strategy (mean, median, most_frequent, constant)
            columns: Specific columns to process. If None, process all numeric columns

        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col not in self.imputers:
                self.imputers[col] = SimpleImputer(strategy=strategy)
                df[col] = self.imputers[col].fit_transform(df[[col]])
            else:
                df[col] = self.imputers[col].transform(df[[col]])

        return df

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "iqr",
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """Remove outliers from specified columns.

        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            method: Method to detect outliers (iqr, zscore)
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        df = df.copy()

        for col in columns:
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

            elif method == "zscore":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]

        return df

    def scale_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "standard"
    ) -> pd.DataFrame:
        """Scale numerical features.

        Args:
            df: Input DataFrame
            columns: Columns to scale
            method: Scaling method (standard, minmax)

        Returns:
            DataFrame with scaled features
        """
        df = df.copy()

        scaler_key = f"{method}_{'_'.join(columns)}"
        if scaler_key not in self.scalers:
            if method == "standard":
                self.scalers[scaler_key] = StandardScaler()
            elif method == "minmax":
                self.scalers[scaler_key] = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")

            df[columns] = self.scalers[scaler_key].fit_transform(df[columns])
        else:
            df[columns] = self.scalers[scaler_key].transform(df[columns])

        return df

    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "label"
    ) -> pd.DataFrame:
        """Encode categorical variables.

        Args:
            df: Input DataFrame
            columns: Columns to encode
            method: Encoding method (label, onehot)

        Returns:
            DataFrame with encoded features
        """
        df = df.copy()

        if method == "label":
            for col in columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[col] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.encoders[col].transform(df[col].astype(str))

        elif method == "onehot":
            df = pd.get_dummies(df, columns=columns, prefix=columns)

        return df

    def create_time_features(
        self,
        df: pd.DataFrame,
        date_column: str
    ) -> pd.DataFrame:
        """Extract time-based features from datetime column.

        Args:
            df: Input DataFrame
            date_column: Name of the datetime column

        Returns:
            DataFrame with additional time features
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        df[f"{date_column}_year"] = df[date_column].dt.year
        df[f"{date_column}_month"] = df[date_column].dt.month
        df[f"{date_column}_day"] = df[date_column].dt.day
        df[f"{date_column}_dayofweek"] = df[date_column].dt.dayofweek
        df[f"{date_column}_quarter"] = df[date_column].dt.quarter
        df[f"{date_column}_is_weekend"] = df[date_column].dt.dayofweek.isin([5, 6]).astype(int)

        return df

    def create_interaction_features(
        self,
        df: pd.DataFrame,
        feature_pairs: List[tuple]
    ) -> pd.DataFrame:
        """Create interaction features from feature pairs.

        Args:
            df: Input DataFrame
            feature_pairs: List of tuples containing feature pairs to interact

        Returns:
            DataFrame with interaction features
        """
        df = df.copy()

        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                df[f"{feat1}_x_{feat2}"] = df[feat1] * df[feat2]
                if (df[feat2] != 0).all():
                    df[f"{feat1}_div_{feat2}"] = df[feat1] / df[feat2]

        return df

    def get_feature_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get comprehensive information about features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with feature statistics
        """
        info = pd.DataFrame({
            'dtype': df.dtypes,
            'missing_count': df.isnull().sum(),
            'missing_pct': (df.isnull().sum() / len(df) * 100).round(2),
            'unique_count': df.nunique(),
            'sample_values': df.apply(lambda x: x.dropna().unique()[:3].tolist())
        })

        return info
