"""Dashboard utility components for Streamlit."""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


class DashboardComponents:
    """Utility components for building dashboards."""

    @staticmethod
    def create_metric_card(
        title: str,
        value: Any,
        delta: Optional[Any] = None,
        delta_color: str = "normal"
    ) -> Dict[str, Any]:
        """Create a metric card configuration.

        Args:
            title: Metric title
            value: Metric value
            delta: Optional delta value
            delta_color: Color for delta (normal, inverse, off)

        Returns:
            Dictionary with metric configuration
        """
        return {
            "title": title,
            "value": value,
            "delta": delta,
            "delta_color": delta_color
        }

    @staticmethod
    def format_large_number(num: float) -> str:
        """Format large numbers with K, M, B suffixes.

        Args:
            num: Number to format

        Returns:
            Formatted string
        """
        if abs(num) >= 1_000_000_000:
            return f"{num / 1_000_000_000:.2f}B"
        elif abs(num) >= 1_000_000:
            return f"{num / 1_000_000:.2f}M"
        elif abs(num) >= 1_000:
            return f"{num / 1_000:.2f}K"
        else:
            return f"{num:.2f}"

    @staticmethod
    def calculate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with summary statistics
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        stats = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "missing_percentage": round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2),
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(df.columns) - len(numeric_cols)
        }

        return stats

    @staticmethod
    def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
        """Get detailed column information.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with column statistics
        """
        info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Missing': df.isnull().sum().values,
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2).values,
            'Unique': df.nunique().values
        })

        return info

    @staticmethod
    def prepare_data_for_download(
        df: pd.DataFrame,
        format: str = "csv"
    ) -> bytes:
        """Prepare data for download.

        Args:
            df: DataFrame to prepare
            format: Format (csv, json)

        Returns:
            Bytes data
        """
        if format == "csv":
            return df.to_csv(index=False).encode('utf-8')
        elif format == "json":
            return df.to_json(orient='records', indent=2).encode('utf-8')
        else:
            raise ValueError(f"Unsupported format: {format}")

    @staticmethod
    def filter_dataframe(
        df: pd.DataFrame,
        filters: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply filters to DataFrame.

        Args:
            df: Input DataFrame
            filters: Dictionary of column: value filters

        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()

        for column, value in filters.items():
            if column in filtered_df.columns:
                if isinstance(value, (list, tuple)):
                    filtered_df = filtered_df[filtered_df[column].isin(value)]
                else:
                    filtered_df = filtered_df[filtered_df[column] == value]

        return filtered_df

    @staticmethod
    def create_color_scale(
        values: List[float],
        colorscale: str = "RdYlGn"
    ) -> List[str]:
        """Create color scale for values.

        Args:
            values: List of values
            colorscale: Name of colorscale (RdYlGn, Viridis, etc.)

        Returns:
            List of color codes
        """
        import plotly.express as px

        # Normalize values to 0-1 range
        min_val = min(values)
        max_val = max(values)
        normalized = [(v - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for v in values]

        # Get colors from plotly colorscale
        colors = px.colors.sample_colorscale(colorscale, normalized)
        return colors
