"""Data visualization utilities."""

from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class DataVisualizer:
    """Comprehensive data visualization toolkit."""

    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """Initialize visualizer.

        Args:
            style: Matplotlib style to use
        """
        plt.style.use('default')
        sns.set_palette("husl")
        self.figsize = (12, 6)

    def plot_distribution(
        self,
        data: pd.Series,
        title: str = "Distribution Plot",
        bins: int = 30,
        kde: bool = True
    ) -> plt.Figure:
        """Plot distribution of a variable.

        Args:
            data: Data to plot
            title: Plot title
            bins: Number of bins for histogram
            kde: Whether to overlay KDE plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        sns.histplot(data, bins=bins, kde=kde, ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(data.name or 'Value')
        ax.set_ylabel('Frequency')

        plt.tight_layout()
        return fig

    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        title: str = "Correlation Matrix",
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """Plot correlation matrix heatmap.

        Args:
            df: DataFrame with numerical columns
            title: Plot title
            figsize: Optional figure size

        Returns:
            Matplotlib figure
        """
        figsize = figsize or (12, 10)
        fig, ax = plt.subplots(figsize=figsize)

        # Calculate correlation matrix
        corr = df.select_dtypes(include=[np.number]).corr()

        # Create heatmap
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: List[float],
        top_n: int = 20,
        title: str = "Feature Importance"
    ) -> plt.Figure:
        """Plot feature importance.

        Args:
            feature_names: List of feature names
            importances: List of importance values
            top_n: Number of top features to display
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Create DataFrame and sort
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=self.figsize)

        sns.barplot(data=df, x='importance', y='feature', ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')

        plt.tight_layout()
        return fig

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Confusion Matrix"
    ) -> plt.Figure:
        """Plot confusion matrix.

        Args:
            cm: Confusion matrix array
            labels: Class labels
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

        plt.tight_layout()
        return fig

    def plot_time_series(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_columns: List[str],
        title: str = "Time Series Plot"
    ) -> plt.Figure:
        """Plot time series data.

        Args:
            df: DataFrame with time series data
            date_column: Name of date column
            value_columns: List of value columns to plot
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for col in value_columns:
            ax.plot(df[date_column], df[col], label=col, marker='o', markersize=3)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    def create_interactive_scatter(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        size: Optional[str] = None,
        title: str = "Interactive Scatter Plot"
    ) -> go.Figure:
        """Create interactive scatter plot using Plotly.

        Args:
            df: DataFrame
            x: X-axis column
            y: Y-axis column
            color: Optional color column
            size: Optional size column
            title: Plot title

        Returns:
            Plotly figure
        """
        fig = px.scatter(
            df,
            x=x,
            y=y,
            color=color,
            size=size,
            title=title,
            hover_data=df.columns
        )

        fig.update_layout(
            template='plotly_white',
            hovermode='closest'
        )

        return fig

    def create_interactive_line(
        self,
        df: pd.DataFrame,
        x: str,
        y: List[str],
        title: str = "Interactive Line Plot"
    ) -> go.Figure:
        """Create interactive line plot using Plotly.

        Args:
            df: DataFrame
            x: X-axis column
            y: List of Y-axis columns
            title: Plot title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        for col in y:
            fig.add_trace(go.Scatter(
                x=df[x],
                y=df[col],
                mode='lines+markers',
                name=col
            ))

        fig.update_layout(
            title=title,
            xaxis_title=x,
            yaxis_title='Value',
            template='plotly_white',
            hovermode='x unified'
        )

        return fig

    def create_dashboard_summary(
        self,
        df: pd.DataFrame,
        numerical_cols: Optional[List[str]] = None
    ) -> plt.Figure:
        """Create summary dashboard with multiple plots.

        Args:
            df: DataFrame to visualize
            numerical_cols: Optional list of numerical columns

        Returns:
            Matplotlib figure with subplots
        """
        if numerical_cols is None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        n_cols = min(len(numerical_cols), 4)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        # Distribution plots
        for i, col in enumerate(numerical_cols[:4]):
            sns.histplot(df[col], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)

        plt.suptitle('Data Summary Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
