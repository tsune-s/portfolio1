"""Data loading utilities for various data sources."""

from pathlib import Path
from typing import Optional, Union
import pandas as pd
import polars as pl
from sqlalchemy import create_engine
import requests
from bs4 import BeautifulSoup


class DataLoader:
    """Flexible data loader supporting multiple sources and formats."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize data loader.

        Args:
            data_dir: Base directory for data files. Defaults to project's data directory.
        """
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_csv(
        self,
        filename: str,
        use_polars: bool = False,
        **kwargs
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Load CSV file using pandas or polars.

        Args:
            filename: Name of the CSV file
            use_polars: Whether to use polars instead of pandas
            **kwargs: Additional arguments passed to read_csv

        Returns:
            DataFrame (pandas or polars)
        """
        filepath = self.raw_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if use_polars:
            return pl.read_csv(filepath, **kwargs)
        return pd.read_csv(filepath, **kwargs)

    def load_from_url(
        self,
        url: str,
        filename: Optional[str] = None,
        save: bool = True
    ) -> pd.DataFrame:
        """Download and load data from URL.

        Args:
            url: URL to download data from
            filename: Optional filename to save the data
            save: Whether to save the downloaded data

        Returns:
            pandas DataFrame
        """
        response = requests.get(url)
        response.raise_for_status()

        # Determine file format from URL or content-type
        if url.endswith('.csv') or 'text/csv' in response.headers.get('content-type', ''):
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
        elif url.endswith('.json') or 'application/json' in response.headers.get('content-type', ''):
            df = pd.read_json(response.text)
        else:
            raise ValueError(f"Unsupported file format from URL: {url}")

        if save and filename:
            save_path = self.raw_dir / filename
            df.to_csv(save_path, index=False)

        return df

    def load_from_sql(
        self,
        query: str,
        connection_string: str
    ) -> pd.DataFrame:
        """Load data from SQL database.

        Args:
            query: SQL query to execute
            connection_string: SQLAlchemy connection string

        Returns:
            pandas DataFrame
        """
        engine = create_engine(connection_string)
        return pd.read_sql(query, engine)

    def scrape_table_from_html(
        self,
        url: str,
        table_index: int = 0
    ) -> pd.DataFrame:
        """Scrape HTML table from webpage.

        Args:
            url: URL of the webpage
            table_index: Index of the table to extract (0-based)

        Returns:
            pandas DataFrame
        """
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        tables = soup.find_all('table')

        if not tables:
            raise ValueError(f"No tables found at URL: {url}")

        if table_index >= len(tables):
            raise IndexError(f"Table index {table_index} out of range. Found {len(tables)} tables.")

        # Use pandas to parse HTML table
        dfs = pd.read_html(str(tables[table_index]))
        return dfs[0]

    def save_processed(
        self,
        df: Union[pd.DataFrame, pl.DataFrame],
        filename: str,
        format: str = "csv"
    ) -> Path:
        """Save processed data to file.

        Args:
            df: DataFrame to save
            filename: Output filename
            format: File format (csv, parquet, json)

        Returns:
            Path to saved file
        """
        filepath = self.processed_dir / f"{filename}.{format}"

        if isinstance(df, pl.DataFrame):
            if format == "csv":
                df.write_csv(filepath)
            elif format == "parquet":
                df.write_parquet(filepath)
            elif format == "json":
                df.write_json(filepath)
        else:
            if format == "csv":
                df.to_csv(filepath, index=False)
            elif format == "parquet":
                df.to_parquet(filepath, index=False)
            elif format == "json":
                df.to_json(filepath, orient="records", indent=2)

        return filepath
