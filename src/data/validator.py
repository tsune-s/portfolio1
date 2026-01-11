"""Data validation utilities using Pydantic."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, field_validator, Field
import pandas as pd
import numpy as np


class DataQualityReport(BaseModel):
    """Data quality report model."""

    dataset_name: str
    row_count: int
    column_count: int
    missing_values: Dict[str, int]
    duplicate_rows: int
    data_types: Dict[str, str]
    quality_score: float = Field(ge=0, le=100)
    issues: List[str] = []


class DataValidator:
    """Data quality validation and reporting."""

    def __init__(self):
        """Initialize data validator."""
        self.validation_rules = {}

    def validate_schema(
        self,
        df: pd.DataFrame,
        expected_columns: List[str],
        expected_dtypes: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Validate DataFrame schema.

        Args:
            df: Input DataFrame
            expected_columns: List of expected column names
            expected_dtypes: Optional dictionary of expected data types

        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "missing_columns": [],
            "extra_columns": [],
            "dtype_mismatches": []
        }

        # Check for missing columns
        missing = set(expected_columns) - set(df.columns)
        if missing:
            results["valid"] = False
            results["missing_columns"] = list(missing)

        # Check for extra columns
        extra = set(df.columns) - set(expected_columns)
        if extra:
            results["extra_columns"] = list(extra)

        # Check data types if provided
        if expected_dtypes:
            for col, expected_dtype in expected_dtypes.items():
                if col in df.columns:
                    actual_dtype = str(df[col].dtype)
                    if expected_dtype not in actual_dtype:
                        results["valid"] = False
                        results["dtype_mismatches"].append({
                            "column": col,
                            "expected": expected_dtype,
                            "actual": actual_dtype
                        })

        return results

    def check_missing_values(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Check for missing values in DataFrame.

        Args:
            df: Input DataFrame
            threshold: Maximum allowed missing value ratio (0-1)

        Returns:
            Dictionary with missing value information
        """
        missing_info = {}
        problematic_columns = []

        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_ratio = missing_count / len(df)

            missing_info[col] = {
                "count": int(missing_count),
                "ratio": float(missing_ratio)
            }

            if missing_ratio > threshold:
                problematic_columns.append(col)

        return {
            "missing_info": missing_info,
            "problematic_columns": problematic_columns,
            "total_missing": df.isnull().sum().sum()
        }

    def check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate rows.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with duplicate information
        """
        duplicate_count = df.duplicated().sum()
        duplicate_ratio = duplicate_count / len(df)

        return {
            "duplicate_count": int(duplicate_count),
            "duplicate_ratio": float(duplicate_ratio),
            "has_duplicates": duplicate_count > 0
        }

    def check_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "iqr"
    ) -> Dict[str, Any]:
        """Detect outliers in numerical columns.

        Args:
            df: Input DataFrame
            columns: Columns to check. If None, check all numeric columns
            method: Detection method (iqr, zscore)

        Returns:
            Dictionary with outlier information
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outlier_info = {}

        for col in columns:
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

            elif method == "zscore":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = (z_scores > 3).sum()

            outlier_info[col] = {
                "count": int(outliers),
                "ratio": float(outliers / len(df))
            }

        return outlier_info

    def check_value_ranges(
        self,
        df: pd.DataFrame,
        range_constraints: Dict[str, tuple]
    ) -> Dict[str, Any]:
        """Check if values are within expected ranges.

        Args:
            df: Input DataFrame
            range_constraints: Dictionary mapping column names to (min, max) tuples

        Returns:
            Dictionary with range violation information
        """
        violations = {}

        for col, (min_val, max_val) in range_constraints.items():
            if col in df.columns:
                outside_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                violations[col] = {
                    "count": int(outside_range),
                    "ratio": float(outside_range / len(df)),
                    "min_expected": min_val,
                    "max_expected": max_val,
                    "min_actual": float(df[col].min()),
                    "max_actual": float(df[col].max())
                }

        return violations

    def generate_quality_report(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset"
    ) -> DataQualityReport:
        """Generate comprehensive data quality report.

        Args:
            df: Input DataFrame
            dataset_name: Name of the dataset

        Returns:
            DataQualityReport object
        """
        # Calculate quality metrics
        missing_info = self.check_missing_values(df)
        duplicate_info = self.check_duplicates(df)

        # Calculate quality score (0-100)
        missing_penalty = min(missing_info["total_missing"] / (len(df) * len(df.columns)) * 100, 50)
        duplicate_penalty = min(duplicate_info["duplicate_ratio"] * 100, 50)
        quality_score = 100 - missing_penalty - duplicate_penalty

        # Collect issues
        issues = []
        if missing_info["problematic_columns"]:
            issues.append(f"High missing values in: {', '.join(missing_info['problematic_columns'])}")
        if duplicate_info["has_duplicates"]:
            issues.append(f"Found {duplicate_info['duplicate_count']} duplicate rows")

        report = DataQualityReport(
            dataset_name=dataset_name,
            row_count=len(df),
            column_count=len(df.columns),
            missing_values={k: v["count"] for k, v in missing_info["missing_info"].items()},
            duplicate_rows=duplicate_info["duplicate_count"],
            data_types={col: str(dtype) for col, dtype in df.dtypes.items()},
            quality_score=round(quality_score, 2),
            issues=issues
        )

        return report

    def print_quality_report(self, report: DataQualityReport) -> None:
        """Print formatted quality report.

        Args:
            report: DataQualityReport object
        """
        print(f"\n{'='*60}")
        print(f"Data Quality Report: {report.dataset_name}")
        print(f"{'='*60}")
        print(f"Dimensions: {report.row_count} rows × {report.column_count} columns")
        print(f"Quality Score: {report.quality_score}/100")
        print(f"Duplicate Rows: {report.duplicate_rows}")
        print(f"\nMissing Values:")
        for col, count in report.missing_values.items():
            if count > 0:
                pct = (count / report.row_count) * 100
                print(f"  {col}: {count} ({pct:.2f}%)")

        if report.issues:
            print(f"\nIssues Found:")
            for issue in report.issues:
                print(f"  - {issue}")
        else:
            print("\nNo major issues found!")

        print(f"{'='*60}\n")
