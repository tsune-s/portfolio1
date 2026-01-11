"""Helper utility functions."""

from pathlib import Path
from typing import Any, Dict
import json
import time
from functools import wraps


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load dictionary from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def timer(func):
    """Decorator to time function execution.

    Args:
        func: Function to time

    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"{func.__name__} took {elapsed_time:.2f} seconds")
        return result

    return wrapper


def format_bytes(bytes: int) -> str:
    """Format bytes to human-readable string.

    Args:
        bytes: Number of bytes

    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"


def create_experiment_name(base_name: str) -> str:
    """Create timestamped experiment name.

    Args:
        base_name: Base name for experiment

    Returns:
        Experiment name with timestamp
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"
