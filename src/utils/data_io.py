"""
Data I/O utilities for Parquet and CSV read/write operations.
Centralizes all file I/O to ensure consistent paths and error handling.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (contains data/)."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "data").is_dir():
            return current
        current = current.parent
    return Path.cwd()


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"


def ensure_dirs():
    """Create data directories if they don't exist."""
    for d in [RAW_DIR, PROCESSED_DIR, REFERENCE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def save_parquet(df: pd.DataFrame, filename: str, subdir: str = "raw") -> Path:
    """
    Save a DataFrame to Parquet in the specified data subdirectory.

    Args:
        df: DataFrame to save.
        filename: File name (e.g., 'touchpoints.parquet').
        subdir: 'raw' or 'processed'.

    Returns:
        Path to the saved file.
    """
    ensure_dirs()
    target_dir = DATA_DIR / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir / filename
    df.to_parquet(filepath, index=False, engine="pyarrow")
    return filepath


def load_parquet(filename: str, subdir: str = "raw") -> pd.DataFrame:
    """
    Load a Parquet file from the specified data subdirectory.

    Args:
        filename: File name (e.g., 'touchpoints.parquet').
        subdir: 'raw' or 'processed'.

    Returns:
        Loaded DataFrame.
    """
    filepath = DATA_DIR / subdir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    return pd.read_parquet(filepath, engine="pyarrow")


def save_csv(df: pd.DataFrame, filename: str, subdir: str = "processed") -> Path:
    """Save a DataFrame to CSV for export."""
    ensure_dirs()
    target_dir = DATA_DIR / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir / filename
    df.to_csv(filepath, index=False)
    return filepath


def load_reference_json(filename: str) -> dict:
    """Load a JSON reference file."""
    import json
    filepath = REFERENCE_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Reference file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
