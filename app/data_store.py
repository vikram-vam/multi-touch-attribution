"""
Central data store for the Dash application.
Loads all pre-computed Parquet files at startup and provides accessor functions.
"""

import sys
import os
from pathlib import Path

import pandas as pd
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_io import PROCESSED_DIR, RAW_DIR
from src.utils.config_loader import load_channel_config, load_demo_narrative, get_channel_display_names

# ── Module-level data loading ──

def _safe_load(filename: str, subdir: str) -> pd.DataFrame:
    """Load a Parquet file, returning empty DataFrame if not found."""
    filepath = PROJECT_ROOT / "data" / subdir / filename
    if filepath.exists():
        return pd.read_parquet(filepath)
    logger.warning(f"Data file not found: {filepath}")
    return pd.DataFrame()


# Load all data at import time
_attribution_results = _safe_load("attribution_results.parquet", "processed")
_model_comparison = _safe_load("model_comparison.parquet", "processed")
_journeys = _safe_load("journeys.parquet", "raw")
_touchpoints = _safe_load("touchpoints.parquet", "raw")
_conversions = _safe_load("conversions.parquet", "raw")
_channel_spend = _safe_load("channel_spend.parquet", "raw")
_identity_graph = _safe_load("identity_graph.parquet", "raw")

# Parse list columns if needed
if "channel_path" not in _journeys.columns and "channel_path_str" in _journeys.columns:
    _journeys["channel_path"] = _journeys["channel_path_str"].apply(
        lambda x: x.split("|") if isinstance(x, str) else []
    )
if "channel_set" not in _journeys.columns and "channel_set_str" in _journeys.columns:
    _journeys["channel_set"] = _journeys["channel_set_str"].apply(
        lambda x: x.split("|") if isinstance(x, str) else []
    )

# Load configs
try:
    _channel_config = load_channel_config()
    _channel_display_names = get_channel_display_names(_channel_config)
    _demo_narrative = load_demo_narrative()
except Exception as e:
    logger.warning(f"Config loading error: {e}")
    _channel_config = []
    _channel_display_names = {}
    _demo_narrative = {}

logger.info(f"Data store loaded: {len(_attribution_results)} attribution rows, "
            f"{len(_journeys)} journeys")


# ── Accessor Functions ──

def get_attribution_results() -> pd.DataFrame:
    return _attribution_results

def get_model_comparison() -> pd.DataFrame:
    return _model_comparison

def get_journeys() -> pd.DataFrame:
    return _journeys

def get_touchpoints() -> pd.DataFrame:
    return _touchpoints

def get_conversions() -> pd.DataFrame:
    return _conversions

def get_channel_spend() -> pd.DataFrame:
    return _channel_spend

def get_identity_graph() -> pd.DataFrame:
    return _identity_graph

def get_channel_display_name(channel_id: str) -> str:
    return _channel_display_names.get(channel_id, channel_id.replace("_", " ").title())

def get_demo_narrative() -> dict:
    return _demo_narrative

def get_available_models() -> list:
    if len(_attribution_results) > 0:
        return sorted(_attribution_results["model_name"].unique().tolist())
    return []

def get_channel_list() -> list:
    if len(_attribution_results) > 0:
        return sorted(_attribution_results["channel_id"].unique().tolist())
    return []
