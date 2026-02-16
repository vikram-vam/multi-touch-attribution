"""
Utility Functions for Erie MCA Demo
Handles configuration loading, validation, and common operations
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

class Config:
    """Configuration manager for the demo"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config = None
        self.load()
    
    def load(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary"""
        return self._config


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    import random
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    logger.info(f"Random seed set to {seed}")


def validate_dataframe_schema(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame has required columns"""
    missing = set(required_columns) - set(df.columns)
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return False
    return True


def ensure_directories():
    """Ensure all required directories exist"""
    dirs = [
        "data/synthetic",
        "data/cache",
        "data/results",
        "logs"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    logger.info("Directory structure verified")


def save_parquet(df: pd.DataFrame, filepath: str, desc: str = ""):
    """Save DataFrame to Parquet with logging"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=False)
    logger.info(f"Saved {desc or filepath}: {len(df):,} rows")


def load_parquet(filepath: str, desc: str = "") -> pd.DataFrame:
    """Load DataFrame from Parquet with logging"""
    df = pd.read_parquet(filepath)
    logger.info(f"Loaded {desc or filepath}: {len(df):,} rows")
    return df


def format_currency(value: float) -> str:
    """Format value as currency"""
    return f"${value:,.0f}"


def format_percent(value: float, decimals: int = 1) -> str:
    """Format value as percentage"""
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 0) -> str:
    """Format number with thousands separator"""
    return f"{value:,.{decimals}f}"


class DataValidator:
    """Validates data quality and schema compliance"""
    
    @staticmethod
    def validate_journey_data(df: pd.DataFrame) -> bool:
        """Validate customer journey data"""
        required_cols = ['customer_id', 'touchpoint_id', 'channel', 'timestamp', 'converted']
        if not validate_dataframe_schema(df, required_cols):
            return False
        
        # Check for nulls
        if df[required_cols].isnull().any().any():
            logger.error("Found null values in required columns")
            return False
        
        # Check conversion flag
        if not df['converted'].isin([0, 1]).all():
            logger.error("Invalid conversion flag values")
            return False
        
        logger.info("Journey data validation passed")
        return True
    
    @staticmethod
    def validate_attribution_results(df: pd.DataFrame, total_conversions: int) -> bool:
        """Validate attribution results satisfy axioms"""
        if 'channel' not in df.columns or 'credit' not in df.columns:
            logger.error("Missing required attribution columns")
            return False
        
        # Shapley efficiency axiom: credits must sum to total conversions
        total_credit = df['credit'].sum()
        tolerance = 0.01  # 1% tolerance
        if abs(total_credit - total_conversions) > tolerance:
            logger.error(f"Efficiency axiom violated: {total_credit} vs {total_conversions}")
            return False
        
        logger.info("Attribution results validation passed")
        return True


def get_channel_info(config: Config, channel_id: str) -> Dict[str, Any]:
    """Get channel information from config"""
    channels = config.get('channels.definitions', [])
    for ch in channels:
        if ch['id'] == channel_id:
            return ch
    return {}


def get_macro_group(config: Config, channel_id: str) -> str:
    """Get macro group for a channel"""
    channel_info = get_channel_info(config, channel_id)
    return channel_info.get('macro_group', 'other')


def calculate_metrics(df: pd.DataFrame, conversions: int, costs: Dict[str, float]) -> Dict[str, float]:
    """Calculate key marketing metrics"""
    total_cost = sum(costs.values())
    
    metrics = {
        'total_conversions': conversions,
        'total_cost': total_cost,
        'cost_per_acquisition': total_cost / conversions if conversions > 0 else 0,
        'conversion_rate': conversions / len(df) if len(df) > 0 else 0
    }
    
    return metrics


if __name__ == "__main__":
    # Test configuration loading
    config = Config()
    print(f"Project: {config.get('project.name')}")
    print(f"Random seed: {config.get('project.random_seed')}")
    print(f"Number of conversions: {config.get('synthetic_data.n_conversions')}")
