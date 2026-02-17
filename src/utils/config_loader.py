"""
Configuration loader with Pydantic v2 validation.
All YAML config files are validated at startup. Provides typed access to
channel definitions, funnel stages, synthetic data params, and model hyperparams.
"""

import os
import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ChannelType(str, Enum):
    CLICK = "click"
    IMPRESSION = "impression"
    CALL = "call"
    AGENT = "agent"
    MAIL = "mail"
    EMAIL = "email"


class FunnelRole(str, Enum):
    AWARENESS = "awareness"
    CONSIDERATION = "consideration"
    CONVERSION = "conversion"


# ---------------------------------------------------------------------------
# Channel Configuration
# ---------------------------------------------------------------------------

class ChannelConfig(BaseModel):
    channel_id: str
    display_name: str
    channel_group: str
    sub_channels: List[str]
    touch_type: ChannelType
    lookback_window_days: int = Field(ge=1, le=180)
    time_decay_half_life_days: float = Field(ge=0.5, le=30.0)
    attribution_weight_default: float = Field(ge=0.0, le=1.0)
    erie_estimated_bind_share: float
    funnel_role: str
    data_source: str
    color: str


# ---------------------------------------------------------------------------
# Funnel Configuration
# ---------------------------------------------------------------------------

class FunnelStageConfig(BaseModel):
    stage_name: str
    erie_definition: str
    key_events: List[str]
    estimated_annual_volume: str
    conversion_rate_to_next: float


# ---------------------------------------------------------------------------
# Synthetic Data Configuration
# ---------------------------------------------------------------------------

class SyntheticDataConfig(BaseModel):
    total_journeys: int = Field(default=150_000, ge=10_000, le=1_000_000)
    simulation_months: int = Field(default=12, ge=1, le=36)
    quote_start_rate: float = Field(default=0.08, ge=0.01, le=0.20)
    bind_rate_from_quotes: float = Field(default=0.35, ge=0.10, le=0.60)
    avg_touchpoints_converting: float = Field(default=3.8, ge=1.0, le=15.0)
    avg_touchpoints_non_converting: float = Field(default=1.6, ge=1.0, le=10.0)
    agent_last_touch_pct: float = Field(default=0.60, ge=0.30, le=0.90)
    journey_duration_median_days: float = Field(default=8.0, ge=1.0, le=60.0)
    journey_duration_mean_days: float = Field(default=14.0, ge=1.0, le=90.0)
    seasonality_peaks: List[str] = ["March", "April", "September", "October"]
    random_seed: int = 42
    total_annual_spend: float = 100_000_000
    avg_premium_value: float = 1200.0


# ---------------------------------------------------------------------------
# Model Parameters Configuration
# ---------------------------------------------------------------------------

class ModelParamsConfig(BaseModel):
    # Shapley parameters
    shapley_min_coalition_obs: int = Field(default=30, ge=5, le=100)
    shapley_use_time_weights: bool = True
    shapley_mc_samples: int = Field(default=10000, ge=1000, le=100000)
    shapley_regression_mc_model: str = "xgboost"
    # CASV parameters
    casv_markov_order: int = Field(default=1, ge=1, le=3)
    # Markov parameters
    markov_order: int = Field(default=2, ge=1, le=3)
    markov_null_state_name: str = "NULL"
    markov_vmm_max_depth: int = Field(default=4, ge=1, le=6)
    # Rule-based parameters
    time_decay_half_life_click: float = 7.0
    time_decay_half_life_impression: float = 3.0
    position_based_first_weight: float = 0.40
    position_based_last_weight: float = 0.40
    # Deep learning parameters
    lstm_hidden_dim: int = 64
    lstm_num_layers: int = 2
    lstm_epochs: int = 50
    lstm_learning_rate: float = 0.001
    transformer_n_heads: int = 4
    transformer_d_model: int = 64
    transformer_n_layers: int = 2
    darnn_encoder_dim: int = 64
    darnn_decoder_dim: int = 64
    # General
    journey_max_touchpoints: int = 20
    session_timeout_minutes: int = 30


# ---------------------------------------------------------------------------
# Top-Level Demo Configuration
# ---------------------------------------------------------------------------

class DemoConfig(BaseModel):
    channels: List[ChannelConfig]
    funnel_stages: List[FunnelStageConfig]
    synthetic_data: SyntheticDataConfig
    model_params: ModelParamsConfig


# ---------------------------------------------------------------------------
# Loader Functions
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Walk up from this file to find the project root (contains config/)."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "config").is_dir():
            return current
        current = current.parent
    # Fallback to CWD
    return Path.cwd()


PROJECT_ROOT = _find_project_root()
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_yaml(filename: str) -> dict:
    """Load a YAML file from config/ directory."""
    filepath = CONFIG_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_channel_config() -> List[ChannelConfig]:
    """Load and validate channel taxonomy from channels.yaml."""
    data = _load_yaml("channels.yaml")
    return [ChannelConfig(**ch) for ch in data["channels"]]


def load_funnel_config() -> List[FunnelStageConfig]:
    """Load and validate funnel stages from funnel.yaml."""
    data = _load_yaml("funnel.yaml")
    return [FunnelStageConfig(**stage) for stage in data["funnel_stages"]]


def load_synthetic_data_config() -> SyntheticDataConfig:
    """Load synthetic data parameters from synthetic_data.yaml."""
    data = _load_yaml("synthetic_data.yaml")
    return SyntheticDataConfig(**data["synthetic_data"])


def load_model_params_config() -> ModelParamsConfig:
    """Load model hyperparameters from model_params.yaml."""
    data = _load_yaml("model_params.yaml")
    return ModelParamsConfig(**data["model_params"])


def load_demo_config() -> DemoConfig:
    """Load the complete demo configuration from all YAML files."""
    channels = load_channel_config()
    funnel_stages = load_funnel_config()
    synthetic_data = load_synthetic_data_config()
    model_params = load_model_params_config()
    return DemoConfig(
        channels=channels,
        funnel_stages=funnel_stages,
        synthetic_data=synthetic_data,
        model_params=model_params,
    )


def load_demo_narrative() -> dict:
    """Load demo narrative talking points from demo_narrative.yaml."""
    return _load_yaml("demo_narrative.yaml")


def get_channel_color_map(channels: List[ChannelConfig]) -> Dict[str, str]:
    """Return mapping of channel_id → hex color for chart usage."""
    return {ch.channel_id: ch.color for ch in channels}


def get_channel_display_names(channels: List[ChannelConfig]) -> Dict[str, str]:
    """Return mapping of channel_id → display_name."""
    return {ch.channel_id: ch.display_name for ch in channels}
