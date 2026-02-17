"""
Model registry — single access point for all attribution models.
"""

from typing import Dict, List, Optional

from src.models.base import BaseAttributionModel, AttributionResult
from src.models.rule_based import (
    FirstTouchAttribution, LastTouchAttribution, LinearAttribution,
    TimeDecayAttribution, PositionBasedAttribution,
)
from src.models.shapley_engine import ShapleyAttribution
from src.models.shapley_casv import CASVAttribution
from src.models.markov_chain import MarkovChainAttribution
from src.models.logistic_attribution import LogisticAttribution
from src.models.survival_attribution import SurvivalAttribution
from src.models.deep_learning import (
    LSTMAttribution, DARNNAttribution, TransformerAttribution, CausalMTAAttribution,
)
from src.models.ensemble import EnsembleAttribution


# All available models by tier
MODEL_REGISTRY: Dict[str, dict] = {
    # Tier 1: Rule-Based
    "first_touch": {
        "class": FirstTouchAttribution,
        "tier": "rule_based",
        "display_name": "First-Touch",
        "description": "100% credit to first channel",
        "starred": False,
    },
    "last_touch": {
        "class": LastTouchAttribution,
        "tier": "rule_based",
        "display_name": "Last-Touch (GA4 Default)",
        "description": "100% credit to last channel — matches GA4",
        "starred": True,
    },
    "linear": {
        "class": LinearAttribution,
        "tier": "rule_based",
        "display_name": "Linear",
        "description": "Equal credit to all touchpoints",
        "starred": False,
    },
    "time_decay": {
        "class": TimeDecayAttribution,
        "tier": "rule_based",
        "display_name": "Time-Decay",
        "description": "Exponential decay favoring recent touches",
        "starred": False,
    },
    "position_based": {
        "class": PositionBasedAttribution,
        "tier": "rule_based",
        "display_name": "Position-Based (U-Shaped)",
        "description": "40% first + 40% last + 20% middle",
        "starred": False,
    },
    # Tier 2: Game-Theoretic
    "shapley": {
        "class": ShapleyAttribution,
        "tier": "game_theoretic",
        "display_name": "Shapley Value",
        "description": "Fair allocation via cooperative game theory",
        "starred": True,
    },
    "casv": {
        "class": CASVAttribution,
        "tier": "game_theoretic",
        "display_name": "CASV",
        "description": "Causal Additive Shapley Values",
        "starred": False,
    },
    # Tier 3: Probabilistic
    "markov_order_1": {
        "class": lambda: MarkovChainAttribution(order=1),
        "tier": "probabilistic",
        "display_name": "Markov Chain (1st Order)",
        "description": "First-order transition probabilities",
        "starred": False,
    },
    "markov_order_2": {
        "class": lambda: MarkovChainAttribution(order=2),
        "tier": "probabilistic",
        "display_name": "Markov Chain (2nd Order)",
        "description": "Second-order captures sequential patterns",
        "starred": True,
    },
    "markov_order_3": {
        "class": lambda: MarkovChainAttribution(order=3),
        "tier": "probabilistic",
        "display_name": "Markov Chain (3rd Order)",
        "description": "Third-order for rich path patterns",
        "starred": False,
    },
    # Tier 4: Statistical
    "logistic_regression": {
        "class": LogisticAttribution,
        "tier": "statistical",
        "display_name": "Logistic Regression",
        "description": "Incremental effect via regression coefficients",
        "starred": False,
    },
    "survival_hazard": {
        "class": SurvivalAttribution,
        "tier": "statistical",
        "display_name": "Survival / Hazard",
        "description": "Cox PH-inspired time-to-conversion model",
        "starred": False,
    },
    # Tier 5: Deep Learning
    "lstm": {
        "class": LSTMAttribution,
        "tier": "deep_learning",
        "display_name": "LSTM",
        "description": "Sequence model with attention",
        "starred": False,
    },
    "darnn": {
        "class": DARNNAttribution,
        "tier": "deep_learning",
        "display_name": "DARNN",
        "description": "Dual-stage attention RNN",
        "starred": False,
    },
    "transformer": {
        "class": TransformerAttribution,
        "tier": "deep_learning",
        "display_name": "Transformer",
        "description": "Multi-head self-attention (SOTA)",
        "starred": True,
    },
    "causal_mta": {
        "class": CausalMTAAttribution,
        "tier": "deep_learning",
        "display_name": "CausalMTA",
        "description": "Causal deep learning with IPW debiasing",
        "starred": False,
    },
    # Tier 6: Meta-model
    "ensemble": {
        "class": EnsembleAttribution,
        "tier": "meta_model",
        "display_name": "Weighted Ensemble",
        "description": "Shapley (45%) + Markov (30%) + Logistic (25%)",
        "starred": True,
    },
}


def get_model(name: str) -> BaseAttributionModel:
    """Instantiate a model by registry name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")

    entry = MODEL_REGISTRY[name]
    cls = entry["class"]

    if callable(cls) and not isinstance(cls, type):
        return cls()  # Lambda factory
    return cls()


def get_all_models(exclude_tiers: List[str] = None) -> Dict[str, BaseAttributionModel]:
    """Instantiate all models, optionally excluding tiers."""
    models = {}
    exclude = set(exclude_tiers or [])
    for name, entry in MODEL_REGISTRY.items():
        if entry["tier"] not in exclude and name != "ensemble":
            models[name] = get_model(name)
    return models


def get_starred_models() -> List[str]:
    """Return names of starred/highlighted models for the UI."""
    return [name for name, entry in MODEL_REGISTRY.items() if entry.get("starred", False)]


def get_model_display_names() -> Dict[str, str]:
    """Return mapping of model_name → display_name."""
    return {name: entry["display_name"] for name, entry in MODEL_REGISTRY.items()}


def get_models_by_tier() -> Dict[str, List[str]]:
    """Group model names by tier."""
    tiers = {}
    for name, entry in MODEL_REGISTRY.items():
        tier = entry["tier"]
        if tier not in tiers:
            tiers[tier] = []
        tiers[tier].append(name)
    return tiers
