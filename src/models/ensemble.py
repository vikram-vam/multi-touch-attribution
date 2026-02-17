"""
Weighted ensemble attribution (Tier 6, Section 6.15).
Combines Shapley (45%) + Markov 2nd-order (30%) + Logistic (25%).
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.base import BaseAttributionModel, AttributionResult
from src.data_generation.channel_transitions import CHANNELS


class EnsembleAttribution(BaseAttributionModel):
    """
    Production-grade ensemble that triangulates across 3 modeling paradigms:
    - Game-theoretic (Shapley): fairness + cooperative credit
    - Probabilistic (Markov): sequential patterns
    - Statistical (Logistic): incremental effect

    Weighted combination: Shapley 0.45, Markov 0.30, Logistic 0.25.
    """

    DEFAULT_WEIGHTS = {
        "shapley": 0.45,
        "markov_order_2": 0.30,
        "logistic_regression": 0.25,
    }

    def __init__(
        self,
        model_results: Optional[Dict[str, AttributionResult]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.model_results = model_results or {}
        self.weights = weights or self.DEFAULT_WEIGHTS

    @property
    def name(self) -> str:
        return "ensemble"

    @property
    def tier(self) -> str:
        return "meta_model"

    def fit(self, journeys: pd.DataFrame) -> None:
        pass  # Ensemble operates on pre-computed results

    def add_result(self, result: AttributionResult) -> None:
        """Add a model's result to the ensemble."""
        self.model_results[result.model_name] = result

    def attribute(self, journeys: pd.DataFrame) -> AttributionResult:
        converting = journeys[journeys["is_converting"]]
        total_conversions = len(converting)

        # Weighted credit combination
        combined_credits = {ch: 0.0 for ch in CHANNELS}

        total_weight_applied = 0.0
        for model_name, weight in self.weights.items():
            if model_name in self.model_results:
                result = self.model_results[model_name]
                for ch in CHANNELS:
                    pct = result.channel_credit_pct.get(ch, 0.0)
                    combined_credits[ch] += pct * weight
                total_weight_applied += weight

        # Normalize by total weight (in case not all models available)
        if total_weight_applied > 0:
            for ch in combined_credits:
                combined_credits[ch] /= total_weight_applied

        # Scale to conversion count
        channel_credits = {
            ch: pct * total_conversions for ch, pct in combined_credits.items()
        }

        return self._build_result(
            channel_credits, journeys,
            metadata={
                "weights": self.weights,
                "models_used": list(self.model_results.keys()),
                "total_weight_applied": total_weight_applied,
            },
        )
