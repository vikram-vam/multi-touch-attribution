"""
Variable-Order Markov / MTD attribution (Tier 3).
Automatically selects the optimal Markov order per channel transition
using BIC-based model selection.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.models.base import BaseAttributionModel, AttributionResult
from src.models.markov_chain import MarkovChainAttribution


class VariableOrderMarkovAttribution(BaseAttributionModel):
    """
    Variable-Order Markov Chain attribution.

    Instead of a fixed order (1, 2, or 3), this model selects the optimal
    order for each state transition using BIC (Bayesian Information Criterion).

    The Mixture Transition Distribution (MTD) approach combines information
    from multiple orders with learned mixing weights:
        P(next | history) = Σ_k λ_k * P_k(next | history_k)
    where λ_k are mixing weights and P_k is the k-th order model.
    """

    def __init__(self, max_order: int = 3, smoothing_alpha: float = 0.01):
        self.max_order = max_order
        self.smoothing_alpha = smoothing_alpha
        self._order_models: Dict[int, MarkovChainAttribution] = {}
        self._mixing_weights: Dict[int, float] = {}
        self._optimal_orders: Dict[str, int] = {}

    @property
    def name(self) -> str:
        return "markov_variable_order"

    @property
    def tier(self) -> str:
        return "probabilistic"

    def _compute_bic(self, log_likelihood: float, n_params: int, n_obs: int) -> float:
        """Bayesian Information Criterion: BIC = -2*LL + k*ln(n)."""
        return -2 * log_likelihood + n_params * np.log(max(n_obs, 1))

    def _compute_log_likelihood(self, model: MarkovChainAttribution,
                                 journeys: pd.DataFrame) -> float:
        """Compute log-likelihood of observed data under fitted model."""
        ll = 0.0
        n = 0

        for _, row in journeys.iterrows():
            path = row.get("channel_path", [])
            if isinstance(path, str):
                path = path.split("|")

            is_converting = row["is_converting"]
            full_path = [model._start_state] + list(path)
            full_path.append(model._conv_state if is_converting else model.null_state)

            for i in range(len(full_path) - 1):
                if model.order == 1:
                    state_from = full_path[i]
                else:
                    start_idx = max(0, i - model.order + 1)
                    state_from = tuple(full_path[start_idx:i + 1])

                state_to = full_path[i + 1]
                transitions = model._transition_matrix.get(state_from, {})
                prob = transitions.get(state_to, self.smoothing_alpha)
                ll += np.log(max(prob, 1e-10))
                n += 1

        return ll

    def fit(self, journeys: pd.DataFrame) -> None:
        """Fit models at each order and compute optimal mixing weights."""
        bic_scores = {}

        for order in range(1, self.max_order + 1):
            model = MarkovChainAttribution(
                order=order, smoothing_alpha=self.smoothing_alpha
            )
            model.fit(journeys)
            self._order_models[order] = model

            ll = self._compute_log_likelihood(model, journeys.head(2000))
            n_params = len(model._transition_matrix) * 5  # approximate
            n_obs = len(journeys)
            bic = self._compute_bic(ll, n_params, n_obs)
            bic_scores[order] = bic

        # Compute mixing weights using inverse BIC (lower BIC = better model)
        min_bic = min(bic_scores.values())
        delta_bic = {k: v - min_bic for k, v in bic_scores.items()}

        # Softmax-style weighting: w_k ∝ exp(-Δ_BIC/2)
        raw_weights = {k: np.exp(-d / 2) for k, d in delta_bic.items()}
        total_weight = sum(raw_weights.values())
        self._mixing_weights = {k: w / total_weight for k, w in raw_weights.items()}

        # Identify best single order
        self._best_order = min(bic_scores, key=bic_scores.get)

    def attribute(self, journeys: pd.DataFrame) -> AttributionResult:
        """Compute weighted ensemble of removal effects across orders."""
        converting = journeys[journeys["is_converting"]]
        all_channels = set()
        for _, row in converting.iterrows():
            path = row.get("channel_path", [])
            if isinstance(path, str):
                path = path.split("|")
            all_channels.update(path)
        channels = sorted(list(all_channels))

        total_conversions = len(converting)

        # Weighted removal effect across orders
        blended_removal = {ch: 0.0 for ch in channels}

        for order, model in self._order_models.items():
            weight = self._mixing_weights.get(order, 0)
            if weight < 0.01:
                continue

            p_baseline = model._absorption_probability()

            for channel in channels:
                p_without = model._absorption_probability(removed_channel=channel)
                removal_effect = max(0, p_baseline - p_without)
                blended_removal[channel] += weight * removal_effect

        # Normalize to total conversions
        total_effect = sum(blended_removal.values())
        if total_effect > 0:
            channel_credits = {
                ch: (effect / total_effect) * total_conversions
                for ch, effect in blended_removal.items()
            }
        else:
            channel_credits = {ch: total_conversions / len(channels) for ch in channels}

        return self._build_result(
            channel_credits, journeys,
            metadata={
                "mixing_weights": self._mixing_weights,
                "best_single_order": self._best_order,
                "method": "variable_order_mtd",
            },
        )
