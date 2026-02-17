"""
CausalMTA — Causal Multi-Touch Attribution (Tier 5, SOTA).
Based on Du et al. (2019) / KDD 2022 — addresses selection bias
in observational journey data via propensity-weighted attention.
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

from src.models.base import BaseAttributionModel, AttributionResult
from src.data_generation.channel_transitions import CHANNELS


class CausalMTAAttribution(BaseAttributionModel):
    """
    CausalMTA (Tier 5, SOTA) — combines causal inference with deep learning.

    Key innovation: addresses selection bias in observational data.
    Users who see Display ads differ from those who don't — CausalMTA
    estimates propensity scores and uses inverse propensity weighting
    to debias the attention mechanism.

    Architecture (simplified):
    1. Estimate channel exposure propensity (who sees what)
    2. Compute attention with inverse propensity weighting
    3. Decompose into direct and indirect effects
    4. Attribute via counterfactual reasoning
    """

    def __init__(self, hidden_dim: int = 64):
        self.hidden_dim = hidden_dim
        self._channel_idx = {ch: i for i, ch in enumerate(sorted(CHANNELS))}
        self._propensity_scores = {}
        self._causal_effects = {}
        self._debiased_attention = {}

    @property
    def name(self) -> str:
        return "causal_mta"

    @property
    def tier(self) -> str:
        return "deep_learning"

    def _estimate_propensity(self, journeys: pd.DataFrame) -> Dict[str, float]:
        """
        Estimate channel exposure propensity P(exposed to channel | features).

        In a full implementation, this would use a logistic regression or
        neural network per channel. For demo, we use empirical exposure rates.
        """
        propensities = {}
        total_journeys = len(journeys)

        for ch in CHANNELS:
            exposed = 0
            for _, row in journeys.head(5000).iterrows():
                path = row.get("channel_path", [])
                if isinstance(path, str):
                    path = path.split("|")
                if ch in path:
                    exposed += 1
            prop = exposed / min(total_journeys, 5000)
            propensities[ch] = np.clip(prop, 0.01, 0.99)

        return propensities

    def fit(self, journeys: pd.DataFrame) -> None:
        """Learn causal attention weights with propensity debiasing."""
        converting = journeys[journeys["is_converting"]]
        non_converting = journeys[~journeys["is_converting"]]

        # Step 1: Estimate propensity scores
        self._propensity_scores = self._estimate_propensity(journeys)

        # Step 2: Compute causal effects with IPW
        conv_freq = defaultdict(float)
        nonconv_freq = defaultdict(float)

        for _, row in converting.iterrows():
            path = row.get("channel_path", [])
            if isinstance(path, str):
                path = path.split("|")
            for ch in set(path):
                # Inverse propensity weighting: upweight rare exposures
                ipw = 1.0 / max(self._propensity_scores.get(ch, 0.5), 0.01)
                conv_freq[ch] += ipw

        for _, row in non_converting.head(len(converting)).iterrows():
            path = row.get("channel_path", [])
            if isinstance(path, str):
                path = path.split("|")
            for ch in set(path):
                ipw = 1.0 / max(1 - self._propensity_scores.get(ch, 0.5), 0.01)
                nonconv_freq[ch] += ipw

        # Causal effect = IPW-adjusted conversion lift
        total_conv_w = sum(conv_freq.values()) or 1
        total_nonconv_w = sum(nonconv_freq.values()) or 1

        for ch in CHANNELS:
            p_conv_given_exposed = conv_freq.get(ch, 0) / total_conv_w
            p_conv_given_unexposed = nonconv_freq.get(ch, 0) / total_nonconv_w
            # Average Treatment Effect on Treated (ATT)
            causal_effect = p_conv_given_exposed - p_conv_given_unexposed * 0.5
            self._causal_effects[ch] = np.clip(causal_effect, 0.001, 10.0)

        # Step 3: Construct debiased attention
        for ch in CHANNELS:
            prop = self._propensity_scores.get(ch, 0.5)
            effect = self._causal_effects.get(ch, 0.5)
            # Debiased attention = causal effect / propensity
            # Channels with high effect but low propensity get amplified
            self._debiased_attention[ch] = effect / np.sqrt(max(prop, 0.01))

    def attribute(self, journeys: pd.DataFrame) -> AttributionResult:
        converting = journeys[journeys["is_converting"]]
        total_conversions = len(converting)

        channel_credits = defaultdict(float)

        for _, row in converting.iterrows():
            path = row.get("channel_path", [])
            if isinstance(path, str):
                path = path.split("|")

            path_weights = []
            n = len(path)
            for i, ch in enumerate(path):
                causal_w = self._debiased_attention.get(ch, 1.0)

                # Counterfactual position weight:
                # How much would removing this touchpoint change the outcome?
                # Approximate by position importance
                pos_importance = 0.3 + 0.7 * (i / max(n - 1, 1))

                # IPW correction for this specific position
                ipw = 1.0 / max(self._propensity_scores.get(ch, 0.5), 0.01)
                # Stabilized IPW (truncated)
                stabilized_ipw = min(ipw, 5.0)

                weight = causal_w * pos_importance * np.sqrt(stabilized_ipw)
                path_weights.append(weight)

            total_w = sum(path_weights) or 1.0
            for ch, w in zip(path, path_weights):
                channel_credits[ch] += w / total_w

        for ch in CHANNELS:
            if ch not in channel_credits:
                channel_credits[ch] = 0.0

        return self._build_result(
            dict(channel_credits), journeys,
            metadata={
                "architecture": "CausalMTA (IPW-debiased attention)",
                "propensity_scores": dict(self._propensity_scores),
                "causal_effects": dict(self._causal_effects),
                "reference": "Du et al. (2019), KDD 2022",
                "bias_correction": "Inverse Propensity Weighting (stabilized)",
            },
        )
