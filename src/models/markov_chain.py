"""
Markov Chain attribution model (Tier 3).
Supports orders 1-3 with Dirichlet smoothing and removal effect calculation.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.models.base import BaseAttributionModel, AttributionResult


class MarkovChainAttribution(BaseAttributionModel):
    """
    Fixed-order Markov Chain attribution via removal effect.

    For each channel, attribution = P(conversion_baseline) - P(conversion_without_channel).
    Results are normalized to sum to total conversions.

    Higher-order chains capture sequential patterns (e.g., Display → Search → Agent).
    """

    def __init__(self, order: int = 2, null_state: str = "NULL",
                 smoothing_alpha: float = 0.01):
        self.order = order
        self.null_state = null_state
        self.smoothing_alpha = smoothing_alpha
        self._transition_matrix: Dict = {}
        self._all_states: Set[str] = set()
        self._start_state = "START"
        self._conv_state = "CONVERSION"

    @property
    def name(self) -> str:
        return f"markov_order_{self.order}"

    @property
    def tier(self) -> str:
        return "probabilistic"

    def fit(self, journeys: pd.DataFrame) -> None:
        """Build transition matrices from journey data."""
        self._transition_matrix = defaultdict(lambda: defaultdict(float))
        self._all_states = set()

        for _, row in journeys.iterrows():
            path = row.get("channel_path", [])
            if isinstance(path, str):
                path = path.split("|")

            is_converting = row["is_converting"]

            # Add start and end states
            full_path = [self._start_state] + list(path)
            if is_converting:
                full_path.append(self._conv_state)
            else:
                full_path.append(self.null_state)

            self._all_states.update(full_path)

            # Build n-gram transitions
            for i in range(len(full_path) - 1):
                if self.order == 1:
                    state_from = full_path[i]
                else:
                    # Higher order: use tuple of previous n states
                    start_idx = max(0, i - self.order + 1)
                    state_from = tuple(full_path[start_idx:i + 1])

                state_to = full_path[i + 1]
                self._transition_matrix[state_from][state_to] += 1

        # Normalize with Dirichlet smoothing
        for state_from in self._transition_matrix:
            total = sum(self._transition_matrix[state_from].values())
            n_targets = len(self._transition_matrix[state_from])
            for state_to in self._transition_matrix[state_from]:
                self._transition_matrix[state_from][state_to] = (
                    (self._transition_matrix[state_from][state_to] + self.smoothing_alpha) /
                    (total + self.smoothing_alpha * n_targets)
                )

    def _absorption_probability(
        self,
        removed_channel: Optional[str] = None,
        max_steps: int = 100,
    ) -> float:
        """
        Compute P(reach CONVERSION from START) using iterative matrix power.

        If removed_channel is specified, all transitions through that channel
        are redirected to NULL (removal effect).
        """
        # Simple simulation-based approach
        n_simulations = 5000
        rng = np.random.default_rng(42)
        conversions = 0

        for _ in range(n_simulations):
            current = self._start_state if self.order == 1 else (self._start_state,)
            for step in range(max_steps):
                if self.order == 1:
                    state = current
                else:
                    state = current

                transitions = dict(self._transition_matrix.get(state, {}))
                if not transitions:
                    break

                # Remove channel if needed
                if removed_channel and removed_channel in transitions:
                    # Redirect to NULL
                    null_prob = transitions.pop(removed_channel, 0)
                    transitions[self.null_state] = transitions.get(self.null_state, 0) + null_prob

                    # Re-normalize
                    total = sum(transitions.values())
                    if total > 0:
                        transitions = {k: v / total for k, v in transitions.items()}

                # Sample next state
                targets = list(transitions.keys())
                probs = np.array(list(transitions.values()))
                if probs.sum() == 0:
                    break
                probs /= probs.sum()

                next_state = targets[int(rng.choice(len(targets), p=probs))]

                if next_state == self._conv_state:
                    conversions += 1
                    break
                elif next_state == self.null_state:
                    break

                # Update current state
                if self.order == 1:
                    current = next_state
                else:
                    if isinstance(current, tuple):
                        current = current[-(self.order - 1):] + (next_state,)
                    else:
                        current = (next_state,)

        return conversions / n_simulations

    def attribute(self, journeys: pd.DataFrame) -> AttributionResult:
        # Get all channels from converting journeys
        converting = journeys[journeys["is_converting"]]
        all_channels = set()
        for _, row in converting.iterrows():
            path = row.get("channel_path", [])
            if isinstance(path, str):
                path = path.split("|")
            all_channels.update(path)

        channels = sorted(list(all_channels))

        # Baseline conversion probability
        p_baseline = self._absorption_probability()

        # Removal effects
        removal_effects = {}
        for channel in channels:
            p_without = self._absorption_probability(removed_channel=channel)
            removal_effect = max(0, p_baseline - p_without)
            removal_effects[channel] = removal_effect

        # Normalize to total conversions
        total_conversions = len(converting)
        total_effect = sum(removal_effects.values())

        if total_effect > 0:
            channel_credits = {
                ch: (effect / total_effect) * total_conversions
                for ch, effect in removal_effects.items()
            }
        else:
            # Fallback: uniform
            channel_credits = {ch: total_conversions / len(channels) for ch in channels}

        return self._build_result(
            channel_credits, journeys,
            metadata={
                "order": self.order,
                "baseline_p_conversion": p_baseline,
                "removal_effects": removal_effects,
            },
        )
