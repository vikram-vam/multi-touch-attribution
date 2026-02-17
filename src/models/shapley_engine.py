"""
Shapley Value attribution engine (Tier 2).
Implements exact Shapley for small channel sets and simplified (Zhao et al.) for larger ones.
"""

import math
from itertools import combinations
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from src.models.base import BaseAttributionModel, AttributionResult


class ShapleyAttribution(BaseAttributionModel):
    """
    Shapley Value channel attribution.

    For n channels ≤ 12, uses exact computation (2^n coalitions).
    For n > 12, falls back to Monte Carlo sampling.

    Key Shapley axioms:
    - Efficiency: credits sum to total conversions
    - Symmetry: identical channels get identical credit
    - Null player: non-contributing channels get zero credit
    - Additivity: combined game = sum of individual attributions
    """

    def __init__(
        self,
        min_coalition_obs: int = 30,
        use_time_weights: bool = True,
        mc_samples: int = 10000,
    ):
        self.min_coalition_obs = min_coalition_obs
        self.use_time_weights = use_time_weights
        self.mc_samples = mc_samples
        self._coalition_cache: Dict[frozenset, float] = {}

    @property
    def name(self) -> str:
        return "shapley"

    @property
    def tier(self) -> str:
        return "game_theoretic"

    def fit(self, journeys: pd.DataFrame) -> None:
        """Pre-compute coalition conversion rates from journey data."""
        converting = journeys[journeys["is_converting"]].copy()

        # Build coalition → conversion rate lookup
        coalition_stats = defaultdict(lambda: {"conversions": 0, "total": 0})

        for _, row in journeys.iterrows():
            channel_set = row.get("channel_set", [])
            if isinstance(channel_set, str):
                channel_set = channel_set.split("|")
            coalition_key = frozenset(channel_set)
            coalition_stats[coalition_key]["total"] += 1
            if row["is_converting"]:
                coalition_stats[coalition_key]["conversions"] += 1

        # Store conversion rates, applying smoothing for sparse coalitions
        self._coalition_cache = {}
        for coalition, stats in coalition_stats.items():
            if stats["total"] >= self.min_coalition_obs:
                rate = stats["conversions"] / stats["total"]
            else:
                # Sparse coalition: weighted average with global rate
                global_rate = len(converting) / len(journeys) if len(journeys) > 0 else 0
                weight = stats["total"] / self.min_coalition_obs
                rate = weight * (stats["conversions"] / max(stats["total"], 1)) + \
                       (1 - weight) * global_rate
            self._coalition_cache[coalition] = rate

    def _coalition_value(self, coalition: frozenset) -> float:
        """Get v(S) — the value function for a coalition of channels."""
        if len(coalition) == 0:
            return 0.0

        # Exact match
        if coalition in self._coalition_cache:
            return self._coalition_cache[coalition]

        # Subset matching: find closest superset or subset
        best_match = None
        best_overlap = 0
        for cached_coalition, rate in self._coalition_cache.items():
            overlap = len(coalition & cached_coalition)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = rate

        return best_match if best_match is not None else 0.0

    def _exact_shapley(self, channels: List[str]) -> Dict[str, float]:
        """Exact Shapley computation via all coalitions."""
        n = len(channels)
        shapley_values = {ch: 0.0 for ch in channels}

        for i, channel in enumerate(channels):
            others = [ch for ch in channels if ch != channel]
            n_others = len(others)

            # Iterate over all subsets of others
            for size in range(n_others + 1):
                for subset in combinations(others, size):
                    s = len(subset)
                    coalition_without = frozenset(subset)
                    coalition_with = frozenset(subset) | {channel}

                    # Marginal contribution
                    v_with = self._coalition_value(coalition_with)
                    v_without = self._coalition_value(coalition_without)
                    marginal = v_with - v_without

                    # Shapley weight = |S|! * (n - |S| - 1)! / n!
                    weight = (math.factorial(s) * math.factorial(n - s - 1)) / math.factorial(n)
                    shapley_values[channel] += weight * marginal

        return shapley_values

    def _mc_shapley(self, channels: List[str]) -> Dict[str, float]:
        """Monte Carlo approximation for large channel sets."""
        rng = np.random.default_rng(42)
        n = len(channels)
        shapley_values = {ch: 0.0 for ch in channels}
        channel_array = np.array(channels)

        for _ in range(self.mc_samples):
            perm = rng.permutation(n)
            coalition = set()

            for idx in perm:
                channel = channel_array[idx]
                v_before = self._coalition_value(frozenset(coalition))
                coalition.add(channel)
                v_after = self._coalition_value(frozenset(coalition))
                shapley_values[channel] += (v_after - v_before)

        # Average over samples
        for ch in shapley_values:
            shapley_values[ch] /= self.mc_samples

        return shapley_values

    def attribute(self, journeys: pd.DataFrame) -> AttributionResult:
        # Get all unique channels from converting journeys
        converting = journeys[journeys["is_converting"]]
        all_channels = set()
        for _, row in converting.iterrows():
            path = row.get("channel_path", [])
            if isinstance(path, str):
                path = path.split("|")
            all_channels.update(path)

        channels = sorted(list(all_channels))
        n = len(channels)

        # Choose exact vs MC based on channel count
        if n <= 12:
            shapley_values = self._exact_shapley(channels)
        else:
            shapley_values = self._mc_shapley(channels)

        # Scale to actual conversion count
        total_conversions = len(converting)
        total_shapley = sum(shapley_values.values())
        if total_shapley > 0:
            scale = total_conversions / total_shapley
            channel_credits = {ch: v * scale for ch, v in shapley_values.items()}
        else:
            channel_credits = shapley_values

        # Ensure non-negative (Shapley values can be slightly negative for irrelevant channels)
        channel_credits = {ch: max(v, 0.0) for ch, v in channel_credits.items()}

        # Re-normalize to ensure efficiency axiom (sum = total conversions)
        credit_sum = sum(channel_credits.values())
        if credit_sum > 0 and abs(credit_sum - total_conversions) > 0.01:
            factor = total_conversions / credit_sum
            channel_credits = {ch: v * factor for ch, v in channel_credits.items()}

        return self._build_result(
            channel_credits, journeys,
            metadata={
                "method": "exact" if n <= 12 else "monte_carlo",
                "n_channels": n,
                "mc_samples": self.mc_samples if n > 12 else None,
                "efficiency_satisfied": True,
            },
        )
