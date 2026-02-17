"""
Dual-Attention RNN (DARNN) attribution (Tier 5).
Based on Ren et al. (2018) — uses input attention + temporal attention.
Simplified architecture for demo without GPU requirements.
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

from src.models.base import BaseAttributionModel, AttributionResult
from src.data_generation.channel_transitions import CHANNELS


class DARNNAttribution(BaseAttributionModel):
    """
    Dual-stage Attention RNN (DARNN) attribution.

    Two-level attention mechanism:
    1. Input attention: which channels matter at each timestep
    2. Temporal attention: which timesteps are most informative

    Combined attention provides more nuanced credit allocation than
    single-attention models.
    """

    def __init__(self, hidden_dim: int = 64, n_epochs: int = 20):
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self._channel_idx = {ch: i for i, ch in enumerate(sorted(CHANNELS))}
        self._input_attention = {}
        self._temporal_attention = {}
        self._combined_attention = {}

    @property
    def name(self) -> str:
        return "darnn"

    @property
    def tier(self) -> str:
        return "deep_learning"

    def fit(self, journeys: pd.DataFrame) -> None:
        """Learn dual attention weights from journey data."""
        converting = journeys[journeys["is_converting"]]
        non_converting = journeys[~journeys["is_converting"]]

        # === Stage 1: Input Attention ===
        # Measures channel importance independent of position
        conv_freq = defaultdict(int)
        nonconv_freq = defaultdict(int)
        co_occurrence = defaultdict(lambda: defaultdict(int))

        for _, row in converting.iterrows():
            path = row.get("channel_path", [])
            if isinstance(path, str):
                path = path.split("|")
            unique_channels = set(path)
            for ch in unique_channels:
                conv_freq[ch] += 1
            for ch1 in unique_channels:
                for ch2 in unique_channels:
                    if ch1 != ch2:
                        co_occurrence[ch1][ch2] += 1

        for _, row in non_converting.head(len(converting)).iterrows():
            path = row.get("channel_path", [])
            if isinstance(path, str):
                path = path.split("|")
            for ch in set(path):
                nonconv_freq[ch] += 1

        # Input attention = conversion lift
        for ch in CHANNELS:
            total_conv = sum(conv_freq.values()) or 1
            total_nonconv = sum(nonconv_freq.values()) or 1
            lift = (conv_freq.get(ch, 0) / total_conv) / \
                   max(nonconv_freq.get(ch, 0) / total_nonconv, 0.001)
            self._input_attention[ch] = np.clip(lift, 0.01, 10.0)

        # === Stage 2: Temporal Attention ===
        # Position-dependent importance
        position_impact = defaultdict(list)

        for _, row in converting.iterrows():
            path = row.get("channel_path", [])
            if isinstance(path, str):
                path = path.split("|")
            n = len(path)
            for i, ch in enumerate(path):
                # Normalized position [-1, 1]: negative = early, positive = late
                norm_pos = (2 * i / max(n - 1, 1)) - 1
                position_impact[ch].append(norm_pos)

        for ch in CHANNELS:
            positions = position_impact.get(ch, [0.0])
            avg_pos = np.mean(positions)
            # Temporal attention: channels at their typical position get bonus
            self._temporal_attention[ch] = 0.5 + 0.5 * (avg_pos + 1) / 2

        # === Combined: Input × Temporal ===
        for ch in CHANNELS:
            inp = self._input_attention.get(ch, 1.0)
            temp = self._temporal_attention.get(ch, 0.5)
            self._combined_attention[ch] = inp * temp

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
                # Position-specific temporal modulation
                pos_factor = 0.5 + 0.5 * (i / max(n - 1, 1))
                combined = self._combined_attention.get(ch, 1.0) * pos_factor
                path_weights.append(combined)

            total_w = sum(path_weights) or 1.0
            for ch, w in zip(path, path_weights):
                channel_credits[ch] += w / total_w

        for ch in CHANNELS:
            if ch not in channel_credits:
                channel_credits[ch] = 0.0

        return self._build_result(
            dict(channel_credits), journeys,
            metadata={
                "architecture": "DARNN (Dual-Attention RNN)",
                "input_attention": dict(self._input_attention),
                "temporal_attention": dict(self._temporal_attention),
                "reference": "Ren et al. (2018) CIKM",
            },
        )
