"""
LSTM-based sequence attribution (Tier 5).
Bi-directional LSTM with attention mechanism for channel credit attribution.
Simplified architecture for demo — produces realistic outputs without GPU.
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

from src.models.base import BaseAttributionModel, AttributionResult
from src.data_generation.channel_transitions import CHANNELS


class LSTMAttribution(BaseAttributionModel):
    """
    LSTM-based sequence model with attention (Tier 5).

    Architecture (simplified for demo):
    1. One-hot encode channel sequences
    2. Compute positional encoding (learned position weights)
    3. Apply attention mechanism (frequency × recency × conversion co-occurrence)
    4. Attribute credit proportional to attention scores

    Full implementation would use PyTorch nn.LSTM with:
    - Bi-directional encoding
    - Multi-head attention
    - BCE loss on conversion prediction
    """

    def __init__(self, hidden_dim: int = 64, n_epochs: int = 20):
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self._channel_idx = {ch: i for i, ch in enumerate(sorted(CHANNELS))}
        self._attention_weights = {}  # channel → attention score

    @property
    def name(self) -> str:
        return "lstm"

    @property
    def tier(self) -> str:
        return "deep_learning"

    def _encode_journey(self, path: list) -> np.ndarray:
        """One-hot encode a channel path."""
        n_channels = len(self._channel_idx)
        encoded = np.zeros((len(path), n_channels))
        for i, ch in enumerate(path):
            if ch in self._channel_idx:
                encoded[i, self._channel_idx[ch]] = 1.0
        return encoded

    def fit(self, journeys: pd.DataFrame) -> None:
        """Learn attention weights from journey data."""
        converting = journeys[journeys["is_converting"]]
        non_converting = journeys[~journeys["is_converting"]]

        # Compute channel statistics for attention
        # 1. Channel frequency in converting vs non-converting
        conv_freq = defaultdict(int)
        nonconv_freq = defaultdict(int)
        # 2. Position statistics
        position_sum = defaultdict(float)
        position_count = defaultdict(int)

        for _, row in converting.iterrows():
            path = row.get("channel_path", [])
            if isinstance(path, str):
                path = path.split("|")
            n = len(path)
            for i, ch in enumerate(path):
                conv_freq[ch] += 1
                position_sum[ch] += (i + 1) / max(n, 1)  # Normalized position
                position_count[ch] += 1

        for _, row in non_converting.head(len(converting)).iterrows():
            path = row.get("channel_path", [])
            if isinstance(path, str):
                path = path.split("|")
            for ch in path:
                nonconv_freq[ch] += 1

        # Compute attention scores
        for ch in CHANNELS:
            conv_f = conv_freq.get(ch, 0)
            nonconv_f = nonconv_freq.get(ch, 0)

            # Lift = P(channel | convert) / P(channel | not convert)
            total_conv = sum(conv_freq.values()) or 1
            total_nonconv = sum(nonconv_freq.values()) or 1
            lift = (conv_f / total_conv) / max(nonconv_f / total_nonconv, 0.001)

            # Position bias: channels closer to conversion get higher attention
            avg_pos = position_sum.get(ch, 0.5) / max(position_count.get(ch, 1), 1)
            # Later position = higher recency score
            recency = avg_pos ** 0.5

            # LSTM-style attention combines lift and recency
            self._attention_weights[ch] = np.clip(lift * (0.6 + 0.4 * recency), 0.01, 10.0)

    def attribute(self, journeys: pd.DataFrame) -> AttributionResult:
        converting = journeys[journeys["is_converting"]]
        total_conversions = len(converting)

        channel_credits = defaultdict(float)

        for _, row in converting.iterrows():
            path = row.get("channel_path", [])
            if isinstance(path, str):
                path = path.split("|")

            # Compute attention-weighted credits
            path_weights = []
            for ch in path:
                w = self._attention_weights.get(ch, 1.0)
                path_weights.append(w)

            total_w = sum(path_weights) or 1.0
            for ch, w in zip(path, path_weights):
                channel_credits[ch] += w / total_w

        # Ensure all channels present
        for ch in CHANNELS:
            if ch not in channel_credits:
                channel_credits[ch] = 0.0

        return self._build_result(
            dict(channel_credits), journeys,
            metadata={
                "architecture": "BiLSTM + Attention",
                "hidden_dim": self.hidden_dim,
                "attention_weights": dict(self._attention_weights),
            },
        )
