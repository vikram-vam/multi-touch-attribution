"""
Logistic regression attribution (Tier 4, Section 6.9).
Uses channel presence as binary features to estimate incremental contribution.
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.base import BaseAttributionModel, AttributionResult
from src.data_generation.channel_transitions import CHANNELS


class LogisticAttribution(BaseAttributionModel):
    """
    Attribution via logistic regression coefficients.

    Each channel is a binary feature. The regression coefficient
    represents the channel's incremental lift on conversion probability.
    Credits proportional to exponentiated coefficients (odds ratios).
    """

    def __init__(self, regularization: float = 1.0):
        self.regularization = regularization
        self._model = None
        self._feature_names: List[str] = []

    @property
    def name(self) -> str:
        return "logistic_regression"

    @property
    def tier(self) -> str:
        return "statistical"

    def fit(self, journeys: pd.DataFrame) -> None:
        self._feature_names = sorted(CHANNELS)

        # Build feature matrix: binary channel presence
        X = np.zeros((len(journeys), len(self._feature_names)))
        y = journeys["is_converting"].astype(int).values

        for i, (_, row) in enumerate(journeys.iterrows()):
            channel_set = row.get("channel_set", [])
            if isinstance(channel_set, str):
                channel_set = channel_set.split("|")
            for ch in channel_set:
                if ch in self._feature_names:
                    col_idx = self._feature_names.index(ch)
                    X[i, col_idx] = 1

        self._model = LogisticRegression(
            C=self.regularization,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
        self._model.fit(X, y)

    def attribute(self, journeys: pd.DataFrame) -> AttributionResult:
        coefs = self._model.coef_[0]
        odds_ratios = np.exp(coefs)
        odds_ratios = np.maximum(odds_ratios, 0)  # Floor at 0

        converting = journeys[journeys["is_converting"]]
        total_conversions = len(converting)

        # Attribution proportional to positive odds ratios
        positive_or = {self._feature_names[i]: float(odds_ratios[i])
                      for i in range(len(self._feature_names))
                      if odds_ratios[i] > 1.0}

        total_or = sum(positive_or.values())
        if total_or > 0:
            channel_credits = {ch: (or_val / total_or) * total_conversions
                             for ch, or_val in positive_or.items()}
        else:
            channel_credits = {ch: total_conversions / len(self._feature_names)
                             for ch in self._feature_names}

        # Add zeros for channels with no positive contribution
        for ch in self._feature_names:
            if ch not in channel_credits:
                channel_credits[ch] = 0.0

        return self._build_result(
            channel_credits, journeys,
            metadata={
                "coefficients": {self._feature_names[i]: float(coefs[i])
                               for i in range(len(self._feature_names))},
                "odds_ratios": {self._feature_names[i]: float(odds_ratios[i])
                              for i in range(len(self._feature_names))},
            },
        )
