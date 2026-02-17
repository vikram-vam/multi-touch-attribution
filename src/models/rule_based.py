"""
Rule-based attribution models (Tier 1).
First-Touch, Last-Touch, Linear, Time-Decay, Position-Based.
"""

from typing import Dict, List
from collections import defaultdict

import numpy as np
import pandas as pd

from src.models.base import BaseAttributionModel, AttributionResult


class FirstTouchAttribution(BaseAttributionModel):
    """100% credit to first touchpoint in each journey."""

    @property
    def name(self) -> str:
        return "first_touch"

    @property
    def tier(self) -> str:
        return "rule_based"

    def fit(self, journeys: pd.DataFrame) -> None:
        pass  # No fitting needed

    def attribute(self, journeys: pd.DataFrame) -> AttributionResult:
        converting = journeys[journeys["is_converting"]].copy()
        credits = defaultdict(float)
        journey_rows = []

        for _, row in converting.iterrows():
            channel = row["first_touch_channel"]
            credits[channel] += 1.0
            journey_rows.append({
                "journey_id": row.get("journey_id", ""),
                "channel_id": channel,
                "credit": 1.0,
            })

        journey_credits = pd.DataFrame(journey_rows)
        return self._build_result(dict(credits), journeys, journey_credits)


class LastTouchAttribution(BaseAttributionModel):
    """100% credit to last touchpoint — matches GA4 default."""

    @property
    def name(self) -> str:
        return "last_touch"

    @property
    def tier(self) -> str:
        return "rule_based"

    def fit(self, journeys: pd.DataFrame) -> None:
        pass

    def attribute(self, journeys: pd.DataFrame) -> AttributionResult:
        converting = journeys[journeys["is_converting"]].copy()
        credits = defaultdict(float)
        journey_rows = []

        for _, row in converting.iterrows():
            channel = row["last_touch_channel"]
            credits[channel] += 1.0
            journey_rows.append({
                "journey_id": row.get("journey_id", ""),
                "channel_id": channel,
                "credit": 1.0,
            })

        journey_credits = pd.DataFrame(journey_rows)
        return self._build_result(
            dict(credits), journeys, journey_credits,
            metadata={"note": "GA4 default attribution model"},
        )


class LinearAttribution(BaseAttributionModel):
    """Equal credit to all touchpoints in journey."""

    @property
    def name(self) -> str:
        return "linear"

    @property
    def tier(self) -> str:
        return "rule_based"

    def fit(self, journeys: pd.DataFrame) -> None:
        pass

    def attribute(self, journeys: pd.DataFrame) -> AttributionResult:
        converting = journeys[journeys["is_converting"]].copy()
        credits = defaultdict(float)
        journey_rows = []

        for _, row in converting.iterrows():
            path = row["channel_path"]
            if isinstance(path, str):
                path = path.split("|")
            n = len(path)
            weight = 1.0 / n if n > 0 else 0.0

            for ch in path:
                credits[ch] += weight
                journey_rows.append({
                    "journey_id": row.get("journey_id", ""),
                    "channel_id": ch,
                    "credit": weight,
                })

        journey_credits = pd.DataFrame(journey_rows)
        return self._build_result(dict(credits), journeys, journey_credits)


class TimeDecayAttribution(BaseAttributionModel):
    """More credit to recent touchpoints using exponential decay."""

    def __init__(self, half_life_days: float = 7.0):
        self.half_life_days = half_life_days

    @property
    def name(self) -> str:
        return "time_decay"

    @property
    def tier(self) -> str:
        return "rule_based"

    def fit(self, journeys: pd.DataFrame) -> None:
        pass

    def attribute(self, journeys: pd.DataFrame) -> AttributionResult:
        converting = journeys[journeys["is_converting"]].copy()
        credits = defaultdict(float)
        journey_rows = []

        for _, row in converting.iterrows():
            path = row["channel_path"]
            if isinstance(path, str):
                path = path.split("|")
            n = len(path)

            # Use time_decay_weights if available, else compute from position
            if "time_decay_weights" in row and row["time_decay_weights"]:
                weights = row["time_decay_weights"]
                if isinstance(weights, str):
                    weights = [float(w) for w in weights.split(",")]
            else:
                # Position-based exponential decay
                duration = row.get("journey_duration_days", 7.0)
                weights = []
                for i in range(n):
                    days_before = duration * (1.0 - i / max(n - 1, 1))
                    w = 2 ** (-days_before / self.half_life_days)
                    weights.append(w)

            # Normalize weights
            total_w = sum(weights)
            if total_w > 0:
                weights = [w / total_w for w in weights]

            for ch, w in zip(path, weights):
                credits[ch] += w
                journey_rows.append({
                    "journey_id": row.get("journey_id", ""),
                    "channel_id": ch,
                    "credit": w,
                })

        journey_credits = pd.DataFrame(journey_rows)
        return self._build_result(dict(credits), journeys, journey_credits)


class PositionBasedAttribution(BaseAttributionModel):
    """
    Position-based (U-shaped): 40% first, 40% last, 20% distributed middle.
    """

    def __init__(self, first_weight: float = 0.40, last_weight: float = 0.40):
        self.first_weight = first_weight
        self.last_weight = last_weight

    @property
    def name(self) -> str:
        return "position_based"

    @property
    def tier(self) -> str:
        return "rule_based"

    def fit(self, journeys: pd.DataFrame) -> None:
        pass

    def attribute(self, journeys: pd.DataFrame) -> AttributionResult:
        converting = journeys[journeys["is_converting"]].copy()
        credits = defaultdict(float)
        journey_rows = []

        middle_weight = 1.0 - self.first_weight - self.last_weight

        for _, row in converting.iterrows():
            path = row["channel_path"]
            if isinstance(path, str):
                path = path.split("|")
            n = len(path)

            if n == 1:
                credits[path[0]] += 1.0
                journey_rows.append({
                    "journey_id": row.get("journey_id", ""),
                    "channel_id": path[0],
                    "credit": 1.0,
                })
            elif n == 2:
                half = 0.5
                for ch in path:
                    credits[ch] += half
                    journey_rows.append({
                        "journey_id": row.get("journey_id", ""),
                        "channel_id": ch,
                        "credit": half,
                    })
            else:
                # First touch
                credits[path[0]] += self.first_weight
                journey_rows.append({
                    "journey_id": row.get("journey_id", ""),
                    "channel_id": path[0],
                    "credit": self.first_weight,
                })

                # Last touch
                credits[path[-1]] += self.last_weight
                journey_rows.append({
                    "journey_id": row.get("journey_id", ""),
                    "channel_id": path[-1],
                    "credit": self.last_weight,
                })

                # Middle touches
                n_middle = n - 2
                mid_credit = middle_weight / n_middle if n_middle > 0 else 0

                for ch in path[1:-1]:
                    credits[ch] += mid_credit
                    journey_rows.append({
                        "journey_id": row.get("journey_id", ""),
                        "channel_id": ch,
                        "credit": mid_credit,
                    })

        journey_credits = pd.DataFrame(journey_rows)
        return self._build_result(dict(credits), journeys, journey_credits)
