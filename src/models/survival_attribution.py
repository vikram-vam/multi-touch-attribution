"""
Survival / hazard attribution (Tier 4, Section 6.10).
Models time-to-conversion conditional on channel exposure.
"""

from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd

from src.models.base import BaseAttributionModel, AttributionResult
from src.data_generation.channel_transitions import CHANNELS


class SurvivalAttribution(BaseAttributionModel):
    """
    Survival analysis attribution.

    Models conversion as a time-to-event process. Channels that shorten
    the expected time-to-conversion receive more credit. Uses Cox
    proportional hazards-inspired coefficients.
    """

    @property
    def name(self) -> str:
        return "survival_hazard"

    @property
    def tier(self) -> str:
        return "statistical"

    def fit(self, journeys: pd.DataFrame) -> None:
        """Estimate hazard ratios from channel presence."""
        self._hazard_ratios = {}

        for channel in CHANNELS:
            # Compare conversion rates of journeys with vs without this channel
            with_channel = journeys[
                journeys.apply(
                    lambda r: channel in (r.get("channel_set", [])
                                         if isinstance(r.get("channel_set", []), list)
                                         else r.get("channel_set_str", "").split("|")),
                    axis=1
                )
            ]
            without_channel = journeys[
                ~journeys.apply(
                    lambda r: channel in (r.get("channel_set", [])
                                         if isinstance(r.get("channel_set", []), list)
                                         else r.get("channel_set_str", "").split("|")),
                    axis=1
                )
            ]

            rate_with = with_channel["is_converting"].mean() if len(with_channel) > 0 else 0
            rate_without = without_channel["is_converting"].mean() if len(without_channel) > 0 else 0.001

            # Hazard ratio: channels accelerating conversion get HR > 1
            hr = rate_with / max(rate_without, 0.001)
            self._hazard_ratios[channel] = max(float(hr), 0.01)

    def attribute(self, journeys: pd.DataFrame) -> AttributionResult:
        converting = journeys[journeys["is_converting"]]
        total_conversions = len(converting)

        # Credit proportional to hazard ratio above baseline (HR > 1 = helpful)
        positive_hrs = {ch: max(hr - 1.0, 0.0) for ch, hr in self._hazard_ratios.items()}
        total_hr = sum(positive_hrs.values())

        if total_hr > 0:
            channel_credits = {ch: (hr / total_hr) * total_conversions
                             for ch, hr in positive_hrs.items()}
        else:
            channel_credits = {ch: total_conversions / len(CHANNELS) for ch in CHANNELS}

        return self._build_result(
            channel_credits, journeys,
            metadata={"hazard_ratios": self._hazard_ratios},
        )
