"""
Journey path metrics.
Computes all metrics for the Journey Paths page (EP-3).
"""

from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd

from src.data_generation.channel_transitions import CHANNELS


@dataclass
class JourneyMetrics:
    """Metrics for the Journey Paths page."""
    total_converting_journeys: int
    avg_path_length: float
    top_paths: pd.DataFrame
    agent_multiplier: float
    single_touch_pct: float
    channel_cooccurrence: pd.DataFrame

    @classmethod
    def from_data(cls, journeys: pd.DataFrame) -> "JourneyMetrics":
        converting = journeys[journeys["is_converting"]]
        total = len(converting)
        avg_len = converting["touchpoint_count"].mean() if total > 0 else 0

        path_counts = converting["channel_path_str"].value_counts().head(20).reset_index()
        path_counts.columns = ["path", "count"]
        path_counts["pct"] = path_counts["count"] / total

        with_agent = journeys[journeys["has_agent_touch"]]
        without_agent = journeys[~journeys["has_agent_touch"]]
        agent_rate = with_agent["is_converting"].mean() if len(with_agent) > 0 else 0
        no_agent_rate = without_agent["is_converting"].mean() if len(without_agent) > 0 else 0.001
        multiplier = agent_rate / max(no_agent_rate, 0.001)

        single = converting[converting["touchpoint_count"] == 1]
        single_pct = len(single) / total if total > 0 else 0

        cooccurrence = pd.DataFrame(0, index=CHANNELS, columns=CHANNELS)
        for _, row in converting.iterrows():
            chs = row.get("channel_set", [])
            if isinstance(chs, str):
                chs = chs.split("|")
            for c1 in chs:
                for c2 in chs:
                    if c1 in CHANNELS and c2 in CHANNELS:
                        cooccurrence.loc[c1, c2] += 1

        return cls(
            total_converting_journeys=total,
            avg_path_length=avg_len,
            top_paths=path_counts,
            agent_multiplier=multiplier,
            single_touch_pct=single_pct,
            channel_cooccurrence=cooccurrence,
        )
