"""
Channel deep-dive metrics.
Computes all metrics for the Channel Deep Dive page (EP-6).
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class ChannelMetrics:
    """Metrics for a single channel deep-dive."""
    channel_id: str
    total_touchpoints: int
    total_attributed_conversions: float
    attribution_pct: float
    avg_position_in_path: float
    total_spend: float
    cost_per_bind: float
    roi: float
    model_agreement: Dict[str, float]
    funnel_position_dist: Dict[str, int]

    @classmethod
    def from_data(
        cls,
        channel_id: str,
        attribution_results: pd.DataFrame,
        journeys: pd.DataFrame,
        channel_spend: pd.DataFrame,
        avg_premium: float = 1200.0,
    ) -> "ChannelMetrics":
        # Attribution across models
        ch_results = attribution_results[attribution_results["channel_id"] == channel_id]
        model_agreement = {}
        for _, row in ch_results.iterrows():
            model_agreement[row["model_name"]] = row.get("attribution_pct", 0)

        # Average attribution
        avg_pct = ch_results["attribution_pct"].mean() if len(ch_results) > 0 else 0
        avg_conv = ch_results["attributed_conversions"].mean() if len(ch_results) > 0 else 0

        # Touchpoints and position
        converting = journeys[journeys["is_converting"]]
        tp_count = 0
        positions = []

        for _, row in converting.iterrows():
            path = row.get("channel_path", [])
            if isinstance(path, str):
                path = path.split("|")
            for i, ch in enumerate(path):
                if ch == channel_id:
                    tp_count += 1
                    positions.append(i / max(len(path) - 1, 1))

        avg_pos = np.mean(positions) if positions else 0.5

        # Funnel position distribution
        funnel_dist = {"intro": 0, "mid": 0, "closer": 0}
        for p in positions:
            if p < 0.33:
                funnel_dist["intro"] += 1
            elif p < 0.67:
                funnel_dist["mid"] += 1
            else:
                funnel_dist["closer"] += 1

        # Spend
        ch_spend = channel_spend[channel_spend["channel_id"] == channel_id]
        total_spend = ch_spend["spend_dollars"].sum() if len(ch_spend) > 0 else 0
        cpb = total_spend / max(avg_conv, 1)
        roi = (avg_conv * avg_premium) / max(total_spend, 1)

        return cls(
            channel_id=channel_id,
            total_touchpoints=tp_count,
            total_attributed_conversions=avg_conv,
            attribution_pct=avg_pct,
            avg_position_in_path=avg_pos,
            total_spend=total_spend,
            cost_per_bind=cpb,
            roi=roi,
            model_agreement=model_agreement,
            funnel_position_dist=funnel_dist,
        )


def compute_all_channel_metrics(
    attribution_results: pd.DataFrame,
    journeys: pd.DataFrame,
    channel_spend: pd.DataFrame,
) -> List[ChannelMetrics]:
    """Compute metrics for all channels."""
    channels = attribution_results["channel_id"].unique()
    return [
        ChannelMetrics.from_data(ch, attribution_results, journeys, channel_spend)
        for ch in channels
    ]
