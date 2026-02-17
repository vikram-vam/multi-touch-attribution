"""
Time decay weight computation.
Applies exponential decay based on channel-specific half-lives.
"""

import math
from typing import Dict, List

import numpy as np
import pandas as pd


# Default half-lives per channel type
DEFAULT_HALF_LIVES = {
    "click": 7.0,        # Search, organic, aggregator
    "impression": 3.0,   # Display, social, TV, video
    "agent": 14.0,       # Agent meetings
    "call": 7.0,         # Call center
    "mail": 10.0,        # Direct mail
    "email": 5.0,        # Email
}


def compute_time_decay_weight(
    days_before_conversion: float,
    half_life_days: float,
) -> float:
    """
    Compute exponential decay weight.

    weight = 2^(-t / half_life)

    Args:
        days_before_conversion: Days between touch and conversion.
        half_life_days: Channel-specific half-life in days.

    Returns:
        Weight in (0, 1].
    """
    if days_before_conversion <= 0:
        return 1.0
    return math.pow(2, -days_before_conversion / half_life_days)


def apply_time_decay(
    journeys_df: pd.DataFrame,
    channel_half_lives: Dict[str, float] = None,
) -> pd.DataFrame:
    """
    Apply time decay weights to assembled journeys.

    Each touchpoint in the journey receives a weight based on its
    temporal distance from conversion.

    Args:
        journeys_df: Assembled journeys with 'channel_path' and timing info.
        channel_half_lives: Mapping of channel_id → half_life_days.

    Returns:
        DataFrame with 'time_decay_weights' column added.
    """
    if channel_half_lives is None:
        # Use defaults based on touch type
        channel_half_lives = {
            "paid_search_brand": 7.0, "paid_search_nonbrand": 7.0,
            "organic_search": 7.0, "direct_organic": 7.0,
            "display_programmatic": 3.0, "paid_social": 3.0,
            "tv_radio": 5.0, "video_ott_ctv": 3.0,
            "independent_agent": 14.0, "call_center": 7.0,
            "direct_mail": 10.0, "email_marketing": 5.0,
            "aggregator_comparator": 5.0,
        }

    df = journeys_df.copy()

    decay_weights_column = []

    for _, row in df.iterrows():
        channel_path = row["channel_path"]
        n_touches = len(channel_path)

        if n_touches <= 1:
            decay_weights_column.append([1.0])
            continue

        # Distribute journey duration evenly across touches
        duration = row.get("journey_duration_days", 0.0)
        if duration <= 0 or n_touches <= 1:
            decay_weights_column.append([1.0] * n_touches)
            continue

        # Compute days before conversion for each touch
        # Last touch = 0 days, first touch = full duration
        days_before = [
            duration * (1.0 - i / (n_touches - 1))
            for i in range(n_touches)
        ]

        weights = []
        for i, channel in enumerate(channel_path):
            half_life = channel_half_lives.get(channel, 7.0)
            w = compute_time_decay_weight(days_before[i], half_life)
            weights.append(round(w, 4))

        decay_weights_column.append(weights)

    df["time_decay_weights"] = decay_weights_column

    return df
