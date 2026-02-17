"""
Temporal engine for timestamp generation.
Handles journey duration, inter-touch gaps, seasonality, and day-of-week patterns.
"""

import math
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np

# Journey duration follows lognormal distribution
JOURNEY_DURATION_PARAMS = {
    "lognormal_mu": 2.08,      # ln(8) — median 8 days
    "lognormal_sigma": 0.8,    # mean ~14 days
    "min_days": 0,
    "max_days": 90,
}

# Inter-touch gap parameters
INTER_TOUCH_GAP_PARAMS = {
    "digital_mean_gap_days": 2.5,
    "offline_mean_gap_days": 5.0,
    "min_gap_minutes": 5,
}

# Monthly seasonality — insurance shopping peaks in spring and fall
MONTHLY_SEASONALITY = {
    1: 0.85, 2: 0.90, 3: 1.20, 4: 1.25, 5: 1.05, 6: 0.95,
    7: 0.90, 8: 0.90, 9: 1.15, 10: 1.20, 11: 0.95, 12: 0.80,
}

# Day-of-week weights (0=Monday, 6=Sunday) — weekdays dominate
DAY_OF_WEEK_WEIGHTS = {
    0: 0.95, 1: 1.10, 2: 1.15, 3: 1.10, 4: 1.00, 5: 0.40, 6: 0.30,
}

# Digital channels: ["paid_search_brand", "paid_search_nonbrand", "organic_search",
# "display_programmatic", "paid_social", "direct_organic", "video_ott_ctv",
# "email_marketing", "aggregator_comparator"]
DIGITAL_CHANNELS = {
    "paid_search_brand", "paid_search_nonbrand", "organic_search",
    "display_programmatic", "paid_social", "direct_organic",
    "video_ott_ctv", "email_marketing", "aggregator_comparator",
}

OFFLINE_CHANNELS = {"independent_agent", "direct_mail", "call_center", "tv_radio"}

# Simulation base date (start of the 12-month simulation window)
SIMULATION_START = datetime(2025, 1, 1)
SIMULATION_END = datetime(2025, 12, 31)


def sample_journey_start_month(rng: np.random.Generator) -> int:
    """Sample a journey start month weighted by seasonality."""
    months = list(range(1, 13))
    weights = np.array([MONTHLY_SEASONALITY[m] for m in months])
    weights /= weights.sum()
    return int(rng.choice(months, p=weights))


def sample_journey_start_timestamp(
    month: int,
    rng: np.random.Generator,
) -> datetime:
    """Generate a random start timestamp within the given month."""
    year = 2025
    # Pick a random day, weighted by day-of-week
    days_in_month = 28 if month == 2 else (30 if month in [4, 6, 9, 11] else 31)
    candidate_days = list(range(1, days_in_month + 1))

    # Weight each day by its day-of-week factor
    day_weights = []
    for d in candidate_days:
        try:
            dow = datetime(year, month, d).weekday()
            day_weights.append(DAY_OF_WEEK_WEIGHTS.get(dow, 1.0))
        except ValueError:
            day_weights.append(0.0)

    day_weights = np.array(day_weights)
    day_weights /= day_weights.sum()

    day = int(rng.choice(candidate_days, p=day_weights))

    # Random hour (8am–10pm range, peak at 10am–2pm)
    hour_weights = np.zeros(24)
    for h in range(8, 22):
        if 10 <= h <= 14:
            hour_weights[h] = 2.0
        elif 18 <= h <= 21:
            hour_weights[h] = 1.5
        else:
            hour_weights[h] = 1.0
    hour_weights /= hour_weights.sum()
    hour = int(rng.choice(24, p=hour_weights))

    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))

    return datetime(year, month, day, hour, minute, second)


def sample_journey_duration_days(
    rng: np.random.Generator,
    has_life_event: bool = False,
) -> float:
    """
    Sample journey duration from lognormal distribution.
    Life events → longer shopping windows.
    """
    mu = JOURNEY_DURATION_PARAMS["lognormal_mu"]
    sigma = JOURNEY_DURATION_PARAMS["lognormal_sigma"]

    if has_life_event:
        mu += 0.3  # ~35% longer journeys

    duration = float(rng.lognormal(mu, sigma))
    duration = max(JOURNEY_DURATION_PARAMS["min_days"],
                   min(duration, JOURNEY_DURATION_PARAMS["max_days"]))
    return duration


def generate_touch_timestamps(
    start_time: datetime,
    journey_duration_days: float,
    channel_path: List[str],
    rng: np.random.Generator,
) -> List[datetime]:
    """
    Generate ordered timestamps for each touchpoint in the journey.

    Inter-touch gaps based on channel type (digital = shorter gaps).

    Args:
        start_time: Journey start timestamp.
        journey_duration_days: Total journey duration in days.
        channel_path: Ordered list of channels.
        rng: NumPy random generator.

    Returns:
        List of timestamps aligned with channel_path.
    """
    n_touches = len(channel_path)
    if n_touches == 0:
        return []
    if n_touches == 1:
        return [start_time]

    timestamps = [start_time]
    remaining_time = timedelta(days=journey_duration_days)

    for i in range(1, n_touches):
        channel = channel_path[i]

        # Mean gap depends on channel type
        if channel in DIGITAL_CHANNELS:
            mean_gap = INTER_TOUCH_GAP_PARAMS["digital_mean_gap_days"]
        else:
            mean_gap = INTER_TOUCH_GAP_PARAMS["offline_mean_gap_days"]

        # Exponential distribution for gap
        gap_days = float(rng.exponential(mean_gap))

        # Enforce minimum gap
        min_gap = INTER_TOUCH_GAP_PARAMS["min_gap_minutes"] / (24 * 60)
        gap_days = max(gap_days, min_gap)

        # Don't exceed journey duration
        max_remaining = journey_duration_days - (timestamps[-1] - start_time).total_seconds() / 86400
        gap_days = min(gap_days, max(max_remaining * 0.5, min_gap))

        next_time = timestamps[-1] + timedelta(days=gap_days)
        timestamps.append(next_time)

    return timestamps


def sample_dwell_time(channel: str, rng: np.random.Generator) -> float:
    """Sample dwell time in seconds based on channel type."""
    if channel in DIGITAL_CHANNELS:
        # Web visits: lognormal, median ~90 seconds
        return float(np.clip(rng.lognormal(4.5, 1.0), 10, 1800))
    elif channel == "independent_agent":
        # Agent meetings: 15-60 minutes
        return float(rng.uniform(900, 3600))
    elif channel == "call_center":
        # Phone calls: 5-30 minutes
        return float(rng.uniform(300, 1800))
    elif channel == "tv_radio":
        # Ad exposure: 15-60 seconds
        return float(rng.uniform(15, 60))
    else:
        return float(rng.uniform(30, 300))


def sample_viewability(channel: str, rng: np.random.Generator) -> float:
    """Sample viewability percentage (MRC standard) for impression channels."""
    if channel in ("display_programmatic", "paid_social", "video_ott_ctv"):
        # Viewability: beta distribution around 60-70%
        return float(np.clip(rng.beta(6, 3), 0.1, 1.0))
    return 1.0
