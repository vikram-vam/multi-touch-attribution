"""
Journey simulator — the core Markov-based state machine.
Generates complete customer journeys from entry to conversion/dropout.
"""

import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_generation.user_profiles import UserProfile
from src.data_generation.channel_transitions import (
    CHANNELS, FIRST_TOUCH_DISTRIBUTION, get_transition_prob,
    CONVERSION, NULL,
)
from src.data_generation.conversion_model import (
    compute_conversion_probability, should_convert,
)
from src.data_generation.timestamp_engine import (
    sample_journey_start_month,
    sample_journey_start_timestamp,
    sample_journey_duration_days,
    generate_touch_timestamps,
    sample_dwell_time,
    sample_viewability,
    DIGITAL_CHANNELS,
)


def _sample_first_touch(
    rng: np.random.Generator,
    digital_propensity: float,
) -> str:
    """Sample the first channel touch, adjusted by digital propensity."""
    dist = dict(FIRST_TOUCH_DISTRIBUTION)

    # Adjust for digital propensity
    if digital_propensity > 0.6:
        for ch in ["paid_search_brand", "paid_search_nonbrand", "organic_search",
                    "display_programmatic", "paid_social"]:
            dist[ch] *= 1.0 + (digital_propensity - 0.5) * 0.5
    elif digital_propensity < 0.4:
        for ch in ["independent_agent", "direct_mail", "tv_radio"]:
            dist[ch] *= 1.0 + (0.5 - digital_propensity) * 0.5

    channels = list(dist.keys())
    probs = np.array(list(dist.values()))
    probs /= probs.sum()
    return channels[rng.choice(len(channels), p=probs)]


def simulate_single_journey(
    profile: UserProfile,
    rng: np.random.Generator,
    max_touchpoints: int = 20,
    agent_last_touch_pct: float = 0.60,
) -> Optional[Dict]:
    """
    Simulate a complete customer journey for one user.

    Uses a Markov-based state machine with conversion probability
    computed at each step via the logistic conversion model.

    Returns:
        Dict with journey data or None if journey has zero touches.
    """
    has_life_event = profile.life_event_trigger is not None

    # Sample journey start
    month = sample_journey_start_month(rng)
    start_time = sample_journey_start_timestamp(month, rng)
    duration_days = sample_journey_duration_days(rng, has_life_event)

    # Generate channel path via state machine
    channel_path = []
    is_converting = False
    prev_channel = None

    # First touch
    first_channel = _sample_first_touch(rng, profile.digital_propensity)
    channel_path.append(first_channel)
    current = first_channel

    # Simulate subsequent touches
    for step in range(1, max_touchpoints):
        # Get next state
        next_state = get_transition_prob(
            current, rng,
            prev=prev_channel,
            digital_propensity=profile.digital_propensity,
        )

        if next_state == CONVERSION:
            is_converting = True
            break
        elif next_state == NULL:
            break
        else:
            channel_path.append(next_state)
            prev_channel = current
            current = next_state

            # Check conversion probability at this step
            recency = max(0, 1.0 - step * 0.1)  # Recency decays with steps
            if should_convert(channel_path, current, rng, has_life_event, recency):
                is_converting = True
                break

    if len(channel_path) == 0:
        return None

    # For converting journeys, ensure agent as last touch at configured rate
    if is_converting and channel_path[-1] != "independent_agent":
        if rng.random() < agent_last_touch_pct:
            channel_path.append("independent_agent")

    # Generate timestamps
    timestamps = generate_touch_timestamps(start_time, duration_days, channel_path, rng)

    # Unique channel set
    channel_set = sorted(set(channel_path))

    # Determine conversion type
    if is_converting:
        conversion_type = "bind"
        conversion_value = float(np.clip(rng.normal(1200, 300), 500, 3000))
    else:
        conversion_type = "null"
        conversion_value = 0.0

    # Agent touch analysis
    has_agent = "independent_agent" in channel_path
    if has_agent:
        agent_positions = [i for i, ch in enumerate(channel_path)
                          if ch == "independent_agent"]
        if agent_positions[-1] == len(channel_path) - 1:
            agent_position = "last"
        elif agent_positions[0] == 0:
            agent_position = "first"
        else:
            agent_position = "middle"
    else:
        agent_position = "none"

    journey_id = str(uuid.uuid4())

    return {
        "journey_id": journey_id,
        "persistent_id": profile.persistent_id,
        "channel_path": channel_path,
        "channel_path_str": "|".join(channel_path),
        "channel_set": channel_set,
        "channel_set_str": "|".join(channel_set),
        "touchpoint_count": len(channel_path),
        "journey_duration_days": round(
            (timestamps[-1] - timestamps[0]).total_seconds() / 86400
            if len(timestamps) > 1 else 0.0, 2
        ),
        "first_touch_channel": channel_path[0],
        "last_touch_channel": channel_path[-1],
        "is_converting": is_converting,
        "conversion_type": conversion_type,
        "conversion_value": round(conversion_value, 2),
        "has_agent_touch": has_agent,
        "agent_touch_position": agent_position,
        "simulation_month": month,
        "age_band": profile.age_band,
        "state": profile.state,
        "life_event_trigger": profile.life_event_trigger or "none",
        # Touchpoint-level data (for separate extraction)
        "_timestamps": timestamps,
        "_profile": profile,
    }


def simulate_journeys(
    profiles: List[UserProfile],
    rng: np.random.Generator,
    max_touchpoints: int = 20,
    agent_last_touch_pct: float = 0.60,
) -> List[Dict]:
    """
    Simulate journeys for all user profiles.

    Args:
        profiles: List of user profiles.
        rng: NumPy random generator.
        max_touchpoints: Maximum touches per journey.
        agent_last_touch_pct: Fraction of converting journeys with agent as last touch.

    Returns:
        List of journey dicts.
    """
    journeys = []
    for profile in profiles:
        journey = simulate_single_journey(
            profile, rng, max_touchpoints, agent_last_touch_pct,
        )
        if journey is not None:
            journeys.append(journey)
    return journeys


def extract_touchpoints(journeys: List[Dict], rng: np.random.Generator) -> pd.DataFrame:
    """
    Extract touchpoint-level DataFrame from journey data.
    Matches touchpoints.parquet schema (Section 3.2).
    """
    rows = []
    for journey in journeys:
        channel_path = journey["channel_path"]
        timestamps = journey["_timestamps"]
        profile = journey["_profile"]

        for i, (channel, ts) in enumerate(zip(channel_path, timestamps)):
            touch_type = "click"
            if channel in ("independent_agent",):
                touch_type = "agent"
            elif channel in ("display_programmatic", "paid_social", "tv_radio", "video_ott_ctv"):
                touch_type = "impression"
            elif channel == "call_center":
                touch_type = "call"
            elif channel == "direct_mail":
                touch_type = "mail"
            elif channel == "email_marketing":
                touch_type = "email"

            device = "offline" if channel in ("independent_agent", "direct_mail", "call_center") else \
                     rng.choice(["desktop", "mobile", "tablet"], p=[0.45, 0.40, 0.15])

            rows.append({
                "touchpoint_id": str(uuid.uuid4()),
                "persistent_id": profile.persistent_id,
                "channel_id": channel,
                "sub_channel": f"{channel}_default",
                "touch_type": touch_type,
                "event_timestamp": ts,
                "session_id": str(uuid.uuid4())[:8],
                "dwell_time_seconds": round(sample_dwell_time(channel, rng), 1),
                "viewability_pct": round(sample_viewability(channel, rng), 3),
                "device_type": device,
                "state": profile.state,
                "campaign_id": f"camp_{channel[:6]}_{rng.integers(1, 20):03d}",
                "creative_id": f"cr_{rng.integers(1, 50):04d}",
                "utm_source": channel.split("_")[0] if "_" in channel else channel,
                "utm_medium": touch_type,
                "utm_campaign": f"erie_auto_2025_{channel[:8]}",
                "is_qualified": True,
                "touch_weight": 1.0 if touch_type in ("click", "agent", "call") else 0.6,
            })

    return pd.DataFrame(rows)


def extract_conversions(journeys: List[Dict], rng: np.random.Generator) -> pd.DataFrame:
    """
    Extract conversion events from journey data.
    Matches conversions.parquet schema (Section 3.3).
    """
    rows = []
    for journey in journeys:
        if not journey["is_converting"]:
            continue

        timestamps = journey["_timestamps"]
        conversion_ts = timestamps[-1]

        rows.append({
            "conversion_id": str(uuid.uuid4()),
            "persistent_id": journey["persistent_id"],
            "conversion_type": journey["conversion_type"],
            "conversion_timestamp": conversion_ts,
            "conversion_value": journey["conversion_value"],
            "channel_of_conversion": journey["last_touch_channel"],
            "agent_id": f"agent_{rng.integers(1, 12000):05d}" if journey["has_agent_touch"] else None,
            "policy_line": "auto",
            "state": journey["state"],
        })

    return pd.DataFrame(rows)


def extract_journey_dataframe(journeys: List[Dict]) -> pd.DataFrame:
    """
    Extract journey-level DataFrame.
    Matches journeys.parquet schema (Section 3.4).
    """
    records = []
    for j in journeys:
        records.append({
            "journey_id": j["journey_id"],
            "persistent_id": j["persistent_id"],
            "channel_path_str": j["channel_path_str"],
            "channel_set_str": j["channel_set_str"],
            "touchpoint_count": j["touchpoint_count"],
            "journey_duration_days": j["journey_duration_days"],
            "first_touch_channel": j["first_touch_channel"],
            "last_touch_channel": j["last_touch_channel"],
            "is_converting": j["is_converting"],
            "conversion_type": j["conversion_type"],
            "conversion_value": j["conversion_value"],
            "has_agent_touch": j["has_agent_touch"],
            "agent_touch_position": j["agent_touch_position"],
            "simulation_month": j["simulation_month"],
            "age_band": j["age_band"],
            "state": j["state"],
            "life_event_trigger": j["life_event_trigger"],
        })
    return pd.DataFrame(records)
