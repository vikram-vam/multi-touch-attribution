"""
Journey assembly module.
Builds complete customer journeys from sessionized, qualified touchpoints.
Produces assembled_journeys.parquet — the primary input for attribution models.
"""

import pandas as pd
import numpy as np
from typing import List, Optional

from src.utils.data_io import load_parquet, save_parquet


def assemble_journeys(
    touchpoints_df: pd.DataFrame,
    conversions_df: pd.DataFrame,
    max_lookback_days: int = 90,
) -> pd.DataFrame:
    """
    Assemble qualified touchpoints into attribution-ready journeys.

    Links touchpoints to conversions within the lookback window and
    creates the channel path representation needed by attribution models.

    Args:
        touchpoints_df: Qualified touchpoints.
        conversions_df: Conversion events.
        max_lookback_days: Maximum days before conversion to include touches.

    Returns:
        DataFrame matching assembled_journeys.parquet schema (Section 3.7).
    """
    tp = touchpoints_df.copy()
    conv = conversions_df.copy()

    tp["event_timestamp"] = pd.to_datetime(tp["event_timestamp"])
    conv["conversion_timestamp"] = pd.to_datetime(conv["conversion_timestamp"])

    # Merge touchpoints with conversions on persistent_id
    merged = pd.merge(
        tp,
        conv[["persistent_id", "conversion_id", "conversion_timestamp",
              "conversion_value", "conversion_type"]],
        on="persistent_id",
        how="left",
    )

    # Filter: keep touches within lookback window of conversion
    merged["days_before_conversion"] = (
        (merged["conversion_timestamp"] - merged["event_timestamp"]).dt.total_seconds() / 86400
    )

    # Keep touches before or at conversion and within lookback
    valid_touches = merged[
        (merged["days_before_conversion"] >= 0) &
        (merged["days_before_conversion"] <= max_lookback_days) |
        merged["conversion_id"].isna()  # Non-converting journeys
    ].copy()

    # Sort by user and time
    valid_touches = valid_touches.sort_values(
        ["persistent_id", "event_timestamp"]
    ).reset_index(drop=True)

    # Build journey-level features
    journeys = []

    for pid, group in valid_touches.groupby("persistent_id"):
        # Channel path
        channel_path = group["channel_id"].tolist()
        channel_path_str = "|".join(channel_path)
        unique_channels = sorted(set(channel_path))

        # Conversion info
        has_conversion = group["conversion_id"].notna().any()
        conversation_value = group["conversion_value"].max() if has_conversion else 0.0
        conversion_id = group["conversion_id"].dropna().iloc[0] if has_conversion else None

        # Time features
        first_ts = group["event_timestamp"].min()
        last_ts = group["event_timestamp"].max()
        duration = (last_ts - first_ts).total_seconds() / 86400

        # Agent features
        has_agent = "independent_agent" in channel_path
        agent_position = "none"
        if has_agent:
            last_idx = len(channel_path) - 1
            agent_positions = [i for i, ch in enumerate(channel_path)
                             if ch == "independent_agent"]
            if agent_positions[-1] == last_idx:
                agent_position = "last"
            elif agent_positions[0] == 0:
                agent_position = "first"
            else:
                agent_position = "middle"

        # Touch weights
        touch_weights = group["touch_weight"].tolist()

        journeys.append({
            "journey_id": f"j_{pid[:8]}",
            "persistent_id": pid,
            "conversion_id": conversion_id,
            "channel_path": channel_path,
            "channel_path_str": channel_path_str,
            "channel_set": unique_channels,
            "channel_set_str": "|".join(unique_channels),
            "touchpoint_count": len(channel_path),
            "distinct_channel_count": len(unique_channels),
            "journey_start": first_ts,
            "journey_end": last_ts,
            "journey_duration_days": round(duration, 2),
            "first_touch_channel": channel_path[0],
            "last_touch_channel": channel_path[-1],
            "is_converting": has_conversion,
            "conversion_value": conversation_value,
            "has_agent_touch": has_agent,
            "agent_touch_position": agent_position,
            "touch_weights": touch_weights,
        })

    return pd.DataFrame(journeys)


def run_pipeline(
    touchpoints_path: str = None,
    conversions_path: str = None,
) -> pd.DataFrame:
    """
    Full pipeline: load raw data → sessionize → qualify → assemble.

    Args:
        touchpoints_path: Path to raw touchpoints Parquet.
        conversions_path: Path to raw conversions Parquet.

    Returns:
        Assembled journeys DataFrame.
    """
    from src.pipeline.sessionizer import sessionize_touchpoints
    from src.pipeline.touch_qualifier import qualify_touchpoints
    from src.pipeline.channel_classifier import ensure_mece_classification

    # Load raw data
    touchpoints = load_parquet("touchpoints.parquet", "raw")
    conversions = load_parquet("conversions.parquet", "raw")

    # Pipeline steps
    touchpoints = sessionize_touchpoints(touchpoints)
    touchpoints = qualify_touchpoints(touchpoints)
    touchpoints = ensure_mece_classification(touchpoints)

    # Assemble into journeys
    assembled = assemble_journeys(touchpoints, conversions)

    # Save
    save_parquet(assembled, "assembled_journeys.parquet", "processed")

    return assembled
