"""
Session detection module.
Groups touchpoints into sessions using 30-minute inactivity threshold.
"""

import pandas as pd
import numpy as np
from typing import Optional


DEFAULT_SESSION_TIMEOUT_MINUTES = 30


def sessionize_touchpoints(
    touchpoints_df: pd.DataFrame,
    timeout_minutes: int = DEFAULT_SESSION_TIMEOUT_MINUTES,
) -> pd.DataFrame:
    """
    Assign session IDs to touchpoints based on inactivity gaps.

    A new session starts when the gap between consecutive touchpoints
    for the same user exceeds `timeout_minutes`.

    Args:
        touchpoints_df: Raw touchpoints with 'persistent_id' and 'event_timestamp'.
        timeout_minutes: Inactivity threshold for session breaks.

    Returns:
        DataFrame with added 'session_id' and 'session_sequence' columns.
    """
    df = touchpoints_df.copy()
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    df = df.sort_values(["persistent_id", "event_timestamp"]).reset_index(drop=True)

    # Compute time gaps within each user
    df["prev_timestamp"] = df.groupby("persistent_id")["event_timestamp"].shift(1)
    df["gap_minutes"] = (
        (df["event_timestamp"] - df["prev_timestamp"]).dt.total_seconds() / 60
    )

    # Flag new sessions
    df["new_session"] = (
        df["prev_timestamp"].isna() |
        (df["gap_minutes"] > timeout_minutes)
    ).astype(int)

    # Cumulative sum to create session groups
    df["session_group"] = df.groupby("persistent_id")["new_session"].cumsum()

    # Create unique session IDs
    df["session_id"] = (
        df["persistent_id"].astype(str) + "_s" + df["session_group"].astype(str)
    )

    # Session sequence number within each session
    df["session_sequence"] = df.groupby(["persistent_id", "session_id"]).cumcount()

    # Clean up temp columns
    df.drop(columns=["prev_timestamp", "gap_minutes", "new_session", "session_group"],
            inplace=True)

    return df
