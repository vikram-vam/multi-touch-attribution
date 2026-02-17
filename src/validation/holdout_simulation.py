"""
Holdout simulation validation.
Remove-and-predict validation: remove a channel and check if the model
correctly predicts the impact on conversions.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from src.models.base import BaseAttributionModel, AttributionResult


def holdout_simulation(
    model: BaseAttributionModel,
    journeys: pd.DataFrame,
    channel_to_remove: str,
    n_simulations: int = 5,
) -> Dict:
    """
    Remove-and-predict validation.

    Removes all touchpoints for a channel from journey data and checks
    whether the re-computed attribution correctly predicts the
    conversion rate change.

    Args:
        model: Fitted attribution model.
        journeys: Full journey DataFrame.
        channel_to_remove: Channel to simulate removal of.
        n_simulations: Number of bootstrap samples.

    Returns:
        Dict with actual_change, predicted_change, and error metrics.
    """
    # Step 1: Get baseline attribution
    model.fit(journeys)
    baseline_result = model.attribute(journeys)
    baseline_credits = baseline_result.channel_credits
    channel_credit_pct = baseline_result.channel_credit_pct.get(channel_to_remove, 0)

    # Step 2: Remove channel from journeys
    modified_journeys = _remove_channel(journeys, channel_to_remove)

    # Step 3: Re-run model on modified data
    model.fit(modified_journeys)
    holdout_result = model.attribute(modified_journeys)

    # Step 4: Compute actual vs predicted change
    baseline_conv = journeys["is_converting"].sum()
    holdout_conv = modified_journeys["is_converting"].sum()
    actual_change = (holdout_conv - baseline_conv) / max(baseline_conv, 1)

    # Predicted change from attribution credits
    predicted_change = -channel_credit_pct  # Removing should reduce by this %

    return {
        "channel_removed": channel_to_remove,
        "baseline_conversions": int(baseline_conv),
        "holdout_conversions": int(holdout_conv),
        "actual_change_pct": actual_change,
        "predicted_change_pct": predicted_change,
        "prediction_error": abs(actual_change - predicted_change),
        "channel_attribution_pct": channel_credit_pct,
    }


def _remove_channel(
    journeys: pd.DataFrame,
    channel_to_remove: str,
) -> pd.DataFrame:
    """Remove a channel from journey paths."""
    df = journeys.copy()

    # Remove channel from paths
    def remove_from_path(path):
        if isinstance(path, str):
            path = path.split("|")
        return [ch for ch in path if ch != channel_to_remove]

    df["channel_path"] = df["channel_path"].apply(remove_from_path)
    df["touchpoint_count"] = df["channel_path"].apply(len)

    # Remove journeys that become empty
    df = df[df["touchpoint_count"] > 0].copy()

    # Update derived columns
    df["channel_path_str"] = df["channel_path"].apply(lambda x: "|".join(x) if isinstance(x, list) else x)
    df["channel_set"] = df["channel_path"].apply(lambda x: sorted(set(x)) if isinstance(x, list) else [])
    df["channel_set_str"] = df["channel_set"].apply(lambda x: "|".join(x) if isinstance(x, list) else x)
    df["distinct_channel_count"] = df["channel_set"].apply(len)
    df["first_touch_channel"] = df["channel_path"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "")
    df["last_touch_channel"] = df["channel_path"].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else "")
    df["has_agent_touch"] = df["channel_set_str"].str.contains("independent_agent", na=False)

    return df


def run_full_holdout_validation(
    model: BaseAttributionModel,
    journeys: pd.DataFrame,
    channels_to_test: List[str] = None,
) -> pd.DataFrame:
    """
    Run holdout simulation for multiple channels.

    Returns DataFrame with validation results per channel.
    """
    if channels_to_test is None:
        # Test top-5 channels by attribution
        model.fit(journeys)
        result = model.attribute(journeys)
        top_channels = sorted(
            result.channel_credits.items(), key=lambda x: -x[1]
        )[:5]
        channels_to_test = [ch for ch, _ in top_channels]

    results = []
    for ch in channels_to_test:
        res = holdout_simulation(model, journeys, ch)
        results.append(res)

    return pd.DataFrame(results)
