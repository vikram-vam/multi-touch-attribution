"""
Marginal ROI and diminishing returns curves per channel.
Provides functions for computing marginal returns at different spend levels.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.optimization.budget_optimizer import (
    DEFAULT_RESPONSE_CURVES, response_curve, marginal_roi
)


def compute_response_curve_points(
    channel_id: str,
    current_spend: float,
    n_points: int = 50,
    spend_range: Tuple[float, float] = (0.1, 3.0),
) -> pd.DataFrame:
    """
    Compute response curve points for a channel across a spend range.

    Args:
        channel_id: Channel identifier.
        current_spend: Current annual spend for scaling.
        n_points: Number of points to compute.
        spend_range: (min_frac, max_frac) of current spend.

    Returns:
        DataFrame with spend_level, conversions, marginal_roi columns.
    """
    params = DEFAULT_RESPONSE_CURVES.get(
        channel_id, {"alpha": 0.5, "beta": 0.5, "saturation_pct": 2.0}
    )
    alpha, beta = params["alpha"], params["beta"]

    min_spend = current_spend * spend_range[0]
    max_spend = current_spend * spend_range[1]
    spend_levels = np.linspace(min_spend, max_spend, n_points)

    records = []
    for spend in spend_levels:
        conv = response_curve(spend, alpha, beta, current_spend)
        mr = marginal_roi(spend, alpha, beta, current_spend)
        records.append({
            "channel_id": channel_id,
            "spend_level": spend,
            "spend_pct_of_current": spend / max(current_spend, 1),
            "projected_conversions": conv,
            "marginal_roi": mr,
            "roi_at_level": conv / max(spend, 1),
        })

    return pd.DataFrame(records)


def compute_all_response_curves(
    channel_spend: pd.DataFrame,
    n_points: int = 50,
) -> pd.DataFrame:
    """Compute response curves for all channels."""
    current_by_channel = channel_spend.groupby("channel_id")["spend_dollars"].sum()

    all_curves = []
    for ch_id, spend in current_by_channel.items():
        curve = compute_response_curve_points(ch_id, spend, n_points)
        all_curves.append(curve)

    return pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()


def find_saturation_point(
    channel_id: str,
    current_spend: float,
    threshold: float = 0.05,
) -> float:
    """
    Find the spend level where marginal ROI drops below threshold.

    Returns:
        Spend level at saturation.
    """
    params = DEFAULT_RESPONSE_CURVES.get(
        channel_id, {"alpha": 0.5, "beta": 0.5, "saturation_pct": 2.0}
    )
    alpha, beta = params["alpha"], params["beta"]

    for mult in np.arange(0.1, 5.0, 0.05):
        spend = current_spend * mult
        mr = marginal_roi(spend, alpha, beta, current_spend)
        if mr < threshold:
            return spend

    return current_spend * 5.0  # Not saturated in range


def compute_diminishing_returns_summary(
    channel_spend: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize diminishing returns for all channels."""
    current_by_channel = channel_spend.groupby("channel_id")["spend_dollars"].sum()

    records = []
    for ch_id, spend in current_by_channel.items():
        params = DEFAULT_RESPONSE_CURVES.get(
            ch_id, {"alpha": 0.5, "beta": 0.5, "saturation_pct": 2.0}
        )
        mr_current = marginal_roi(spend, params["alpha"], params["beta"], spend)
        sat_point = find_saturation_point(ch_id, spend)
        headroom = (sat_point - spend) / max(spend, 1)

        records.append({
            "channel_id": ch_id,
            "current_spend": spend,
            "current_marginal_roi": mr_current,
            "saturation_spend": sat_point,
            "headroom_pct": headroom,
            "recommendation": "invest_more" if headroom > 0.3 else
                             "at_saturation" if headroom < 0.1 else "moderate",
        })

    return pd.DataFrame(records)
