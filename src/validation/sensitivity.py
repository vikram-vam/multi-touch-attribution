"""
Parameter sensitivity analysis.
Tests how attribution results change with different model parameters.
"""

from typing import Dict, List, Any

import numpy as np
import pandas as pd

from src.models.base import BaseAttributionModel, AttributionResult


def sensitivity_analysis(
    model_class: type,
    journeys: pd.DataFrame,
    param_name: str,
    param_values: List[Any],
    **fixed_params,
) -> pd.DataFrame:
    """
    Run a model across a range of parameter values.

    Args:
        model_class: Attribution model class.
        journeys: Journey data.
        param_name: Parameter to vary.
        param_values: List of values to test.
        **fixed_params: Other parameters held constant.

    Returns:
        DataFrame with parameter value × channel attribution results.
    """
    all_results = []

    for val in param_values:
        params = {**fixed_params, param_name: val}
        model = model_class(**params)
        model.fit(journeys)
        result = model.attribute(journeys)

        for ch, credit_pct in result.channel_credit_pct.items():
            all_results.append({
                "param_name": param_name,
                "param_value": val,
                "channel_id": ch,
                "attribution_pct": credit_pct,
                "attributed_conversions": result.channel_credits.get(ch, 0),
            })

    return pd.DataFrame(all_results)


def lookback_window_sensitivity(
    model_class: type,
    journeys: pd.DataFrame,
    windows: List[int] = None,
) -> pd.DataFrame:
    """Test sensitivity to lookback window duration."""
    if windows is None:
        windows = [7, 14, 30, 60, 90]

    results = []

    for window in windows:
        # Filter journeys by duration
        filtered = journeys[
            journeys["journey_duration_days"] <= window
        ].copy() if "journey_duration_days" in journeys.columns else journeys.copy()

        if len(filtered) < 10:
            continue

        model = model_class()
        model.fit(filtered)
        result = model.attribute(filtered)

        for ch, pct in result.channel_credit_pct.items():
            results.append({
                "lookback_window_days": window,
                "channel_id": ch,
                "attribution_pct": pct,
                "n_journeys": len(filtered),
            })

    return pd.DataFrame(results)


def time_decay_sensitivity(
    journeys: pd.DataFrame,
    half_lives: List[float] = None,
) -> pd.DataFrame:
    """Test sensitivity to time decay half-life."""
    from src.models.rule_based import TimeDecayAttribution

    if half_lives is None:
        half_lives = [1.0, 3.0, 7.0, 14.0, 30.0]

    return sensitivity_analysis(
        TimeDecayAttribution, journeys,
        param_name="half_life_days",
        param_values=half_lives,
    )


def markov_order_sensitivity(
    journeys: pd.DataFrame,
) -> pd.DataFrame:
    """Test sensitivity to Markov chain order."""
    from src.models.markov_chain import MarkovChainAttribution

    return sensitivity_analysis(
        MarkovChainAttribution, journeys,
        param_name="order",
        param_values=[1, 2, 3],
    )


def compute_sensitivity_summary(
    sensitivity_df: pd.DataFrame,
    param_name: str,
) -> pd.DataFrame:
    """
    Summarize sensitivity: range, std, and stability score per channel.
    """
    summary = sensitivity_df.groupby("channel_id").agg(
        mean_pct=("attribution_pct", "mean"),
        std_pct=("attribution_pct", "std"),
        min_pct=("attribution_pct", "min"),
        max_pct=("attribution_pct", "max"),
    ).reset_index()

    summary["range_pp"] = (summary["max_pct"] - summary["min_pct"]) * 100
    summary["cv"] = summary["std_pct"] / summary["mean_pct"].clip(lower=0.001)
    summary["stability"] = np.where(
        summary["cv"] < 0.10, "HIGH",
        np.where(summary["cv"] < 0.25, "MODERATE", "LOW")
    )
    summary["param_tested"] = param_name

    return summary
