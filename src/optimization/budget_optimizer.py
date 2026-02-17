"""
Budget optimizer using LP/IP (Section 8.2).
Optimizes spend allocation based on attribution-derived channel ROI.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Diminishing returns response curve parameters
DEFAULT_RESPONSE_CURVES = {
    "independent_agent":     {"alpha": 0.65, "beta": 0.45, "saturation_pct": 1.5},
    "paid_search_brand":     {"alpha": 0.80, "beta": 0.50, "saturation_pct": 1.8},
    "paid_search_nonbrand":  {"alpha": 0.70, "beta": 0.55, "saturation_pct": 2.0},
    "organic_search":        {"alpha": 0.90, "beta": 0.30, "saturation_pct": 2.5},
    "display_programmatic":  {"alpha": 0.50, "beta": 0.60, "saturation_pct": 2.0},
    "paid_social":           {"alpha": 0.55, "beta": 0.55, "saturation_pct": 1.8},
    "tv_radio":              {"alpha": 0.40, "beta": 0.70, "saturation_pct": 1.5},
    "direct_mail":           {"alpha": 0.60, "beta": 0.50, "saturation_pct": 1.6},
    "email_marketing":       {"alpha": 0.85, "beta": 0.35, "saturation_pct": 3.0},
    "call_center":           {"alpha": 0.70, "beta": 0.45, "saturation_pct": 1.5},
    "aggregator_comparator": {"alpha": 0.55, "beta": 0.60, "saturation_pct": 2.0},
    "direct_organic":        {"alpha": 0.95, "beta": 0.20, "saturation_pct": 3.0},
    "video_ott_ctv":         {"alpha": 0.45, "beta": 0.65, "saturation_pct": 1.8},
}


def response_curve(spend: float, alpha: float, beta: float,
                    current_spend: float) -> float:
    """
    Hill function response curve: conversions = alpha * (spend^beta) / (spend^beta + K^beta)

    Where K = current_spend (half-saturation point).
    """
    if spend <= 0:
        return 0.0
    k = max(current_spend, 1.0)
    return alpha * (spend ** beta) / (spend ** beta + k ** beta)


def marginal_roi(spend: float, alpha: float, beta: float,
                 current_spend: float, delta: float = 1000) -> float:
    """Marginal ROI at given spend level (derivative approximation)."""
    r1 = response_curve(spend, alpha, beta, current_spend)
    r2 = response_curve(spend + delta, alpha, beta, current_spend)
    return (r2 - r1) / delta


def optimize_budget(
    channel_spend: pd.DataFrame,
    attribution_results: pd.DataFrame,
    total_budget: Optional[float] = None,
    min_spend_pct: float = 0.50,
    max_spend_pct: float = 2.00,
    avg_premium: float = 1200.0,
) -> pd.DataFrame:
    """
    Optimize budget allocation using greedy marginal allocation.

    Args:
        channel_spend: Current channel spend DataFrame.
        attribution_results: Attribution results with channel credits.
        total_budget: Total budget constraint (None = current total).
        min_spend_pct: Minimum spend as fraction of current (floor).
        max_spend_pct: Maximum spend as fraction of current (ceiling).
        avg_premium: Average premium for value calculation.

    Returns:
        DataFrame with current vs optimal allocation per channel.
    """
    # Aggregate current annual spend by channel
    current_spend = channel_spend.groupby("channel_id")["spend_dollars"].sum().to_dict()

    if total_budget is None:
        total_budget = sum(current_spend.values())

    # Get attribution credits
    if isinstance(attribution_results, pd.DataFrame) and "channel_id" in attribution_results.columns:
        credits = attribution_results.groupby("channel_id")["attributed_conversions"].sum().to_dict()
    else:
        credits = {ch: 1.0 for ch in current_spend}

    # Compute ROI for each channel
    channel_roi = {}
    for ch in current_spend:
        spend = current_spend.get(ch, 1.0)
        credit = credits.get(ch, 0.0)
        roi = (credit * avg_premium) / max(spend, 1.0)
        channel_roi[ch] = roi

    # Greedy marginal allocation
    allocation_step = total_budget / 100
    optimal_spend = {ch: max(current_spend.get(ch, 0) * min_spend_pct, allocation_step)
                     for ch in current_spend}

    remaining_budget = total_budget - sum(optimal_spend.values())

    while remaining_budget > allocation_step:
        # Find channel with highest marginal ROI
        best_channel = None
        best_marginal = -1

        for ch in current_spend:
            if optimal_spend[ch] >= current_spend.get(ch, 0) * max_spend_pct:
                continue  # At ceiling

            params = DEFAULT_RESPONSE_CURVES.get(ch, {"alpha": 0.5, "beta": 0.5, "saturation_pct": 2.0})
            mr = marginal_roi(
                optimal_spend[ch], params["alpha"], params["beta"],
                current_spend.get(ch, 1.0),
            )
            if mr > best_marginal:
                best_marginal = mr
                best_channel = ch

        if best_channel is None:
            break

        optimal_spend[best_channel] += allocation_step
        remaining_budget -= allocation_step

    # Build output DataFrame
    results = []
    for ch in current_spend:
        curr = current_spend.get(ch, 0)
        opt = optimal_spend.get(ch, 0)
        curr_credit = credits.get(ch, 0)
        params = DEFAULT_RESPONSE_CURVES.get(ch, {"alpha": 0.5, "beta": 0.5, "saturation_pct": 2.0})
        opt_credit = response_curve(opt, params["alpha"], params["beta"], max(curr, 1.0)) * curr_credit * 2

        results.append({
            "channel_id": ch,
            "current_spend": curr,
            "optimal_spend": opt,
            "spend_change_pct": (opt - curr) / max(curr, 1) * 100,
            "current_attributed_conversions": curr_credit,
            "projected_conversions": opt_credit,
            "current_roi": channel_roi.get(ch, 0),
            "current_cost_per_bind": curr / max(curr_credit, 1),
        })

    return pd.DataFrame(results)


# Pre-defined scenarios (Section 8.3)
BUDGET_SCENARIOS = {
    "status_quo": {
        "description": "Current allocation — no changes",
        "adjustment": {},
    },
    "shapley_optimal": {
        "description": "Fully Shapley-aligned reallocation",
        "adjustment": {
            "independent_agent": 1.30, "paid_search_brand": 0.85,
            "display_programmatic": 1.20, "paid_social": 1.15,
            "tv_radio": 0.90, "direct_mail": 1.10,
        },
    },
    "agent_investment": {
        "description": "Increase agent support +30%, funded by brand search",
        "adjustment": {
            "independent_agent": 1.30, "paid_search_brand": 0.70,
        },
    },
    "digital_shift": {
        "description": "Shift 20% from TV/radio to digital channels",
        "adjustment": {
            "tv_radio": 0.80, "display_programmatic": 1.25,
            "paid_social": 1.20, "video_ott_ctv": 1.30,
        },
    },
    "efficiency_focus": {
        "description": "Cut 15% total budget, optimize remaining allocation",
        "adjustment": {"_total_multiplier": 0.85},
    },
    "growth_mode": {
        "description": "Increase budget 15%, maximize conversions",
        "adjustment": {"_total_multiplier": 1.15},
    },
}


def apply_scenario(
    channel_spend: pd.DataFrame,
    scenario_name: str,
) -> pd.DataFrame:
    """Apply a named scenario to current spend data."""
    scenario = BUDGET_SCENARIOS.get(scenario_name)
    if scenario is None:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    df = channel_spend.copy()
    adj = scenario["adjustment"]

    # Total budget multiplier
    total_mult = adj.pop("_total_multiplier", 1.0)
    df["spend_dollars"] *= total_mult

    # Channel-specific adjustments
    for ch, mult in adj.items():
        mask = df["channel_id"] == ch
        df.loc[mask, "spend_dollars"] *= mult

    return df
