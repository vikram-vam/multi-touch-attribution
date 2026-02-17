"""
Business logic sanity checks.
Validates attribution results against known Erie-specific constraints.
"""

from typing import Dict, List
from dataclasses import dataclass

import pandas as pd


@dataclass
class SanityCheckResult:
    """Result of a single sanity check."""
    check_name: str
    passed: bool
    actual_value: float
    expected_range: str
    details: str
    severity: str  # "error", "warning", "info"


def check_agent_attribution_range(
    attribution_results: pd.DataFrame,
    model_name: str = "ensemble",
    min_pct: float = 0.15,
    max_pct: float = 0.50,
) -> SanityCheckResult:
    """Agent attribution should be between 15-50% for Erie."""
    model_results = attribution_results[attribution_results["model_name"] == model_name]
    if len(model_results) == 0:
        return SanityCheckResult(
            "Agent Attribution Range", False, 0.0,
            f"{min_pct:.0%}-{max_pct:.0%}", f"Model '{model_name}' not found", "error"
        )
    agent_row = model_results[model_results["channel_id"] == "independent_agent"]
    agent_pct = agent_row["attribution_pct"].values[0] if len(agent_row) > 0 else 0
    passed = min_pct <= agent_pct <= max_pct
    return SanityCheckResult(
        "Agent Attribution Range", passed, agent_pct,
        f"{min_pct:.0%}-{max_pct:.0%}",
        f"Agent gets {agent_pct:.1%} credit ({model_name})",
        "warning" if not passed else "info",
    )


def check_no_zero_channels(
    attribution_results: pd.DataFrame,
    model_name: str = "ensemble",
) -> SanityCheckResult:
    """All 13 channels should have some attribution credit."""
    model_results = attribution_results[attribution_results["model_name"] == model_name]
    zero_channels = model_results[model_results["attribution_pct"] <= 0.001]
    n_zero = len(zero_channels)
    return SanityCheckResult(
        "No Zero-Credit Channels", n_zero == 0, float(n_zero),
        "0 channels with zero credit",
        f"{n_zero} channels have zero credit" if n_zero > 0 else "All channels credited",
        "warning" if n_zero > 0 else "info",
    )


def check_credit_sums_to_one(
    attribution_results: pd.DataFrame,
    tolerance: float = 0.01,
) -> SanityCheckResult:
    """Attribution percentages should sum to ~100% per model."""
    sums = attribution_results.groupby("model_name")["attribution_pct"].sum()
    bad_models = sums[abs(sums - 1.0) > tolerance]
    n_bad = len(bad_models)
    return SanityCheckResult(
        "Credits Sum to 100%", n_bad == 0, float(n_bad),
        "All models sum to 100%+/-1%",
        f"{n_bad} models don't sum to 100%" if n_bad > 0 else "All models pass",
        "error" if n_bad > 0 else "info",
    )


def check_conversion_rate_plausible(
    journeys: pd.DataFrame,
    min_rate: float = 0.05,
    max_rate: float = 0.30,
) -> SanityCheckResult:
    """Conversion rate should be between 5-30% for P&C auto."""
    rate = journeys["is_converting"].mean() if "is_converting" in journeys.columns else 0
    return SanityCheckResult(
        "Conversion Rate Plausible", min_rate <= rate <= max_rate, rate,
        f"{min_rate:.0%}-{max_rate:.0%}", f"Conversion rate: {rate:.1%}",
        "error" if not (min_rate <= rate <= max_rate) else "info",
    )


def check_last_touch_vs_shapley_divergence(
    attribution_results: pd.DataFrame,
    min_divergence: float = 0.05,
) -> SanityCheckResult:
    """Last-touch and Shapley should meaningfully disagree."""
    lt = attribution_results[attribution_results["model_name"] == "last_touch"]
    sh = attribution_results[attribution_results["model_name"] == "shapley"]
    if len(lt) == 0 or len(sh) == 0:
        return SanityCheckResult(
            "LT vs Shapley Divergence", False, 0.0,
            f">{min_divergence:.0%}", "Models not found", "error",
        )
    merged = lt.merge(sh, on="channel_id", suffixes=("_lt", "_sh"))
    avg_diff = abs(merged["attribution_pct_lt"] - merged["attribution_pct_sh"]).mean()
    return SanityCheckResult(
        "LT vs Shapley Divergence", avg_diff >= min_divergence, avg_diff,
        f">{min_divergence:.0%}", f"Average divergence: {avg_diff:.1%}",
        "warning" if avg_diff < min_divergence else "info",
    )


def run_all_sanity_checks(
    attribution_results: pd.DataFrame,
    journeys: pd.DataFrame,
) -> List[SanityCheckResult]:
    """Run all business logic sanity checks."""
    return [
        check_agent_attribution_range(attribution_results),
        check_no_zero_channels(attribution_results),
        check_credit_sums_to_one(attribution_results),
        check_conversion_rate_plausible(journeys),
        check_last_touch_vs_shapley_divergence(attribution_results),
    ]


def sanity_check_summary(results: List[SanityCheckResult]) -> pd.DataFrame:
    """Summarize sanity check results."""
    records = []
    for r in results:
        records.append({
            "check": r.check_name,
            "status": "PASS" if r.passed else "FAIL",
            "value": r.actual_value,
            "expected": r.expected_range,
            "severity": r.severity,
            "details": r.details,
        })
    return pd.DataFrame(records)
