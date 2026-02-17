"""
Output validator for synthetic data.
Validates generated data against target distributions from Section 4.7.
"""

from typing import Dict, Tuple

import pandas as pd
from loguru import logger

VALIDATION_TARGETS: Dict[str, Dict[str, float]] = {
    "overall_conversion_rate": {"min": 0.020, "max": 0.055},
    "agent_last_touch_pct_of_conversions": {"min": 0.50, "max": 0.75},
    "avg_touchpoints_converting": {"min": 2.8, "max": 5.0},
    "avg_touchpoints_non_converting": {"min": 1.0, "max": 2.6},
    "pct_single_touch_journeys": {"min": 0.25, "max": 0.55},
    "pct_journeys_with_agent": {"min": 0.12, "max": 0.35},
    "agent_conversion_multiplier": {"min": 2.5, "max": 8.0},
    "median_journey_duration_days": {"min": 3.0, "max": 15.0},
    "state_PA_pct": {"min": 0.25, "max": 0.35},
}


def validate_generated_data(
    journeys_df: pd.DataFrame,
) -> Tuple[bool, Dict[str, Dict]]:
    """
    Validate generated data against target distributions.

    Args:
        journeys_df: Journey-level DataFrame.

    Returns:
        (all_passed, results_dict) where results_dict contains per-target details.
    """
    results = {}
    all_passed = True
    total = len(journeys_df)
    converting = journeys_df[journeys_df["is_converting"]]
    non_converting = journeys_df[~journeys_df["is_converting"]]

    # 1. Overall conversion rate
    conv_rate = len(converting) / total if total > 0 else 0
    results["overall_conversion_rate"] = _check(conv_rate, "overall_conversion_rate")

    # 2. Agent last-touch % of conversions
    if len(converting) > 0:
        agent_last = converting[converting["last_touch_channel"] == "independent_agent"]
        agent_last_pct = len(agent_last) / len(converting)
    else:
        agent_last_pct = 0
    results["agent_last_touch_pct_of_conversions"] = _check(
        agent_last_pct, "agent_last_touch_pct_of_conversions"
    )

    # 3. Avg touchpoints — converting
    avg_tp_conv = converting["touchpoint_count"].mean() if len(converting) > 0 else 0
    results["avg_touchpoints_converting"] = _check(avg_tp_conv, "avg_touchpoints_converting")

    # 4. Avg touchpoints — non-converting
    avg_tp_noconv = non_converting["touchpoint_count"].mean() if len(non_converting) > 0 else 0
    results["avg_touchpoints_non_converting"] = _check(
        avg_tp_noconv, "avg_touchpoints_non_converting"
    )

    # 5. Single-touch journeys %
    single = journeys_df[journeys_df["touchpoint_count"] == 1]
    pct_single = len(single) / total if total > 0 else 0
    results["pct_single_touch_journeys"] = _check(pct_single, "pct_single_touch_journeys")

    # 6. Journeys with agent %
    with_agent = journeys_df[journeys_df["has_agent_touch"]]
    pct_agent = len(with_agent) / total if total > 0 else 0
    results["pct_journeys_with_agent"] = _check(pct_agent, "pct_journeys_with_agent")

    # 7. Agent conversion multiplier
    agent_conv_rate = with_agent["is_converting"].mean() if len(with_agent) > 0 else 0
    without_agent = journeys_df[~journeys_df["has_agent_touch"]]
    no_agent_conv_rate = without_agent["is_converting"].mean() if len(without_agent) > 0 else 0.001
    multiplier = agent_conv_rate / max(no_agent_conv_rate, 0.001)
    results["agent_conversion_multiplier"] = _check(multiplier, "agent_conversion_multiplier")

    # 8. Median journey duration
    conv_duration = converting["journey_duration_days"].median() if len(converting) > 0 else 0
    results["median_journey_duration_days"] = _check(
        conv_duration, "median_journey_duration_days"
    )

    # 9. PA state %
    pa_pct = len(journeys_df[journeys_df["state"] == "PA"]) / total if total > 0 else 0
    results["state_PA_pct"] = _check(pa_pct, "state_PA_pct")

    # Log results
    for name, result in results.items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        logger.info(
            f"  {status} {name}: {result['actual']:.4f} "
            f"(target: {result['min']:.3f}–{result['max']:.3f})"
        )
        if not result["passed"]:
            all_passed = False

    return all_passed, results


def _check(actual: float, target_name: str) -> Dict:
    """Check if actual value falls within target range."""
    target = VALIDATION_TARGETS.get(target_name, {"min": 0, "max": 1})
    passed = target["min"] <= actual <= target["max"]
    return {
        "actual": actual,
        "min": target["min"],
        "max": target["max"],
        "passed": passed,
    }
