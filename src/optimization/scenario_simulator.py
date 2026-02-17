"""
What-if scenario engine for budget changes.
Allows simulation of budget adjustments and projects impact on conversions.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.optimization.budget_optimizer import (
    optimize_budget, apply_scenario, BUDGET_SCENARIOS,
    response_curve, DEFAULT_RESPONSE_CURVES,
)


@dataclass
class ScenarioResult:
    """Result of running a budget scenario simulation."""
    scenario_name: str
    description: str
    current_spend: Dict[str, float]
    proposed_spend: Dict[str, float]
    current_conversions: float
    projected_conversions: float
    conversion_lift_pct: float
    total_spend_change_pct: float
    channel_changes: pd.DataFrame
    risk_assessment: str


class ScenarioSimulator:
    """
    What-if scenario engine.

    Runs budget scenarios against response curves and projects
    impact on conversions, revenue, and ROI.
    """

    def __init__(self, avg_premium: float = 1200.0):
        self.avg_premium = avg_premium

    def run_scenario(
        self,
        channel_spend: pd.DataFrame,
        attribution_results: pd.DataFrame,
        scenario_name: str,
    ) -> ScenarioResult:
        """Run a named budget scenario and project outcomes."""
        scenario = BUDGET_SCENARIOS.get(scenario_name)
        if scenario is None:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        # Current allocation
        current_by_channel = channel_spend.groupby("channel_id")["spend_dollars"].sum().to_dict()
        current_total = sum(current_by_channel.values())

        # Apply scenario
        adjusted_spend = apply_scenario(channel_spend, scenario_name)
        proposed_by_channel = adjusted_spend.groupby("channel_id")["spend_dollars"].sum().to_dict()
        proposed_total = sum(proposed_by_channel.values())

        # Project conversions using response curves
        current_conv = 0.0
        projected_conv = 0.0
        changes = []

        for ch in current_by_channel:
            curr = current_by_channel.get(ch, 0)
            prop = proposed_by_channel.get(ch, 0)
            params = DEFAULT_RESPONSE_CURVES.get(
                ch, {"alpha": 0.5, "beta": 0.5, "saturation_pct": 2.0}
            )

            curr_c = response_curve(curr, params["alpha"], params["beta"], max(curr, 1))
            prop_c = response_curve(prop, params["alpha"], params["beta"], max(curr, 1))
            current_conv += curr_c
            projected_conv += prop_c

            changes.append({
                "channel_id": ch,
                "current_spend": curr,
                "proposed_spend": prop,
                "spend_change_pct": (prop - curr) / max(curr, 1) * 100,
                "current_projected_conv": curr_c,
                "proposed_projected_conv": prop_c,
                "conv_change_pct": (prop_c - curr_c) / max(curr_c, 0.001) * 100,
            })

        lift = (projected_conv - current_conv) / max(current_conv, 1)
        spend_change = (proposed_total - current_total) / max(current_total, 1) * 100

        # Risk assessment
        max_decrease = min(c["spend_change_pct"] for c in changes) if changes else 0
        if max_decrease < -40:
            risk = "HIGH — Aggressive reductions may disrupt established channels"
        elif max_decrease < -20:
            risk = "MODERATE — Significant shifts require monitoring"
        else:
            risk = "LOW — Conservative adjustments within safe bounds"

        return ScenarioResult(
            scenario_name=scenario_name,
            description=scenario["description"],
            current_spend=current_by_channel,
            proposed_spend=proposed_by_channel,
            current_conversions=current_conv,
            projected_conversions=projected_conv,
            conversion_lift_pct=lift,
            total_spend_change_pct=spend_change,
            channel_changes=pd.DataFrame(changes),
            risk_assessment=risk,
        )

    def run_all_scenarios(
        self,
        channel_spend: pd.DataFrame,
        attribution_results: pd.DataFrame,
    ) -> Dict[str, ScenarioResult]:
        """Run all predefined scenarios."""
        results = {}
        for name in BUDGET_SCENARIOS:
            results[name] = self.run_scenario(
                channel_spend, attribution_results, name
            )
        return results

    def custom_scenario(
        self,
        channel_spend: pd.DataFrame,
        adjustments: Dict[str, float],
    ) -> ScenarioResult:
        """
        Run a custom scenario with user-defined adjustments.

        Args:
            adjustments: Dict of channel_id → multiplier (1.0 = no change).
        """
        current_by_channel = channel_spend.groupby("channel_id")["spend_dollars"].sum().to_dict()
        current_total = sum(current_by_channel.values())

        proposed_by_channel = {}
        for ch, spend in current_by_channel.items():
            mult = adjustments.get(ch, 1.0)
            proposed_by_channel[ch] = spend * mult

        proposed_total = sum(proposed_by_channel.values())

        current_conv = 0.0
        projected_conv = 0.0
        changes = []

        for ch in current_by_channel:
            curr = current_by_channel[ch]
            prop = proposed_by_channel.get(ch, curr)
            params = DEFAULT_RESPONSE_CURVES.get(
                ch, {"alpha": 0.5, "beta": 0.5, "saturation_pct": 2.0}
            )
            curr_c = response_curve(curr, params["alpha"], params["beta"], max(curr, 1))
            prop_c = response_curve(prop, params["alpha"], params["beta"], max(curr, 1))
            current_conv += curr_c
            projected_conv += prop_c
            changes.append({
                "channel_id": ch,
                "current_spend": curr,
                "proposed_spend": prop,
                "spend_change_pct": (prop - curr) / max(curr, 1) * 100,
            })

        lift = (projected_conv - current_conv) / max(current_conv, 1)

        return ScenarioResult(
            scenario_name="custom",
            description="Custom user-defined scenario",
            current_spend=current_by_channel,
            proposed_spend=proposed_by_channel,
            current_conversions=current_conv,
            projected_conversions=projected_conv,
            conversion_lift_pct=lift,
            total_spend_change_pct=(proposed_total - current_total) / max(current_total, 1) * 100,
            channel_changes=pd.DataFrame(changes),
            risk_assessment="USER-DEFINED",
        )
