"""
Budget optimizer metrics.
Computes all metrics for the Budget Optimizer page (EP-4).
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class BudgetMetrics:
    """Metrics for the Budget Optimizer page."""
    current_total_spend: float
    optimal_total_spend: float
    channel_comparison: pd.DataFrame
    projected_lift_pct: float
    projected_additional_binds: int
    projected_additional_premium: float

    @classmethod
    def from_data(
        cls,
        budget_results: pd.DataFrame,
        avg_premium: float = 1200.0,
    ) -> "BudgetMetrics":
        current_total = budget_results["current_spend"].sum()
        optimal_total = budget_results["optimal_spend"].sum()
        current_conv = budget_results["current_attributed_conversions"].sum()
        projected_conv = budget_results["projected_conversions"].sum()
        lift = (projected_conv - current_conv) / max(current_conv, 1)
        add_binds = int(projected_conv - current_conv)
        add_premium = add_binds * avg_premium

        return cls(
            current_total_spend=current_total,
            optimal_total_spend=optimal_total,
            channel_comparison=budget_results,
            projected_lift_pct=lift,
            projected_additional_binds=max(add_binds, 0),
            projected_additional_premium=max(add_premium, 0),
        )
