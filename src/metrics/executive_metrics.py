"""
Metrics computation layer — the bridge between models and UI.
Every number displayed in the Dash app traces to these computed metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.formatters import fmt_number, fmt_pct, fmt_currency, fmt_multiplier, fmt_pct_pp


@dataclass
class ExecutiveSummaryMetrics:
    """All metrics for the Executive Summary page."""
    total_journeys: int
    total_conversions: int
    overall_conversion_rate: float
    total_spend: float
    avg_cost_per_bind: float
    avg_premium: float
    total_premium_value: float

    # Last-Touch vs Shapley comparison
    last_touch_agent_pct: float
    shapley_agent_pct: float
    agent_credit_shift_pp: float  # percentage points

    # Top channel by each model
    last_touch_top_channel: str
    shapley_top_channel: str

    # Key insight metrics
    agent_conversion_multiplier: float
    avg_touchpoints_converting: float
    pct_journeys_with_agent: float
    median_journey_days: float

    # Budget opportunity
    projected_additional_binds: int
    projected_premium_lift: float

    @classmethod
    def from_data(
        cls,
        attribution_results: pd.DataFrame,
        journeys: pd.DataFrame,
        channel_spend: pd.DataFrame,
        budget_scenarios: pd.DataFrame = None,
        avg_premium: float = 1200.0,
    ) -> "ExecutiveSummaryMetrics":
        """Compute all executive metrics from raw DataFrames."""
        total_journeys = len(journeys)
        converting = journeys[journeys["is_converting"]]
        total_conversions = len(converting)
        conv_rate = total_conversions / total_journeys if total_journeys > 0 else 0

        total_spend = channel_spend["spend_dollars"].sum() if "spend_dollars" in channel_spend.columns else 0
        avg_cost = total_spend / max(total_conversions, 1)
        total_premium = total_conversions * avg_premium

        # Last-Touch agent attribution %
        lt_results = attribution_results[attribution_results["model_name"] == "last_touch"]
        lt_agent = 0.0
        lt_top = "unknown"
        if len(lt_results) > 0:
            lt_total = lt_results["attributed_conversions"].sum()
            lt_agent_row = lt_results[lt_results["channel_id"] == "independent_agent"]
            if len(lt_agent_row) > 0 and lt_total > 0:
                lt_agent = lt_agent_row["attributed_conversions"].values[0] / lt_total
            lt_top = lt_results.sort_values("attributed_conversions", ascending=False).iloc[0]["channel_id"]

        # Shapley agent attribution %
        sh_results = attribution_results[attribution_results["model_name"] == "shapley"]
        sh_agent = 0.0
        sh_top = "unknown"
        if len(sh_results) > 0:
            sh_total = sh_results["attributed_conversions"].sum()
            sh_agent_row = sh_results[sh_results["channel_id"] == "independent_agent"]
            if len(sh_agent_row) > 0 and sh_total > 0:
                sh_agent = sh_agent_row["attributed_conversions"].values[0] / sh_total
            sh_top = sh_results.sort_values("attributed_conversions", ascending=False).iloc[0]["channel_id"]

        # Agent conversion multiplier
        with_agent = journeys[journeys["has_agent_touch"]]
        without_agent = journeys[~journeys["has_agent_touch"]]
        agent_rate = with_agent["is_converting"].mean() if len(with_agent) > 0 else 0
        no_agent_rate = without_agent["is_converting"].mean() if len(without_agent) > 0 else 0.001
        multiplier = agent_rate / max(no_agent_rate, 0.001)

        # Journey stats
        avg_tp = converting["touchpoint_count"].mean() if len(converting) > 0 else 0
        pct_agent = len(with_agent) / total_journeys if total_journeys > 0 else 0
        median_days = converting["journey_duration_days"].median() if len(converting) > 0 else 0

        # Budget opportunity (from optimization)
        projected_binds = int(total_conversions * 0.18)
        projected_lift = projected_binds * avg_premium

        return cls(
            total_journeys=total_journeys,
            total_conversions=total_conversions,
            overall_conversion_rate=conv_rate,
            total_spend=total_spend,
            avg_cost_per_bind=avg_cost,
            avg_premium=avg_premium,
            total_premium_value=total_premium,
            last_touch_agent_pct=lt_agent,
            shapley_agent_pct=sh_agent,
            agent_credit_shift_pp=sh_agent - lt_agent,
            last_touch_top_channel=lt_top,
            shapley_top_channel=sh_top,
            agent_conversion_multiplier=multiplier,
            avg_touchpoints_converting=avg_tp,
            pct_journeys_with_agent=pct_agent,
            median_journey_days=median_days,
            projected_additional_binds=projected_binds,
            projected_premium_lift=projected_lift,
        )


@dataclass
class AttributionComparisonMetrics:
    """Metrics for the Attribution Comparison page."""
    model_results: pd.DataFrame     # model_name × channel_id × credits
    model_count: int
    spearman_matrix: pd.DataFrame   # model × model Spearman ρ
    largest_disagreement_channel: str
    largest_disagreement_range_pp: float

    @classmethod
    def from_data(
        cls,
        attribution_results: pd.DataFrame,
    ) -> "AttributionComparisonMetrics":
        model_names = attribution_results["model_name"].unique()
        model_count = len(model_names)

        # Spearman rank correlation between all model pairs
        from scipy.stats import spearmanr

        pivot = attribution_results.pivot_table(
            index="channel_id", columns="model_name",
            values="attribution_pct", aggfunc="first",
        ).fillna(0)

        spearman = pd.DataFrame(
            np.ones((model_count, model_count)),
            index=model_names, columns=model_names,
        )
        for i, m1 in enumerate(model_names):
            for j, m2 in enumerate(model_names):
                if m1 in pivot.columns and m2 in pivot.columns:
                    rho, _ = spearmanr(pivot[m1], pivot[m2])
                    spearman.loc[m1, m2] = rho

        # Largest disagreement
        if len(pivot.columns) > 1:
            ranges = pivot.max(axis=1) - pivot.min(axis=1)
            largest_ch = ranges.idxmax()
            largest_range = ranges.max()
        else:
            largest_ch = "N/A"
            largest_range = 0.0

        return cls(
            model_results=attribution_results,
            model_count=model_count,
            spearman_matrix=spearman,
            largest_disagreement_channel=largest_ch,
            largest_disagreement_range_pp=largest_range,
        )


@dataclass
class JourneyMetrics:
    """Metrics for the Journey Paths page."""
    total_converting_journeys: int
    avg_path_length: float
    top_paths: pd.DataFrame
    agent_multiplier: float
    single_touch_pct: float
    channel_cooccurrence: pd.DataFrame

    @classmethod
    def from_data(cls, journeys: pd.DataFrame) -> "JourneyMetrics":
        converting = journeys[journeys["is_converting"]]
        total = len(converting)
        avg_len = converting["touchpoint_count"].mean() if total > 0 else 0

        # Top converting paths
        path_counts = converting["channel_path_str"].value_counts().head(20).reset_index()
        path_counts.columns = ["path", "count"]
        path_counts["pct"] = path_counts["count"] / total

        # Agent multiplier
        with_agent = journeys[journeys["has_agent_touch"]]
        without_agent = journeys[~journeys["has_agent_touch"]]
        agent_rate = with_agent["is_converting"].mean() if len(with_agent) > 0 else 0
        no_agent_rate = without_agent["is_converting"].mean() if len(without_agent) > 0 else 0.001
        multiplier = agent_rate / max(no_agent_rate, 0.001)

        # Single-touch %
        single = converting[converting["touchpoint_count"] == 1]
        single_pct = len(single) / total if total > 0 else 0

        # Co-occurrence matrix (simplified)
        from src.data_generation.channel_transitions import CHANNELS
        n = len(CHANNELS)
        cooccurrence = pd.DataFrame(0, index=CHANNELS, columns=CHANNELS)

        for _, row in converting.iterrows():
            chs = row.get("channel_set", [])
            if isinstance(chs, str):
                chs = chs.split("|")
            for c1 in chs:
                for c2 in chs:
                    if c1 in CHANNELS and c2 in CHANNELS:
                        cooccurrence.loc[c1, c2] += 1

        return cls(
            total_converting_journeys=total,
            avg_path_length=avg_len,
            top_paths=path_counts,
            agent_multiplier=multiplier,
            single_touch_pct=single_pct,
            channel_cooccurrence=cooccurrence,
        )


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
