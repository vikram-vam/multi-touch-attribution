"""
Tests for budget optimization.
"""

import sys, os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimization.budget_optimizer import (
    optimize_budget, apply_scenario, response_curve,
    marginal_roi, BUDGET_SCENARIOS, DEFAULT_RESPONSE_CURVES,
)
from src.optimization.marginal_roi import (
    compute_response_curve_points, find_saturation_point,
)
from src.optimization.constraints import (
    get_constraints, apply_constraints, ERIE_CONSTRAINTS,
)
from src.optimization.scenario_simulator import ScenarioSimulator


def make_spend_data():
    channels = list(DEFAULT_RESPONSE_CURVES.keys())
    records = []
    for ch in channels:
        for m in range(1, 13):
            records.append({
                "channel_id": ch,
                "month": f"2024-{m:02d}",
                "spend_dollars": np.random.uniform(5000, 50000),
            })
    return pd.DataFrame(records)


def make_attribution_data():
    channels = list(DEFAULT_RESPONSE_CURVES.keys())
    records = []
    for ch in channels:
        records.append({
            "model_name": "shapley",
            "channel_id": ch,
            "attributed_conversions": np.random.uniform(50, 500),
            "attribution_pct": 1.0 / len(channels),
        })
    return pd.DataFrame(records)


class TestResponseCurve:
    def test_zero_spend_returns_zero(self):
        assert response_curve(0, 0.5, 0.5, 10000) == 0.0

    def test_positive_spend_returns_positive(self):
        result = response_curve(10000, 0.5, 0.5, 10000)
        assert result > 0

    def test_diminishing_returns(self):
        r1 = response_curve(10000, 0.5, 0.5, 10000)
        r2 = response_curve(20000, 0.5, 0.5, 10000)
        r3 = response_curve(30000, 0.5, 0.5, 10000)
        # Diminishing: r2-r1 > r3-r2
        assert (r2 - r1) > (r3 - r2)


class TestMarginalROI:
    def test_positive(self):
        mr = marginal_roi(10000, 0.5, 0.5, 10000)
        assert mr > 0

    def test_decreasing(self):
        mr1 = marginal_roi(10000, 0.5, 0.5, 10000)
        mr2 = marginal_roi(50000, 0.5, 0.5, 10000)
        assert mr1 > mr2


class TestBudgetOptimizer:
    def test_optimize_returns_dataframe(self):
        spend = make_spend_data()
        attr = make_attribution_data()
        result = optimize_budget(spend, attr)
        assert isinstance(result, pd.DataFrame)
        assert "optimal_spend" in result.columns

    def test_all_channels_present(self):
        spend = make_spend_data()
        attr = make_attribution_data()
        result = optimize_budget(spend, attr)
        assert len(result) == len(DEFAULT_RESPONSE_CURVES)


class TestScenarios:
    def test_all_scenarios_exist(self):
        assert len(BUDGET_SCENARIOS) >= 6

    def test_apply_scenario(self):
        spend = make_spend_data()
        result = apply_scenario(spend, "status_quo")
        assert len(result) == len(spend)


class TestConstraints:
    def test_agent_has_high_floor(self):
        c = get_constraints("independent_agent")
        assert c.min_spend_pct >= 0.80

    def test_direct_organic_locked(self):
        c = get_constraints("direct_organic")
        assert c.locked is True


class TestSaturation:
    def test_saturation_found(self):
        sat = find_saturation_point("paid_search_brand", 50000)
        assert sat > 0
