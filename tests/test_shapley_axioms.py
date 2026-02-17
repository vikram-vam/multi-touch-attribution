"""
Tests for Shapley axiom compliance.
"""

import sys, os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.shapley_engine import ShapleyAttribution
from src.models.shapley_casv import CASVAttribution
from src.validation.axiom_tests import (
    test_efficiency_axiom, test_symmetry_axiom,
    test_null_player_axiom, run_all_axiom_tests,
)


def make_test_journeys():
    """Create minimal test journey data."""
    import pandas as pd
    rng = np.random.default_rng(42)
    n = 200
    channels = ["paid_search_brand", "independent_agent", "display_programmatic",
                "email_marketing", "organic_search"]
    journeys = []
    for i in range(n):
        path_len = rng.integers(1, 6)
        path = [channels[rng.integers(0, len(channels))] for _ in range(path_len)]
        is_conv = rng.random() < 0.15
        journeys.append({
            "journey_id": f"j_{i}",
            "persistent_id": f"p_{i}",
            "is_converting": is_conv,
            "conversion_value": 1200.0 if is_conv else 0.0,
            "touchpoint_count": path_len,
            "channel_path": path,
            "channel_path_str": "|".join(path),
            "channel_set": sorted(set(path)),
            "channel_set_str": "|".join(sorted(set(path))),
            "distinct_channel_count": len(set(path)),
            "first_touch_channel": path[0],
            "last_touch_channel": path[-1],
            "has_agent_touch": "independent_agent" in path,
            "journey_duration_days": rng.integers(1, 60),
        })
    return pd.DataFrame(journeys)


class TestShapleyEfficiency:
    """Shapley credits should sum to total conversions."""

    def test_efficiency_exact(self):
        journeys = make_test_journeys()
        model = ShapleyAttribution()
        model.fit(journeys)
        result = model.attribute(journeys)
        test_result = test_efficiency_axiom(result)
        assert test_result.passed, f"Efficiency failed: {test_result.details}"

    def test_efficiency_casv(self):
        journeys = make_test_journeys()
        model = CASVAttribution()
        model.fit(journeys)
        result = model.attribute(journeys)
        test_result = test_efficiency_axiom(result, tolerance=0.05)
        assert test_result.passed, f"CASV efficiency: {test_result.details}"


class TestShapleySymmetry:
    """Symmetric channels should get equal credit."""

    def test_symmetry_trivial(self):
        journeys = make_test_journeys()
        model = ShapleyAttribution()
        model.fit(journeys)
        result = model.attribute(journeys)
        test_result = test_symmetry_axiom(result)
        assert test_result.passed


class TestShapleyNullPlayer:
    """Non-contributing channels should get zero credit."""

    def test_null_player(self):
        journeys = make_test_journeys()
        model = ShapleyAttribution()
        model.fit(journeys)
        result = model.attribute(journeys)
        test_result = test_null_player_axiom(result)
        assert test_result.passed


class TestAllAxioms:
    """Run all axiom tests together."""

    def test_all_pass(self):
        journeys = make_test_journeys()
        model = ShapleyAttribution()
        model.fit(journeys)
        result = model.attribute(journeys)
        results = run_all_axiom_tests(result)
        for r in results:
            assert r.passed, f"{r.axiom_name} failed: {r.details}"
