"""
Tests for Markov Chain attribution models.
"""

import sys, os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.markov_chain import MarkovChainAttribution
from src.models.markov_variable_order import VariableOrderMarkovAttribution


def make_test_journeys(n=300):
    rng = np.random.default_rng(42)
    channels = ["paid_search_brand", "independent_agent", "display_programmatic",
                "email_marketing", "organic_search"]
    journeys = []
    for i in range(n):
        path_len = rng.integers(1, 6)
        path = [channels[rng.integers(0, len(channels))] for _ in range(path_len)]
        is_conv = rng.random() < 0.15
        journeys.append({
            "journey_id": f"j_{i}", "persistent_id": f"p_{i}",
            "is_converting": is_conv, "conversion_value": 1200.0 if is_conv else 0,
            "touchpoint_count": path_len, "channel_path": path,
            "channel_path_str": "|".join(path),
            "channel_set": sorted(set(path)),
            "first_touch_channel": path[0], "last_touch_channel": path[-1],
            "has_agent_touch": "independent_agent" in path,
        })
    return pd.DataFrame(journeys)


class TestMarkovOrder1:
    def test_fit_and_attribute(self):
        journeys = make_test_journeys()
        model = MarkovChainAttribution(order=1)
        model.fit(journeys)
        result = model.attribute(journeys)
        assert result.model_name == "markov_order_1"
        assert result.tier == "probabilistic"
        assert sum(result.channel_credits.values()) > 0

    def test_credits_sum_to_conversions(self):
        journeys = make_test_journeys()
        model = MarkovChainAttribution(order=1)
        model.fit(journeys)
        result = model.attribute(journeys)
        total_credit = sum(result.channel_credits.values())
        total_conv = result.total_conversions
        assert abs(total_credit - total_conv) / max(total_conv, 1) < 0.1


class TestMarkovOrder2:
    def test_fit_and_attribute(self):
        journeys = make_test_journeys()
        model = MarkovChainAttribution(order=2)
        model.fit(journeys)
        result = model.attribute(journeys)
        assert result.model_name == "markov_order_2"

    def test_removal_effects_non_negative(self):
        journeys = make_test_journeys()
        model = MarkovChainAttribution(order=2)
        model.fit(journeys)
        result = model.attribute(journeys)
        for ch, credit in result.channel_credits.items():
            assert credit >= 0, f"Negative credit for {ch}: {credit}"


class TestVariableOrderMarkov:
    def test_fit_and_attribute(self):
        journeys = make_test_journeys()
        model = VariableOrderMarkovAttribution(max_order=2)
        model.fit(journeys)
        result = model.attribute(journeys)
        assert result.model_name == "markov_variable_order"

    def test_mixing_weights_sum_to_one(self):
        journeys = make_test_journeys()
        model = VariableOrderMarkovAttribution(max_order=2)
        model.fit(journeys)
        total = sum(model._mixing_weights.values())
        assert abs(total - 1.0) < 0.01
