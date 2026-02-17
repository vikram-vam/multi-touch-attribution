"""
Tests for synthetic data generation.
"""

import sys, os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation.user_profiles import generate_user_profiles
from src.data_generation.channel_transitions import CHANNELS, build_transition_matrix
from src.data_generation.spend_generator import generate_channel_spend
from src.data_generation.timestamp_engine import generate_touch_timestamps
from src.data_generation.identity_simulator import generate_identity_graph


class TestUserProfiles:
    def test_generates_correct_count(self):
        profiles = generate_user_profiles(100)
        assert len(profiles) == 100

    def test_has_required_columns(self):
        profiles = generate_user_profiles(50)
        required = ["user_id", "age_group", "state", "is_homeowner"]
        for col in required:
            assert col in profiles.columns, f"Missing column: {col}"

    def test_states_in_erie_footprint(self):
        profiles = generate_user_profiles(200)
        # Erie operates in 12 states + DC
        valid_states = {
            "PA", "OH", "IN", "IL", "WI", "NY", "NC", "VA",
            "WV", "TN", "KY", "MD", "DC"
        }
        actual_states = set(profiles["state"].unique())
        assert actual_states.issubset(valid_states)


class TestChannelTransitions:
    def test_channel_count(self):
        assert len(CHANNELS) == 13

    def test_transition_matrix_rows_sum_to_one(self):
        matrix = build_transition_matrix()
        for state, transitions in matrix.items():
            total = sum(transitions.values())
            assert abs(total - 1.0) < 0.01, f"Row {state} sums to {total}"


class TestSpendGenerator:
    def test_generates_12_months(self):
        spend = generate_channel_spend()
        months = spend["month"].nunique()
        assert months == 12

    def test_all_channels_present(self):
        spend = generate_channel_spend()
        channels = set(spend["channel_id"].unique())
        assert channels == set(CHANNELS)

    def test_no_negative_spend(self):
        spend = generate_channel_spend()
        assert (spend["spend_dollars"] >= 0).all()


class TestIdentitySimulator:
    def test_generates_fragments(self):
        graph = generate_identity_graph([f'p_{i}' for i in range(100)], np.random.default_rng(42))
        assert len(graph) > 0

    def test_has_confidence_scores(self):
        graph = generate_identity_graph([f'p_{i}' for i in range(50)], np.random.default_rng(42))
        assert "match_confidence" in graph.columns
        assert (graph["match_confidence"] >= 0).all()
        assert (graph["match_confidence"] <= 1).all()
