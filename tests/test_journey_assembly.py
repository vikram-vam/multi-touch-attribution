"""
Tests for journey assembly pipeline.
"""

import sys, os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.sessionizer import sessionize_touchpoints
from src.pipeline.journey_assembler import assemble_journeys
from src.pipeline.touch_qualifier import qualify_touchpoints
from src.pipeline.time_decay import compute_time_decay_weight
from src.pipeline.channel_classifier import classify_channel


def make_touchpoints(n=500):
    """Create synthetic touchpoint data for testing."""
    rng = np.random.default_rng(42)
    channels = ["paid_search_brand", "independent_agent", "display_programmatic",
                "email_marketing", "organic_search"]
    records = []
    for i in range(n):
        records.append({
            "touchpoint_id": f"tp_{i}",
            "journey_id": f"j_{i // 5}",
            "persistent_id": f"p_{i // 5}",
            "channel_id": channels[rng.integers(0, len(channels))],
            "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=rng.integers(0, 8760)),
            "session_id": f"s_{i // 3}",
        })
    return pd.DataFrame(records)


class TestSessionizer:
    def test_assigns_sessions(self):
        tp = make_touchpoints()
        result = sessionize_touchpoints(tp)
        assert "session_id" in result.columns
        assert len(result) == len(tp)


class TestJourneyAssembler:
    def test_assembles_journeys(self):
        tp = make_touchpoints()
        journeys = assemble_journeys(tp)
        assert "journey_id" in journeys.columns
        assert "channel_path" in journeys.columns
        assert len(journeys) > 0

    def test_journey_has_required_fields(self):
        tp = make_touchpoints()
        journeys = assemble_journeys(tp)
        required = ["journey_id", "touchpoint_count", "first_touch_channel",
                    "last_touch_channel"]
        for col in required:
            assert col in journeys.columns, f"Missing: {col}"


class TestTouchQualifier:
    def test_qualifies_touches(self):
        tp = make_touchpoints()
        result = qualify_touchpoints(tp)
        assert len(result) > 0


class TestTimeDecay:
    def test_zero_days_returns_one(self):
        w = compute_time_decay_weight(0.0, 7.0)
        assert w == 1.0

    def test_one_half_life_returns_half(self):
        w = compute_time_decay_weight(7.0, 7.0)
        assert abs(w - 0.5) < 0.01

    def test_recent_touches_get_more_weight(self):
        w_recent = compute_time_decay_weight(1.0, 7.0)
        w_old = compute_time_decay_weight(14.0, 7.0)
        assert w_recent > w_old


class TestChannelClassifier:
    def test_classifies_known_channels(self):
        result = classify_channel("google.com/search", "cpc")
        assert result in ["paid_search_brand", "paid_search_nonbrand"]
