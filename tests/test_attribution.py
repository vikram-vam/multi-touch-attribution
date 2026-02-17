"""
Test suite for the Erie MCA attribution models and pipeline.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation.channel_transitions import CHANNELS


# ── Fixtures ──

@pytest.fixture
def sample_journeys():
    """Create minimal journey DataFrame for testing."""
    np.random.seed(42)
    n = 200
    channel_list = list(CHANNELS)

    journeys = []
    for i in range(n):
        path_len = np.random.randint(1, 8)
        path = [np.random.choice(channel_list) for _ in range(path_len)]
        is_conv = np.random.random() < 0.15
        journeys.append({
            "journey_id": f"j_{i}",
            "persistent_id": f"u_{i}",
            "channel_path": path,
            "channel_path_str": "|".join(path),
            "channel_set": sorted(set(path)),
            "channel_set_str": "|".join(sorted(set(path))),
            "touchpoint_count": len(path),
            "distinct_channel_count": len(set(path)),
            "journey_start": pd.Timestamp("2024-01-01"),
            "journey_end": pd.Timestamp("2024-02-01"),
            "journey_duration_days": 31.0,
            "first_touch_channel": path[0],
            "last_touch_channel": path[-1],
            "is_converting": is_conv,
            "conversion_value": 1200.0 if is_conv else 0.0,
            "has_agent_touch": "independent_agent" in path,
            "agent_touch_position": "last" if path[-1] == "independent_agent" else "none",
            "touch_weights": [1.0] * len(path),
        })

    return pd.DataFrame(journeys)


@pytest.fixture
def sample_touchpoints():
    """Create minimal touchpoint DataFrame."""
    np.random.seed(42)
    records = []
    for i in range(500):
        records.append({
            "touchpoint_id": f"tp_{i}",
            "persistent_id": f"u_{i % 100}",
            "channel_id": np.random.choice(list(CHANNELS)),
            "touch_type": np.random.choice(["impression", "click", "agent"]),
            "event_timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i),
            "viewability_pct": np.random.uniform(0.3, 1.0),
            "dwell_time_seconds": np.random.uniform(0.5, 30),
        })
    return pd.DataFrame(records)


# ── Rule-Based Model Tests ──

class TestRuleBasedModels:
    def test_first_touch(self, sample_journeys):
        from src.models.rule_based import FirstTouchAttribution
        model = FirstTouchAttribution()
        model.fit(sample_journeys)
        result = model.attribute(sample_journeys)

        assert result.model_name == "first_touch"
        assert sum(result.channel_credits.values()) > 0

    def test_last_touch(self, sample_journeys):
        from src.models.rule_based import LastTouchAttribution
        model = LastTouchAttribution()
        model.fit(sample_journeys)
        result = model.attribute(sample_journeys)

        assert result.model_name == "last_touch"
        total_credit = sum(result.channel_credits.values())
        total_conv = sample_journeys["is_converting"].sum()
        assert abs(total_credit - total_conv) < 0.01  # Efficiency

    def test_linear(self, sample_journeys):
        from src.models.rule_based import LinearAttribution
        model = LinearAttribution()
        model.fit(sample_journeys)
        result = model.attribute(sample_journeys)

        total_credit = sum(result.channel_credits.values())
        total_conv = sample_journeys["is_converting"].sum()
        assert abs(total_credit - total_conv) < 0.1

    def test_time_decay(self, sample_journeys):
        from src.models.rule_based import TimeDecayAttribution
        model = TimeDecayAttribution(half_life_days=7.0)
        model.fit(sample_journeys)
        result = model.attribute(sample_journeys)
        assert result.model_name == "time_decay"

    def test_position_based(self, sample_journeys):
        from src.models.rule_based import PositionBasedAttribution
        model = PositionBasedAttribution()
        model.fit(sample_journeys)
        result = model.attribute(sample_journeys)
        assert abs(sum(result.channel_credit_pct.values()) - 1.0) < 0.01


# ── Shapley Tests ──

class TestShapley:
    def test_shapley_basic(self, sample_journeys):
        from src.models.shapley_engine import ShapleyAttribution
        model = ShapleyAttribution(mc_samples=100)
        model.fit(sample_journeys)
        result = model.attribute(sample_journeys)

        assert result.model_name == "shapley"
        assert all(v >= 0 for v in result.channel_credits.values())

    def test_shapley_efficiency(self, sample_journeys):
        from src.models.shapley_engine import ShapleyAttribution
        model = ShapleyAttribution(mc_samples=100)
        model.fit(sample_journeys)
        result = model.attribute(sample_journeys)

        total = sum(result.channel_credits.values())
        expected = sample_journeys["is_converting"].sum()
        assert abs(total - expected) < 1.0  # Within 1 conversion


# ── Markov Tests ──

class TestMarkov:
    def test_markov_order_1(self, sample_journeys):
        from src.models.markov_chain import MarkovChainAttribution
        model = MarkovChainAttribution(order=1)
        model.fit(sample_journeys)
        result = model.attribute(sample_journeys)
        assert result.model_name == "markov_order_1"

    def test_markov_order_2(self, sample_journeys):
        from src.models.markov_chain import MarkovChainAttribution
        model = MarkovChainAttribution(order=2)
        model.fit(sample_journeys)
        result = model.attribute(sample_journeys)
        assert result.model_name == "markov_order_2"


# ── Ensemble Tests ──

class TestEnsemble:
    def test_ensemble(self, sample_journeys):
        from src.models.rule_based import LastTouchAttribution
        from src.models.shapley_engine import ShapleyAttribution
        from src.models.ensemble import EnsembleAttribution

        lt = LastTouchAttribution()
        lt.fit(sample_journeys)
        lt_result = lt.attribute(sample_journeys)

        sh = ShapleyAttribution(mc_samples=100)
        sh.fit(sample_journeys)
        sh_result = sh.attribute(sample_journeys)

        ensemble = EnsembleAttribution(
            model_results={lt_result.model_name: lt_result, sh_result.model_name: sh_result},
            weights={"last_touch": 0.5, "shapley": 0.5},
        )
        result = ensemble.attribute(sample_journeys)
        assert abs(sum(result.channel_credit_pct.values()) - 1.0) < 0.01


# ── Pipeline Tests ──

class TestPipeline:
    def test_sessionizer(self, sample_touchpoints):
        from src.pipeline.sessionizer import sessionize_touchpoints
        result = sessionize_touchpoints(sample_touchpoints)
        assert "session_id" in result.columns
        assert len(result) == len(sample_touchpoints)

    def test_touch_qualifier(self, sample_touchpoints):
        from src.pipeline.touch_qualifier import qualify_touchpoints
        result = qualify_touchpoints(sample_touchpoints)
        assert "is_qualified" in result.columns
        assert "touch_weight" in result.columns

    def test_channel_classifier(self, sample_touchpoints):
        from src.pipeline.channel_classifier import ensure_mece_classification
        result = ensure_mece_classification(sample_touchpoints)
        assert all(ch in CHANNELS for ch in result["channel_id"].unique())


# ── Model Registry Tests ──

class TestModelRegistry:
    def test_get_model(self):
        from src.models.model_registry import get_model, MODEL_REGISTRY
        for name in ["first_touch", "last_touch", "shapley", "ensemble"]:
            model = get_model(name)
            assert model is not None

    def test_all_models_registered(self):
        from src.models.model_registry import MODEL_REGISTRY
        assert len(MODEL_REGISTRY) >= 15

    def test_starred_models(self):
        from src.models.model_registry import get_starred_models
        starred = get_starred_models()
        assert "shapley" in starred
        assert "ensemble" in starred


# ── Data Quality Tests ──

class TestDataQuality:
    def test_channel_count(self):
        assert len(CHANNELS) == 13

    def test_channel_ids_valid(self):
        for ch in CHANNELS:
            assert isinstance(ch, str)
            assert "_" in ch or ch.isalpha()

    def test_attribution_result_schema(self, sample_journeys):
        from src.models.rule_based import LinearAttribution
        model = LinearAttribution()
        model.fit(sample_journeys)
        result = model.attribute(sample_journeys)

        summary = result.to_summary_df()
        assert "model_name" in summary.columns
        assert "channel_id" in summary.columns
        assert "attributed_conversions" in summary.columns
        assert "attribution_pct" in summary.columns
        assert "rank" in summary.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
