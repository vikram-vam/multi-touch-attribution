"""
Tests for data contract validation.
"""

import sys, os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validation.data_contract_tests import (
    validate_schema, validate_all_datasets,
    validate_referential_integrity, SCHEMAS,
)


class TestJourneyContract:
    def test_valid_journeys_pass(self):
        df = pd.DataFrame({
            "journey_id": ["j1", "j2"],
            "persistent_id": ["p1", "p2"],
            "is_converting": [True, False],
            "touchpoint_count": [3, 2],
            "channel_path": [["a", "b"], ["c"]],
            "first_touch_channel": ["a", "c"],
            "last_touch_channel": ["b", "c"],
            "has_agent_touch": [True, False],
        })
        violations = validate_schema(df, "journeys")
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0

    def test_missing_column_fails(self):
        df = pd.DataFrame({"journey_id": ["j1"]})
        violations = validate_schema(df, "journeys")
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) > 0

    def test_empty_dataset_fails(self):
        df = pd.DataFrame(columns=SCHEMAS["journeys"]["required"])
        violations = validate_schema(df, "journeys")
        errors = [v for v in violations if v.issue == "Empty dataset"]
        assert len(errors) == 1


class TestAttributionContract:
    def test_valid_attribution_passes(self):
        df = pd.DataFrame({
            "model_name": ["shapley", "shapley"],
            "model_tier": ["game_theoretic", "game_theoretic"],
            "channel_id": ["a", "b"],
            "attributed_conversions": [100.0, 50.0],
            "attribution_pct": [0.67, 0.33],
            "rank": [1, 2],
        })
        violations = validate_schema(df, "attribution_results")
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0


class TestReferentialIntegrity:
    def test_matching_ids_pass(self):
        tp = pd.DataFrame({"journey_id": ["j1", "j2"]})
        j = pd.DataFrame({"journey_id": ["j1", "j2", "j3"]})
        violations = validate_referential_integrity(tp, j)
        assert len(violations) == 0

    def test_orphan_detection(self):
        tp = pd.DataFrame({"journey_id": ["j1", "j_orphan"]})
        j = pd.DataFrame({"journey_id": ["j1"]})
        violations = validate_referential_integrity(tp, j)
        assert len(violations) > 0


class TestAllDatasets:
    def test_validates_multiple(self):
        datasets = {
            "journeys": pd.DataFrame({
                "journey_id": ["j1"], "persistent_id": ["p1"],
                "is_converting": [True], "touchpoint_count": [2],
                "channel_path": [["a"]], "first_touch_channel": ["a"],
                "last_touch_channel": ["a"], "has_agent_touch": [False],
            }),
        }
        violations = validate_all_datasets(datasets)
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0
