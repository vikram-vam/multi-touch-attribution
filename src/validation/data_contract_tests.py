"""
Data contract validation.
Validates schema contracts between pipeline layers and ensures
all UI data sources have the expected columns and types.
"""

from typing import Dict, List
from dataclasses import dataclass

import pandas as pd


@dataclass
class ContractViolation:
    """A violation of a data contract."""
    dataset: str
    issue: str
    severity: str  # "error", "warning"
    details: str


# Expected schemas for each data layer
SCHEMAS = {
    "touchpoints": {
        "required": ["touchpoint_id", "journey_id", "persistent_id", "channel_id",
                      "timestamp", "session_id"],
        "types": {"touchpoint_id": "str", "journey_id": "str", "channel_id": "str"},
    },
    "journeys": {
        "required": ["journey_id", "persistent_id", "is_converting",
                      "touchpoint_count", "channel_path", "first_touch_channel",
                      "last_touch_channel", "has_agent_touch"],
        "types": {"is_converting": "bool", "touchpoint_count": "int"},
    },
    "attribution_results": {
        "required": ["model_name", "model_tier", "channel_id",
                      "attributed_conversions", "attribution_pct", "rank"],
        "types": {"attributed_conversions": "float", "attribution_pct": "float"},
    },
    "channel_spend": {
        "required": ["channel_id", "month", "spend_dollars"],
        "types": {"spend_dollars": "float"},
    },
    "identity_graph": {
        "required": ["fragment_id", "persistent_id", "match_tier", "confidence"],
        "types": {"confidence": "float"},
    },
}


def validate_schema(
    df: pd.DataFrame,
    dataset_name: str,
) -> List[ContractViolation]:
    """Validate a DataFrame against its expected schema."""
    violations = []
    schema = SCHEMAS.get(dataset_name)

    if schema is None:
        violations.append(ContractViolation(
            dataset_name, "Unknown dataset", "warning",
            f"No schema defined for '{dataset_name}'",
        ))
        return violations

    # Check required columns
    for col in schema["required"]:
        if col not in df.columns:
            violations.append(ContractViolation(
                dataset_name, "Missing column", "error",
                f"Required column '{col}' not found. Available: {list(df.columns)}",
            ))

    # Check for empty
    if len(df) == 0:
        violations.append(ContractViolation(
            dataset_name, "Empty dataset", "error",
            f"Dataset '{dataset_name}' has 0 rows",
        ))

    # Check for nulls in required columns
    for col in schema["required"]:
        if col in df.columns and df[col].isnull().any():
            n_null = df[col].isnull().sum()
            violations.append(ContractViolation(
                dataset_name, "Null values", "warning",
                f"Column '{col}' has {n_null} null values ({n_null/len(df):.1%})",
            ))

    return violations


def validate_all_datasets(
    datasets: Dict[str, pd.DataFrame],
) -> List[ContractViolation]:
    """Validate all datasets against their schemas."""
    all_violations = []
    for name, df in datasets.items():
        all_violations.extend(validate_schema(df, name))
    return all_violations


def validate_referential_integrity(
    touchpoints: pd.DataFrame,
    journeys: pd.DataFrame,
) -> List[ContractViolation]:
    """Check that touchpoint journey_ids match journey dataset."""
    violations = []

    if "journey_id" in touchpoints.columns and "journey_id" in journeys.columns:
        tp_ids = set(touchpoints["journey_id"].unique())
        j_ids = set(journeys["journey_id"].unique())
        orphan_tps = tp_ids - j_ids
        if orphan_tps:
            violations.append(ContractViolation(
                "touchpoints→journeys", "Orphan records", "warning",
                f"{len(orphan_tps)} touchpoint journey_ids not found in journeys",
            ))

    return violations


def contract_summary(violations: List[ContractViolation]) -> pd.DataFrame:
    """Summarize contract violations."""
    if not violations:
        return pd.DataFrame({"status": ["All contracts passed"]})
    records = []
    for v in violations:
        records.append({
            "dataset": v.dataset,
            "issue": v.issue,
            "severity": v.severity,
            "details": v.details,
        })
    return pd.DataFrame(records)
