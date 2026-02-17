"""
Identity resolution simulator.
Generates simulated identity fragmentation and match data for the Identity Resolution page.
"""

import uuid
from typing import Dict, List

import numpy as np
import pandas as pd

# Match rate targets from Section 5.1
MATCH_RATE_TARGETS = {
    "tier_1_deterministic": 0.65,    # 65% matched via email/phone hash
    "tier_2_household": 0.12,        # +12% via address matching
    "tier_3_probabilistic": 0.10,    # +10% via IP+UA (lower confidence)
    "unmatched": 0.13,               # 13% remain fragmented
}

IDENTIFIER_TYPES = ["email_hash", "phone_hash", "ga4_client_id", "gclid", "address_hash"]
DEVICE_TYPES = ["desktop", "mobile", "tablet"]

CONFIDENCE_BY_TIER = {
    1: {"mean": 0.95, "std": 0.03},
    2: {"mean": 0.78, "std": 0.08},
    3: {"mean": 0.55, "std": 0.12},
}

# Per user: ~2.1 GA4 client IDs (multi-device), ~1.3 ad click IDs
IDENTIFIERS_PER_USER = {
    "ga4_client_id": {"mean": 2.1, "std": 0.8, "min": 1, "max": 5},
    "email_hash": {"mean": 1.0, "std": 0.2, "min": 0, "max": 2},
    "phone_hash": {"mean": 0.9, "std": 0.3, "min": 0, "max": 2},
    "gclid": {"mean": 1.3, "std": 0.7, "min": 0, "max": 4},
    "address_hash": {"mean": 0.8, "std": 0.2, "min": 0, "max": 1},
}


def generate_identity_graph(
    persistent_ids: List[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate identity fragmentation data for all users.

    Simulates multi-device, multi-identifier scenarios and assigns
    match tiers with confidence scores.

    Args:
        persistent_ids: List of ground-truth user identifiers.
        rng: NumPy random generator.

    Returns:
        DataFrame matching identity_graph.parquet schema (Section 3.5).
    """
    rows = []

    for pid in persistent_ids:
        # Determine match tier for this user
        tier_roll = rng.random()
        if tier_roll < MATCH_RATE_TARGETS["tier_1_deterministic"]:
            primary_tier = 1
        elif tier_roll < (MATCH_RATE_TARGETS["tier_1_deterministic"] +
                          MATCH_RATE_TARGETS["tier_2_household"]):
            primary_tier = 2
        elif tier_roll < (1.0 - MATCH_RATE_TARGETS["unmatched"]):
            primary_tier = 3
        else:
            primary_tier = 0  # Unmatched — still generate some IDs

        # Generate identifiers per type
        for id_type, params in IDENTIFIERS_PER_USER.items():
            count = int(np.clip(
                rng.normal(params["mean"], params["std"]),
                params["min"], params["max"]
            ))

            for _ in range(count):
                # Select match tier (most IDs at primary tier, some at lower)
                if primary_tier == 0:
                    tier = 3 if rng.random() < 0.3 else 0
                else:
                    tier = primary_tier if rng.random() < 0.7 else min(primary_tier + 1, 3)

                if tier == 0:
                    continue

                # Confidence score from tier distribution
                conf_params = CONFIDENCE_BY_TIER[tier]
                confidence = float(np.clip(
                    rng.normal(conf_params["mean"], conf_params["std"]),
                    0.1, 1.0
                ))

                # Device type
                device = rng.choice(DEVICE_TYPES)

                rows.append({
                    "persistent_id": pid,
                    "identifier_type": id_type,
                    "identifier_value": str(uuid.uuid4())[:16],
                    "match_tier": tier,
                    "match_confidence": round(confidence, 3),
                    "device_type": device,
                })

    return pd.DataFrame(rows)


def generate_sample_identity_graphs(
    n_users: int = 50,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Generate detailed identity graphs for n sample users.
    Used for the Identity Resolution visualization page.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    sample_ids = [str(uuid.uuid4()) for _ in range(n_users)]
    return generate_identity_graph(sample_ids, rng)
