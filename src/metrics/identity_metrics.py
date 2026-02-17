"""
Identity resolution metrics.
Computes all metrics for the Identity Resolution page (EP-5).
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class IdentityMetrics:
    """Metrics for the Identity Resolution page."""
    total_fragments: int
    total_persistent_ids: int
    avg_fragments_per_id: float
    match_tier_distribution: Dict[str, int]
    avg_confidence: float
    median_confidence: float
    resolution_rate: float

    @classmethod
    def from_data(
        cls,
        identity_graph: pd.DataFrame,
        journeys: pd.DataFrame = None,
    ) -> "IdentityMetrics":
        total_fragments = len(identity_graph)
        total_pids = identity_graph["persistent_id"].nunique()
        avg_frags = total_fragments / max(total_pids, 1)

        tier_dist = {}
        if "match_tier" in identity_graph.columns:
            tier_dist = identity_graph["match_tier"].value_counts().to_dict()

        avg_conf = identity_graph["confidence"].mean() if "confidence" in identity_graph.columns else 0
        med_conf = identity_graph["confidence"].median() if "confidence" in identity_graph.columns else 0

        # Resolution rate = pids with multiple fragments / total pids
        frags_per_pid = identity_graph.groupby("persistent_id").size()
        multi_frag_pids = (frags_per_pid > 1).sum()
        resolution_rate = multi_frag_pids / max(total_pids, 1)

        return cls(
            total_fragments=total_fragments,
            total_persistent_ids=total_pids,
            avg_fragments_per_id=avg_frags,
            match_tier_distribution=tier_dist,
            avg_confidence=avg_conf,
            median_confidence=med_conf,
            resolution_rate=resolution_rate,
        )
