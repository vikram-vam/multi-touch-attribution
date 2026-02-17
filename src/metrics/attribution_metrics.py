"""
Attribution comparison metrics.
Computes all metrics for the Attribution Comparison page (EP-2).
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class AttributionComparisonMetrics:
    """Metrics for the Attribution Comparison page."""
    model_results: pd.DataFrame
    model_count: int
    spearman_matrix: pd.DataFrame
    largest_disagreement_channel: str
    largest_disagreement_range_pp: float

    @classmethod
    def from_data(
        cls,
        attribution_results: pd.DataFrame,
    ) -> "AttributionComparisonMetrics":
        model_names = attribution_results["model_name"].unique()
        model_count = len(model_names)

        pivot = attribution_results.pivot_table(
            index="channel_id", columns="model_name",
            values="attribution_pct", aggfunc="first",
        ).fillna(0)

        spearman = pd.DataFrame(
            np.ones((model_count, model_count)),
            index=model_names, columns=model_names,
        )
        for i, m1 in enumerate(model_names):
            for j, m2 in enumerate(model_names):
                if m1 in pivot.columns and m2 in pivot.columns:
                    rho, _ = spearmanr(pivot[m1], pivot[m2])
                    spearman.loc[m1, m2] = rho

        if len(pivot.columns) > 1:
            ranges = pivot.max(axis=1) - pivot.min(axis=1)
            largest_ch = ranges.idxmax()
            largest_range = ranges.max()
        else:
            largest_ch = "N/A"
            largest_range = 0.0

        return cls(
            model_results=attribution_results,
            model_count=model_count,
            spearman_matrix=spearman,
            largest_disagreement_channel=largest_ch,
            largest_disagreement_range_pp=largest_range,
        )
