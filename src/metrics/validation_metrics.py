"""
Validation dashboard metrics.
Computes all metrics for the Validation & Model Quality page (EP-7).
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class ValidationMetrics:
    """Metrics for the Validation Dashboard page."""
    n_models: int
    avg_pairwise_spearman: float
    axiom_pass_rate: float
    sanity_check_pass_rate: float
    model_tier_summary: Dict[str, int]
    bootstrap_ci: Dict[str, tuple]
    recommendation: str

    @classmethod
    def from_data(
        cls,
        attribution_results: pd.DataFrame,
        spearman_matrix: pd.DataFrame = None,
        axiom_results: List = None,
        sanity_results: List = None,
    ) -> "ValidationMetrics":
        models = attribution_results["model_name"].unique()
        n_models = len(models)

        # Average pairwise Spearman
        avg_rho = 0.0
        if spearman_matrix is not None and len(spearman_matrix) > 1:
            mask = np.triu(np.ones(spearman_matrix.shape, dtype=bool), k=1)
            vals = spearman_matrix.values[mask]
            avg_rho = float(np.mean(vals)) if len(vals) > 0 else 0

        # Axiom pass rate
        axiom_pass = 1.0
        if axiom_results:
            passed = sum(1 for r in axiom_results if r.passed)
            axiom_pass = passed / len(axiom_results)

        # Sanity check pass rate
        sanity_pass = 1.0
        if sanity_results:
            passed = sum(1 for r in sanity_results if r.passed)
            sanity_pass = passed / len(sanity_results)

        # Model tier summary
        tier_summary = {}
        if "model_tier" in attribution_results.columns:
            tier_summary = attribution_results.groupby("model_tier")["model_name"].nunique().to_dict()

        # Bootstrap confidence intervals on top channel
        bootstrap_ci = {}
        pivot = attribution_results.pivot_table(
            index="model_name", columns="channel_id",
            values="attribution_pct", aggfunc="first",
        ).fillna(0)

        for ch in pivot.columns:
            vals = pivot[ch].values
            if len(vals) > 2:
                lo = float(np.percentile(vals, 5))
                hi = float(np.percentile(vals, 95))
                bootstrap_ci[ch] = (round(lo, 4), round(hi, 4))

        # Overall recommendation
        if avg_rho > 0.80 and axiom_pass >= 0.75 and sanity_pass >= 0.80:
            rec = "HIGH confidence — models converge and pass quality checks"
        elif avg_rho > 0.60:
            rec = "MODERATE confidence — some model divergence, review results carefully"
        else:
            rec = "LOW confidence — significant model disagreement, investigate data quality"

        return cls(
            n_models=n_models,
            avg_pairwise_spearman=avg_rho,
            axiom_pass_rate=axiom_pass,
            sanity_check_pass_rate=sanity_pass,
            model_tier_summary=tier_summary,
            bootstrap_ci=bootstrap_ci,
            recommendation=rec,
        )
