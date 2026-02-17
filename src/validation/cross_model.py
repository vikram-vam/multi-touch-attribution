"""
Cross-model comparison and rank correlation analysis.
Computes Spearman ρ, Kendall τ, and rank agreement across model pairs.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


def compute_spearman_matrix(
    attribution_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute pairwise Spearman ρ between all model pairs.

    Returns:
        Square DataFrame with Spearman ρ values.
    """
    pivot = attribution_results.pivot_table(
        index="channel_id", columns="model_name",
        values="attribution_pct", aggfunc="first",
    ).fillna(0)

    models = pivot.columns.tolist()
    n = len(models)
    matrix = pd.DataFrame(1.0, index=models, columns=models)

    for i in range(n):
        for j in range(i + 1, n):
            rho, p_val = spearmanr(pivot[models[i]], pivot[models[j]])
            matrix.loc[models[i], models[j]] = rho
            matrix.loc[models[j], models[i]] = rho

    return matrix


def compute_kendall_matrix(
    attribution_results: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pairwise Kendall τ between all model pairs."""
    pivot = attribution_results.pivot_table(
        index="channel_id", columns="model_name",
        values="attribution_pct", aggfunc="first",
    ).fillna(0)

    models = pivot.columns.tolist()
    n = len(models)
    matrix = pd.DataFrame(1.0, index=models, columns=models)

    for i in range(n):
        for j in range(i + 1, n):
            tau, p_val = kendalltau(pivot[models[i]], pivot[models[j]])
            matrix.loc[models[i], models[j]] = tau
            matrix.loc[models[j], models[i]] = tau

    return matrix


def compute_rank_agreement(
    attribution_results: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Compute top-k rank agreement between model pairs.

    Measures: what % of model A's top-k channels appear in model B's top-k.
    """
    pivot = attribution_results.pivot_table(
        index="channel_id", columns="model_name",
        values="attribution_pct", aggfunc="first",
    ).fillna(0)

    models = pivot.columns.tolist()
    agreement = pd.DataFrame(1.0, index=models, columns=models)

    for mi in models:
        top_i = set(pivot[mi].nlargest(top_k).index)
        for mj in models:
            if mi != mj:
                top_j = set(pivot[mj].nlargest(top_k).index)
                overlap = len(top_i & top_j) / top_k
                agreement.loc[mi, mj] = overlap

    return agreement


def find_model_clusters(
    spearman_matrix: pd.DataFrame,
    threshold: float = 0.85,
) -> List[List[str]]:
    """
    Find clusters of highly correlated models.

    Models with Spearman ρ > threshold are grouped together.
    """
    models = spearman_matrix.index.tolist()
    visited = set()
    clusters = []

    for model in models:
        if model in visited:
            continue
        cluster = [model]
        visited.add(model)

        for other in models:
            if other not in visited:
                rho = spearman_matrix.loc[model, other]
                if rho >= threshold:
                    cluster.append(other)
                    visited.add(other)

        clusters.append(cluster)

    return clusters


def cross_model_summary(
    attribution_results: pd.DataFrame,
) -> Dict:
    """Generate comprehensive cross-model comparison summary."""
    spearman = compute_spearman_matrix(attribution_results)
    kendall = compute_kendall_matrix(attribution_results)
    rank_agree = compute_rank_agreement(attribution_results)
    clusters = find_model_clusters(spearman)

    # Find most/least agreeing pairs
    models = spearman.index.tolist()
    pairs = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            pairs.append((models[i], models[j], spearman.loc[models[i], models[j]]))

    pairs.sort(key=lambda x: -x[2])

    return {
        "spearman_matrix": spearman,
        "kendall_matrix": kendall,
        "rank_agreement_top5": rank_agree,
        "model_clusters": clusters,
        "most_similar": pairs[:3] if pairs else [],
        "most_divergent": pairs[-3:] if pairs else [],
        "avg_pairwise_rho": np.mean([p[2] for p in pairs]) if pairs else 0,
        "min_pairwise_rho": min(p[2] for p in pairs) if pairs else 0,
    }
