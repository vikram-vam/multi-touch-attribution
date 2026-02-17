"""
Channel-level spend data generation for ROI calculations.
Erie estimated annual marketing spend ~$100M.
"""

import numpy as np
import pandas as pd
from typing import Dict

from src.data_generation.timestamp_engine import MONTHLY_SEASONALITY

# Annual spend allocation by channel (fraction of $100M total)
ANNUAL_SPEND_ALLOCATION: Dict[str, float] = {
    "paid_search_brand": 0.15,
    "paid_search_nonbrand": 0.08,
    "organic_search": 0.05,
    "display_programmatic": 0.10,
    "paid_social": 0.07,
    "tv_radio": 0.25,
    "direct_mail": 0.10,
    "email_marketing": 0.03,
    "call_center": 0.05,
    "aggregator_comparator": 0.02,
    "direct_organic": 0.00,  # No spend (organic)
    "video_ott_ctv": 0.06,
    "independent_agent": 0.04,
}

# CPM / CPC benchmarks for impression / click estimation
CHANNEL_COST_METRICS: Dict[str, Dict[str, float]] = {
    "paid_search_brand": {"cpc": 3.50, "ctr": 0.08},
    "paid_search_nonbrand": {"cpc": 8.50, "ctr": 0.035},
    "organic_search": {"cpc": 0.0, "ctr": 0.05},
    "display_programmatic": {"cpm": 4.50, "ctr": 0.003},
    "paid_social": {"cpm": 7.00, "ctr": 0.012},
    "tv_radio": {"cpm": 15.00, "ctr": 0.0},
    "direct_mail": {"cost_per_piece": 0.85, "ctr": 0.0},
    "email_marketing": {"cost_per_send": 0.02, "ctr": 0.035},
    "call_center": {"cost_per_call": 15.00, "ctr": 0.0},
    "aggregator_comparator": {"cpc": 12.00, "ctr": 0.04},
    "direct_organic": {"cpc": 0.0, "ctr": 0.0},
    "video_ott_ctv": {"cpm": 12.00, "ctr": 0.005},
    "independent_agent": {"cost_per_interaction": 25.00, "ctr": 0.0},
}


def generate_channel_spend(
    total_annual_spend: float = 100_000_000,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Generate monthly channel-level spend data.

    Spend varies with seasonality + ±10% random noise per month.
    Also estimates impressions and clicks based on cost benchmarks.

    Args:
        total_annual_spend: Total annual marketing budget.
        rng: NumPy random generator.

    Returns:
        DataFrame matching channel_spend.parquet schema (Section 3.6).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    rows = []
    for channel_id, allocation_pct in ANNUAL_SPEND_ALLOCATION.items():
        annual_channel_spend = total_annual_spend * allocation_pct

        for month in range(1, 13):
            # Monthly spend = annual/12 × seasonality × noise
            seasonality = MONTHLY_SEASONALITY.get(month, 1.0)
            noise = 1.0 + rng.uniform(-0.10, 0.10)
            monthly_spend = (annual_channel_spend / 12) * seasonality * noise

            # Estimate impressions and clicks
            metrics = CHANNEL_COST_METRICS.get(channel_id, {})
            impressions = 0
            clicks = 0

            if "cpm" in metrics and metrics["cpm"] > 0:
                impressions = int(monthly_spend / metrics["cpm"] * 1000)
                clicks = int(impressions * metrics.get("ctr", 0))
            elif "cpc" in metrics and metrics["cpc"] > 0:
                clicks = int(monthly_spend / metrics["cpc"])
                impressions = int(clicks / max(metrics.get("ctr", 0.05), 0.001))
            elif "cost_per_piece" in metrics and metrics["cost_per_piece"] > 0:
                impressions = int(monthly_spend / metrics["cost_per_piece"])
            elif "cost_per_send" in metrics and metrics["cost_per_send"] > 0:
                impressions = int(monthly_spend / metrics["cost_per_send"])
                clicks = int(impressions * metrics.get("ctr", 0.035))
            elif "cost_per_call" in metrics and metrics["cost_per_call"] > 0:
                clicks = int(monthly_spend / metrics["cost_per_call"])
            elif "cost_per_interaction" in metrics and metrics["cost_per_interaction"] > 0:
                clicks = int(monthly_spend / metrics["cost_per_interaction"])

            rows.append({
                "channel_id": channel_id,
                "month": month,
                "year": 2025,
                "spend_dollars": round(monthly_spend, 2),
                "impressions": impressions,
                "clicks": clicks,
            })

    return pd.DataFrame(rows)
