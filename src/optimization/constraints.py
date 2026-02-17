"""
Business constraints for budget optimization.
Defines floor/ceiling spend constraints, channel lock rules, and regulatory limits.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

import pandas as pd


@dataclass
class ChannelConstraint:
    """Spend constraint for a single channel."""
    channel_id: str
    min_spend_pct: float = 0.50      # Floor: 50% of current
    max_spend_pct: float = 2.00      # Ceiling: 200% of current
    locked: bool = False             # If True, spend cannot change
    min_absolute: float = 0.0        # Absolute minimum spend
    max_absolute: float = float("inf")
    notes: str = ""


# Erie-specific business constraints
ERIE_CONSTRAINTS: Dict[str, ChannelConstraint] = {
    "independent_agent": ChannelConstraint(
        channel_id="independent_agent",
        min_spend_pct=0.80,  # Agent investment is strategic — floor at 80%
        max_spend_pct=1.50,
        notes="Agent network is Erie's strategic differentiator. Cannot cut deeply.",
    ),
    "paid_search_brand": ChannelConstraint(
        channel_id="paid_search_brand",
        min_spend_pct=0.50,
        max_spend_pct=1.50,
        notes="Brand search protects against competitor bidding on Erie brand terms.",
    ),
    "paid_search_nonbrand": ChannelConstraint(
        channel_id="paid_search_nonbrand",
        min_spend_pct=0.40,
        max_spend_pct=2.00,
        notes="Non-brand search is highly competitive but has clear ROI signals.",
    ),
    "organic_search": ChannelConstraint(
        channel_id="organic_search",
        min_spend_pct=0.70,
        max_spend_pct=1.50,
        notes="SEO investment is long-term; drastic cuts damage rankings for years.",
    ),
    "display_programmatic": ChannelConstraint(
        channel_id="display_programmatic",
        min_spend_pct=0.30,
        max_spend_pct=3.00,
        notes="Display is highly scalable — can increase significantly if data supports it.",
    ),
    "paid_social": ChannelConstraint(
        channel_id="paid_social",
        min_spend_pct=0.30,
        max_spend_pct=3.00,
        notes="Social is audience-targetable and scalable.",
    ),
    "tv_radio": ChannelConstraint(
        channel_id="tv_radio",
        min_spend_pct=0.50,
        max_spend_pct=1.20,
        notes="TV/radio contracts are typically annual. Limited mid-year flexibility.",
    ),
    "direct_mail": ChannelConstraint(
        channel_id="direct_mail",
        min_spend_pct=0.40,
        max_spend_pct=1.80,
        notes="Direct mail has long lead times but can be scaled seasonally.",
    ),
    "email_marketing": ChannelConstraint(
        channel_id="email_marketing",
        min_spend_pct=0.50,
        max_spend_pct=2.50,
        notes="Low marginal cost. Can increase volume through better segmentation.",
    ),
    "call_center": ChannelConstraint(
        channel_id="call_center",
        min_spend_pct=0.70,
        max_spend_pct=1.50,
        locked=False,
        notes="Call center staffing has 3-month ramp-up time.",
    ),
    "aggregator_comparator": ChannelConstraint(
        channel_id="aggregator_comparator",
        min_spend_pct=0.30,
        max_spend_pct=2.50,
        notes="Aggregator partnerships can be adjusted quarterly.",
    ),
    "direct_organic": ChannelConstraint(
        channel_id="direct_organic",
        locked=True,
        notes="Direct/organic is not a paid channel — cannot be optimized via spend.",
    ),
    "video_ott_ctv": ChannelConstraint(
        channel_id="video_ott_ctv",
        min_spend_pct=0.30,
        max_spend_pct=3.00,
        notes="Growing channel. Video/CTV is under-invested relative to attribution value.",
    ),
}


def get_constraints(channel_id: str) -> ChannelConstraint:
    """Get constraints for a channel, with defaults for unknown channels."""
    return ERIE_CONSTRAINTS.get(
        channel_id,
        ChannelConstraint(channel_id=channel_id),
    )


def apply_constraints(
    proposed_spend: Dict[str, float],
    current_spend: Dict[str, float],
) -> Dict[str, float]:
    """Apply business constraints to a proposed spend allocation."""
    constrained = {}

    for ch, proposed in proposed_spend.items():
        constraint = get_constraints(ch)
        current = current_spend.get(ch, proposed)

        if constraint.locked:
            constrained[ch] = current
            continue

        # Apply percentage bounds
        floor = max(current * constraint.min_spend_pct, constraint.min_absolute)
        ceiling = min(current * constraint.max_spend_pct, constraint.max_absolute)
        constrained[ch] = np.clip(proposed, floor, ceiling)

    return constrained


def validate_total_budget(
    proposed_spend: Dict[str, float],
    budget_limit: float,
    tolerance: float = 0.02,
) -> bool:
    """Check if proposed allocation respects total budget constraint."""
    total = sum(proposed_spend.values())
    return abs(total - budget_limit) / max(budget_limit, 1) <= tolerance


# NumPy import for apply_constraints
import numpy as np
