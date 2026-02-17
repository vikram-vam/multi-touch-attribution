"""
Triangulation framework — MMM / incrementality comparison.
Positions MCA as part of a broader measurement strategy.
Anchored in Gordon et al. (2023) estimation error findings.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TriangulationComparison:
    """Comparison across measurement methodologies."""
    methodology: str
    description: str
    strengths: List[str]
    limitations: List[str]
    data_requirements: List[str]
    timeline_weeks: int
    coverage: str  # "channel", "campaign", "user"
    erie_readiness: str  # "ready", "needs_data", "future"


MEASUREMENT_METHODS = [
    TriangulationComparison(
        methodology="Multi-Touch Attribution (MTA)",
        description="User-level journey analysis with Shapley/Markov models",
        strengths=[
            "User-level granularity",
            "Real-time campaign optimization",
            "Channel interaction effects",
            "Works with existing data",
        ],
        limitations=[
            "Observational bias (Gordon et al. 2023: 488–948% error range)",
            "Limited offline/TV measurement",
            "Cookie deprecation challenges",
            "Selection bias in exposure",
        ],
        data_requirements=[
            "User-level touchpoint data",
            "Conversion tracking",
            "Identity resolution",
        ],
        timeline_weeks=8,
        coverage="user",
        erie_readiness="ready",
    ),
    TriangulationComparison(
        methodology="Marketing Mix Modeling (MMM)",
        description="Aggregate time-series regression (e.g., Google Meridian)",
        strengths=[
            "Privacy-safe (aggregate data only)",
            "Captures offline channels (TV, radio, OOH)",
            "Long-term and carryover effects",
            "No cookie dependency",
        ],
        limitations=[
            "Requires 2–3 years of data",
            "Weekly/monthly granularity only",
            "Cannot optimize individual campaigns",
            "Model specification sensitivity",
        ],
        data_requirements=[
            "2+ years weekly/monthly spend by channel",
            "Conversion volumes",
            "External factors (seasonality, macro)",
        ],
        timeline_weeks=12,
        coverage="channel",
        erie_readiness="needs_data",
    ),
    TriangulationComparison(
        methodology="Incrementality Testing",
        description="Randomized controlled experiments (geo-lift, ghost ads)",
        strengths=[
            "Gold standard causal inference",
            "Unbiased treatment effects",
            "Directly measures incremental impact",
            "Validates other methodologies",
        ],
        limitations=[
            "Expensive (requires holdout)",
            "Tests one channel at a time",
            "Requires statistical power (large volumes)",
            "Results are point-in-time",
        ],
        data_requirements=[
            "Ability to run holdout experiments",
            "Geo-level conversion data",
            "4–8 weeks per test cycle",
        ],
        timeline_weeks=6,
        coverage="campaign",
        erie_readiness="future",
    ),
]


@dataclass
class TriangulationResult:
    """Result of triangulating across methods."""
    channel_id: str
    mta_attribution_pct: float
    mmm_estimated_pct: Optional[float] = None
    incrementality_pct: Optional[float] = None
    confidence: str = "low"
    convergence: bool = False
    recommendation: str = ""


def compute_triangulation_roadmap() -> pd.DataFrame:
    """
    Generate recommended measurement roadmap for Erie.

    Returns DataFrame with methodology comparison and implementation timeline.
    """
    records = []
    for method in MEASUREMENT_METHODS:
        records.append({
            "methodology": method.methodology,
            "description": method.description,
            "timeline_weeks": method.timeline_weeks,
            "coverage": method.coverage,
            "erie_readiness": method.erie_readiness,
            "key_strength": method.strengths[0] if method.strengths else "",
            "key_limitation": method.limitations[0] if method.limitations else "",
        })
    return pd.DataFrame(records)


def gordon_error_context() -> Dict:
    """
    Context from Gordon et al. (2023) on MTA estimation errors.

    Key finding: Observational MTA produces 488–948% errors in ad effect
    estimation compared to experimental ground truth.

    Our demo addresses this by:
    1. Showing multiple MTA models (convergence builds confidence)
    2. Explicitly stating estimation uncertainty
    3. Positioning triangulation as the solution
    """
    return {
        "paper": "Gordon et al. (2023)",
        "title": "Close Enough? A Large-Scale Exploration of Non-Experimental Approaches to Ad Effectiveness",
        "finding": "Observational methods produce 488–948% errors in ad effect estimation",
        "implication": "No single MTA model should be trusted alone",
        "our_approach": [
            "Run 17+ models across 6 tiers to find convergence",
            "Compare Shapley, Markov, and statistical approaches",
            "Use ensemble weighting to triangulate",
            "Recommend incrementality tests for high-stakes decisions",
            "Position MTA as optimization tool, not ground truth",
        ],
        "demo_talking_point": (
            "Gordon et al. (2023) showed that observational MTA can have "
            "488–948% errors vs. experimental ground truth. That's exactly why "
            "we run 17 models — not because any one is 'right,' but because "
            "convergence across fundamentally different methods is our best signal. "
            "The next step is incrementality testing to validate."
        ),
    }


TRIANGULATION_ROADMAP = """
Phase 1 (Now): Multi-Touch Attribution
  → 17 models, ensemble weighting, Shapley-derived optimization
  → Timeline: 8 weeks to production

Phase 2 (Q3): Marketing Mix Modeling
  → Google Meridian or similar
  → Requires 2+ years of aggregate data
  → Timeline: 12 weeks after data onboarding

Phase 3 (Q4): Incrementality Testing
  → Geo-lift experiments on top 3 channels
  → Validates MTA and MMM findings
  → Timeline: 6 weeks per test cycle

Convergence Dashboard:
  → Shows MTA, MMM, and incrementality side-by-side
  → Confidence score based on method agreement
  → Decision framework for budget allocation
"""
