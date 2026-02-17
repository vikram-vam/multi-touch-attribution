"""
Conversion probability model using logistic regression.
Determines P(conversion) at each step based on accumulated channel features.
"""

import math
from typing import Dict, List, Set


# Logistic model coefficients — calibrated for Erie's ~3% conversion rate
# Note: should_convert() is called at EVERY step, so per-step P must be very low
# to achieve ~3% overall journey conversion rate across multi-step journeys.
CONVERSION_FEATURES: Dict[str, float] = {
    "intercept":               -9.0,     # Base per-step conversion ~0.01%
    "agent_present":            1.0,     # Agent presence: moderate lift
    "agent_is_last_touch":      0.4,     # Agent as last touch: mild additional lift
    "num_distinct_channels":    0.15,    # Channel diversity: mild lift per distinct channel
    "has_search_touch":         0.25,    # Search = active shopping intent
    "has_display_or_social":    0.1,     # Upper funnel awareness: small contribution
    "has_direct_mail":          0.15,    # Direct mail = targeted prospect
    "has_email_click":          0.3,     # Email click = engagement signal
    "has_call_center":          0.35,    # Phone call = high intent
    "journey_length_penalty":  -0.08,    # Stronger decay with journey length
    "recency_factor":           0.1,     # Recency: small per-unit lift
    "life_event_trigger":       0.4,     # Life event: moderate motivation boost
}


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def compute_conversion_probability(
    channel_path: List[str],
    current_channel: str,
    has_life_event: bool = False,
    recency_score: float = 0.5,
) -> float:
    """
    Compute P(conversion) at the current step of a journey.

    Uses a logistic model with channel features. The accumulation of
    diverse channels and agent presence dramatically increase conversion.

    Args:
        channel_path: List of channel_ids touched so far (including current).
        current_channel: The current channel in the journey.
        has_life_event: Whether the user has a life event trigger.
        recency_score: 0-1 score for how recent the activity is.

    Returns:
        Probability of conversion (0.0 - 1.0).

    Calibration targets:
        - Overall journey-to-bind rate: ~2.5-3.5%
        - Quote-start-to-bind rate: ~30-40%
        - Journeys with agent touch convert at 4-5x rate of pure digital
        - Agent as last touch + prior digital: highest converting segment
    """
    channel_set: Set[str] = set(channel_path)

    logit = CONVERSION_FEATURES["intercept"]

    # Agent presence
    if "independent_agent" in channel_set:
        logit += CONVERSION_FEATURES["agent_present"]

    # Agent as last/current touch
    if current_channel == "independent_agent":
        logit += CONVERSION_FEATURES["agent_is_last_touch"]

    # Channel diversity
    num_distinct = len(channel_set)
    logit += CONVERSION_FEATURES["num_distinct_channels"] * min(num_distinct, 5)

    # Search presence
    search_channels = {"paid_search_brand", "paid_search_nonbrand", "organic_search"}
    if channel_set & search_channels:
        logit += CONVERSION_FEATURES["has_search_touch"]

    # Display/Social presence
    display_social = {"display_programmatic", "paid_social", "video_ott_ctv"}
    if channel_set & display_social:
        logit += CONVERSION_FEATURES["has_display_or_social"]

    # Direct mail
    if "direct_mail" in channel_set:
        logit += CONVERSION_FEATURES["has_direct_mail"]

    # Email engagement
    if "email_marketing" in channel_set:
        logit += CONVERSION_FEATURES["has_email_click"]

    # Call center
    if "call_center" in channel_set:
        logit += CONVERSION_FEATURES["has_call_center"]

    # Journey length penalty
    logit += CONVERSION_FEATURES["journey_length_penalty"] * len(channel_path)

    # Recency
    logit += CONVERSION_FEATURES["recency_factor"] * recency_score

    # Life event boost
    if has_life_event:
        logit += CONVERSION_FEATURES["life_event_trigger"]

    return sigmoid(logit)


def should_convert(
    channel_path: List[str],
    current_channel: str,
    rng,
    has_life_event: bool = False,
    recency_score: float = 0.5,
) -> bool:
    """
    Stochastic conversion decision at this journey step.

    Returns True if the user converts based on accumulated features.
    """
    prob = compute_conversion_probability(
        channel_path, current_channel, has_life_event, recency_score,
    )
    return float(rng.random()) < prob
