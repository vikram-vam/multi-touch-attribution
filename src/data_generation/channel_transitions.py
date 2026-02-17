"""
Channel transition matrix for Markov-based journey simulation.
Encodes Erie's specific channel interaction patterns.
"""

import numpy as np
from typing import Dict, List, Tuple

CHANNELS = [
    "independent_agent",       # 0
    "paid_search_brand",       # 1
    "paid_search_nonbrand",    # 2
    "organic_search",          # 3
    "display_programmatic",    # 4
    "paid_social",             # 5
    "tv_radio",                # 6
    "direct_mail",             # 7
    "email_marketing",         # 8
    "call_center",             # 9
    "aggregator_comparator",   # 10
    "direct_organic",          # 11
    "video_ott_ctv",           # 12
]

# Special states
START = "START"
CONVERSION = "CONVERSION"
NULL = "NULL"

ALL_STATES = [START] + CHANNELS + [CONVERSION, NULL]

# First-touch channel distribution (entry points for journeys)
FIRST_TOUCH_DISTRIBUTION = {
    "paid_search_brand": 0.12, "paid_search_nonbrand": 0.10, "organic_search": 0.15,
    "display_programmatic": 0.10, "paid_social": 0.08, "tv_radio": 0.12,
    "direct_mail": 0.08, "email_marketing": 0.02, "call_center": 0.02,
    "aggregator_comparator": 0.04, "direct_organic": 0.10, "video_ott_ctv": 0.05,
    "independent_agent": 0.02,
}

# Transition design — P(next | current) for key channel patterns
TRANSITION_DESIGN: Dict[str, Dict[str, float]] = {
    "display_programmatic": {
        "organic_search": 0.24, "paid_search_brand": 0.22, "paid_social": 0.14,
        "independent_agent": 0.04, "direct_organic": 0.11,
        "NULL": 0.20, "CONVERSION": 0.00, "paid_search_nonbrand": 0.05,
    },
    "paid_search_brand": {
        "independent_agent": 0.15, "direct_organic": 0.16, "email_marketing": 0.12,
        "CONVERSION": 0.01, "NULL": 0.34, "call_center": 0.08, "organic_search": 0.06,
        "paid_search_nonbrand": 0.04, "aggregator_comparator": 0.04,
    },
    "paid_search_nonbrand": {
        "independent_agent": 0.10, "paid_search_brand": 0.20, "direct_organic": 0.14,
        "organic_search": 0.14, "CONVERSION": 0.01, "NULL": 0.29,
        "call_center": 0.05, "email_marketing": 0.04, "aggregator_comparator": 0.03,
    },
    "independent_agent": {
        "CONVERSION": 0.03, "email_marketing": 0.10, "NULL": 0.72,
        "call_center": 0.06, "paid_search_brand": 0.05, "direct_organic": 0.04,
    },
    "direct_mail": {
        "independent_agent": 0.18, "paid_search_brand": 0.20, "call_center": 0.10,
        "direct_organic": 0.16, "NULL": 0.28, "CONVERSION": 0.01,
        "organic_search": 0.07,
    },
    "paid_social": {
        "organic_search": 0.26, "paid_search_brand": 0.20, "display_programmatic": 0.14,
        "direct_organic": 0.10, "independent_agent": 0.03,
        "NULL": 0.20, "CONVERSION": 0.01, "paid_search_nonbrand": 0.06,
    },
    "organic_search": {
        "independent_agent": 0.10, "paid_search_brand": 0.18, "direct_organic": 0.20,
        "email_marketing": 0.06, "CONVERSION": 0.01, "NULL": 0.30,
        "call_center": 0.05, "paid_search_nonbrand": 0.05, "aggregator_comparator": 0.05,
    },
    "email_marketing": {
        "independent_agent": 0.12, "paid_search_brand": 0.14, "direct_organic": 0.20,
        "CONVERSION": 0.01, "NULL": 0.37, "call_center": 0.06, "organic_search": 0.10,
    },
    "call_center": {
        "independent_agent": 0.18, "CONVERSION": 0.02, "email_marketing": 0.08,
        "NULL": 0.51, "direct_organic": 0.10, "paid_search_brand": 0.06,
        "organic_search": 0.05,
    },
    "tv_radio": {
        "paid_search_brand": 0.28, "organic_search": 0.24, "direct_organic": 0.16,
        "independent_agent": 0.03, "NULL": 0.22, "CONVERSION": 0.01,
        "paid_social": 0.02, "video_ott_ctv": 0.02, "paid_search_nonbrand": 0.02,
    },
    "video_ott_ctv": {
        "paid_search_brand": 0.22, "organic_search": 0.22, "paid_social": 0.14,
        "direct_organic": 0.14, "independent_agent": 0.03,
        "NULL": 0.18, "CONVERSION": 0.01, "display_programmatic": 0.06,
    },
    "aggregator_comparator": {
        "paid_search_brand": 0.30, "independent_agent": 0.08, "organic_search": 0.14,
        "direct_organic": 0.14, "CONVERSION": 0.01, "NULL": 0.27,
        "call_center": 0.03, "paid_search_nonbrand": 0.03,
    },
    "direct_organic": {
        "independent_agent": 0.10, "paid_search_brand": 0.12, "email_marketing": 0.06,
        "CONVERSION": 0.01, "NULL": 0.54, "call_center": 0.06,
        "organic_search": 0.06, "paid_search_nonbrand": 0.05,
    },
}

# 2nd-order contextual boosts: (prev, current) → boosted next probabilities
SECOND_ORDER_BOOSTS: Dict[Tuple[str, str], Dict[str, float]] = {
    ("display_programmatic", "paid_search_brand"): {"independent_agent": 0.22},
    ("direct_mail", "call_center"): {"independent_agent": 0.25},
    ("paid_social", "organic_search"): {"independent_agent": 0.15},
    ("tv_radio", "paid_search_brand"): {"independent_agent": 0.18},
    ("video_ott_ctv", "paid_search_brand"): {"independent_agent": 0.16},
    ("aggregator_comparator", "paid_search_brand"): {"independent_agent": 0.18},
}


def build_transition_matrix() -> np.ndarray:
    """
    Build the full transition matrix from the design specification.
    Rows = current state, Columns = next state.
    States indexed by ALL_STATES list.

    Returns:
        np.ndarray of shape (n_states, n_states) with row-stochastic probabilities.
    """
    n = len(ALL_STATES)
    state_idx = {s: i for i, s in enumerate(ALL_STATES)}
    matrix = np.zeros((n, n))

    # START → first touch channels
    for ch, prob in FIRST_TOUCH_DISTRIBUTION.items():
        matrix[state_idx[START], state_idx[ch]] = prob

    # Channel → Channel/CONVERSION/NULL transitions
    for channel, transitions in TRANSITION_DESIGN.items():
        if channel not in state_idx:
            continue
        row_idx = state_idx[channel]
        for target, prob in transitions.items():
            if target in state_idx:
                matrix[row_idx, state_idx[target]] = prob

    # Absorbing states: CONVERSION and NULL stay put
    matrix[state_idx[CONVERSION], state_idx[CONVERSION]] = 1.0
    matrix[state_idx[NULL], state_idx[NULL]] = 1.0

    # Normalize rows (handle residual probability)
    for i in range(n):
        row_sum = matrix[i].sum()
        if row_sum > 0 and abs(row_sum - 1.0) > 1e-10:
            matrix[i] /= row_sum

    return matrix


def get_transition_prob(
    current: str,
    rng: np.random.Generator,
    prev: str = None,
    digital_propensity: float = 0.5,
) -> str:
    """
    Sample next state given current (and optionally previous) channel.

    Applies 2nd-order boosts and digital propensity adjustments.

    Args:
        current: Current channel state.
        prev: Previous channel state (for 2nd-order patterns).
        rng: NumPy random generator.
        digital_propensity: User's digital propensity (0-1).

    Returns:
        Next state name.
    """
    if current in (CONVERSION, NULL):
        return current

    if current not in TRANSITION_DESIGN:
        return NULL

    transitions = dict(TRANSITION_DESIGN[current])

    # Apply 2nd-order boosts
    if prev and (prev, current) in SECOND_ORDER_BOOSTS:
        for target, boosted_prob in SECOND_ORDER_BOOSTS[(prev, current)].items():
            if target in transitions:
                transitions[target] = boosted_prob

    # Adjust for digital propensity
    # Higher digital propensity → more digital channels, lower → more agent/offline
    if digital_propensity > 0.6:
        # Boost digital channels slightly
        for ch in ["paid_search_brand", "paid_search_nonbrand", "organic_search",
                    "display_programmatic", "paid_social", "direct_organic"]:
            if ch in transitions:
                transitions[ch] *= 1.0 + (digital_propensity - 0.5) * 0.3
    elif digital_propensity < 0.4:
        # Boost offline channels
        for ch in ["independent_agent", "direct_mail", "call_center"]:
            if ch in transitions:
                transitions[ch] *= 1.0 + (0.5 - digital_propensity) * 0.4

    # Normalize
    targets = list(transitions.keys())
    probs = np.array([transitions[t] for t in targets])
    probs = probs / probs.sum()

    choice_idx = rng.choice(len(targets), p=probs)
    return targets[choice_idx]
