"""
User profile generator for synthetic data.
Samples demographics matching Erie Insurance's customer base.
"""

import uuid
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# Erie's 12 states + DC with population-weighted distribution
ERIE_STATE_DISTRIBUTION = {
    "PA": 0.30, "OH": 0.12, "NY": 0.11, "NC": 0.08, "VA": 0.07,
    "MD": 0.06, "IN": 0.05, "WI": 0.05, "TN": 0.04, "WV": 0.04,
    "IL": 0.04, "KY": 0.03, "DC": 0.01,
}

AGE_DISTRIBUTION = {
    "18-24": 0.08, "25-34": 0.20, "35-44": 0.25,
    "45-54": 0.22, "55-64": 0.15, "65+": 0.10,
}

VEHICLE_TYPE_DISTRIBUTION = {
    "sedan": 0.35, "suv": 0.30, "truck": 0.15, "luxury": 0.10, "sports": 0.10,
}

LIFE_EVENT_DISTRIBUTION = {
    "none": 0.60, "new_car": 0.15, "relocation": 0.08,
    "teen_driver": 0.07, "marriage": 0.05, "home_purchase": 0.05,
}

# Digital propensity varies by age — younger skews digital, older skews agent-first
DIGITAL_PROPENSITY_BY_AGE = {
    "18-24": {"mean": 0.75, "std": 0.12},
    "25-34": {"mean": 0.65, "std": 0.15},
    "35-44": {"mean": 0.55, "std": 0.18},
    "45-54": {"mean": 0.45, "std": 0.18},
    "55-64": {"mean": 0.35, "std": 0.15},
    "65+":   {"mean": 0.25, "std": 0.12},
}


@dataclass
class UserProfile:
    """A single simulated user for journey generation."""
    persistent_id: str
    age_band: str
    state: str
    vehicle_type: str
    life_event_trigger: Optional[str]
    digital_propensity: float
    price_sensitivity: float


def _sample_from_distribution(rng: np.random.Generator, dist: dict) -> str:
    """Sample a single key from a probability distribution dict."""
    keys = list(dist.keys())
    probabilities = np.array(list(dist.values()))
    probabilities = probabilities / probabilities.sum()  # Normalize
    return keys[rng.choice(len(keys), p=probabilities)]


def generate_user_profiles(
    n: int,
    rng: np.random.Generator,
) -> List[UserProfile]:
    """
    Generate n user profiles from Erie-calibrated distributions.

    Users with life events have longer journeys and higher conversion rates.
    Digital propensity determines entry channel preferences.
    """
    profiles = []
    for _ in range(n):
        age_band = _sample_from_distribution(rng, AGE_DISTRIBUTION)
        state = _sample_from_distribution(rng, ERIE_STATE_DISTRIBUTION)
        vehicle_type = _sample_from_distribution(rng, VEHICLE_TYPE_DISTRIBUTION)
        life_event = _sample_from_distribution(rng, LIFE_EVENT_DISTRIBUTION)

        # Digital propensity from age-based normal distribution (clipped)
        dp_params = DIGITAL_PROPENSITY_BY_AGE[age_band]
        digital_propensity = float(np.clip(
            rng.normal(dp_params["mean"], dp_params["std"]),
            0.05, 0.95,
        ))

        # Price sensitivity: uniform(0.2, 0.8) with slight age correlation
        base_sensitivity = rng.uniform(0.2, 0.8)
        # Younger → more price sensitive
        age_factor = {"18-24": 0.15, "25-34": 0.10, "35-44": 0.0,
                      "45-54": -0.05, "55-64": -0.10, "65+": -0.15}
        price_sensitivity = float(np.clip(
            base_sensitivity + age_factor.get(age_band, 0), 0.05, 0.95,
        ))

        profiles.append(UserProfile(
            persistent_id=str(uuid.uuid4()),
            age_band=age_band,
            state=state,
            vehicle_type=vehicle_type,
            life_event_trigger=life_event if life_event != "none" else None,
            digital_propensity=digital_propensity,
            price_sensitivity=price_sensitivity,
        ))

    return profiles
