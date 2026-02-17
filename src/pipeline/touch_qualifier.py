"""
Touch qualification module.
Filters and weights touchpoints by MRC viewability standards.
"""

import pandas as pd


# MRC (Media Rating Council) minimum viewability thresholds
MRC_THRESHOLDS = {
    "display": {"min_viewability": 0.50, "min_dwell_seconds": 1.0},
    "video": {"min_viewability": 0.50, "min_dwell_seconds": 2.0},
    "click": {"min_viewability": 1.0, "min_dwell_seconds": 0.0},
    "agent": {"min_viewability": 1.0, "min_dwell_seconds": 0.0},
    "call": {"min_viewability": 1.0, "min_dwell_seconds": 0.0},
    "mail": {"min_viewability": 1.0, "min_dwell_seconds": 0.0},
    "email": {"min_viewability": 0.0, "min_dwell_seconds": 0.0},
}

# Touch type → MRC category mapping
TOUCH_TYPE_CATEGORY = {
    "impression": "display",
    "click": "click",
    "agent": "agent",
    "call": "call",
    "mail": "mail",
    "email": "email",
}

# Channel → impression/click category
IMPRESSION_CHANNELS = {
    "display_programmatic", "paid_social", "tv_radio", "video_ott_ctv",
}


def qualify_touchpoints(touchpoints_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply MRC viewability standards to qualify touchpoints.

    Touchpoints below minimum thresholds are downweighted (not removed)
    to retain the journey structure while reflecting quality.

    Args:
        touchpoints_df: DataFrame with 'touch_type', 'viewability_pct',
                        'dwell_time_seconds' columns.

    Returns:
        DataFrame with 'is_qualified' and 'touch_weight' updated.
    """
    df = touchpoints_df.copy()

    # Map touch types to MRC categories
    df["mrc_category"] = df["touch_type"].map(TOUCH_TYPE_CATEGORY).fillna("click")

    # Apply qualification rules
    qualified = []
    weights = []

    for _, row in df.iterrows():
        category = row["mrc_category"]
        thresholds = MRC_THRESHOLDS.get(category, MRC_THRESHOLDS["click"])

        viewability_ok = row.get("viewability_pct", 1.0) >= thresholds["min_viewability"]
        dwell_ok = row.get("dwell_time_seconds", 0.0) >= thresholds["min_dwell_seconds"]

        is_qual = viewability_ok and dwell_ok
        qualified.append(is_qual)

        # Compute weight: qualified = full weight, else reduced
        if is_qual:
            # Weight based on channel type: clicks/agent = 1.0, impressions = 0.6
            if row["touch_type"] in ("click", "agent", "call"):
                w = 1.0
            elif row["touch_type"] in ("impression",):
                w = 0.6
            else:
                w = 0.5
        else:
            w = 0.1  # Unqualified but still in journey

        weights.append(w)

    df["is_qualified"] = qualified
    df["touch_weight"] = weights

    # Drop temp column
    df.drop(columns=["mrc_category"], inplace=True)

    return df
