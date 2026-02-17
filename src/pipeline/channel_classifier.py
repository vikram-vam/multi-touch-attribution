"""
MECE channel classification.
Ensures every touchpoint maps to exactly one of 13 channels.
"""

import pandas as pd

from src.data_generation.channel_transitions import CHANNELS


CHANNEL_CLASSIFICATION_RULES = {
    "independent_agent": ["agent", "in_person", "broker"],
    "paid_search_brand": ["google_brand", "bing_brand", "brand_search", "erie_search"],
    "paid_search_nonbrand": ["google_nonbrand", "bing_nonbrand", "generic_search",
                              "auto_insurance_search"],
    "organic_search": ["google_organic", "bing_organic", "seo"],
    "display_programmatic": ["google_display", "dv360", "retargeting", "banner",
                              "programmatic"],
    "paid_social": ["facebook", "instagram", "meta", "social_ad"],
    "tv_radio": ["tv_spot", "radio_spot", "broadcast", "national_tv", "local_tv"],
    "direct_mail": ["mailer", "postcard", "direct_mail"],
    "email_marketing": ["email", "newsletter", "nurture"],
    "call_center": ["phone", "inbound_call", "outbound_call", "call"],
    "aggregator_comparator": ["the_zebra", "policygenius", "nerdwallet", "comparator",
                               "comparison", "aggregator"],
    "direct_organic": ["direct", "bookmark", "typed_url", "app_open"],
    "video_ott_ctv": ["youtube", "hulu", "roku", "ctv", "ott", "video_ad"],
}

# Build reverse lookup: sub_channel → parent channel
_REVERSE_LOOKUP = {}
for channel_id, sub_channels in CHANNEL_CLASSIFICATION_RULES.items():
    for sub in sub_channels:
        _REVERSE_LOOKUP[sub.lower()] = channel_id


def classify_channel(sub_channel: str) -> str:
    """
    Classify a sub-channel or touchpoint source to its parent channel.

    Returns:
        Parent channel_id (one of 13 MECE channels).
    """
    sub = sub_channel.lower().strip()

    # Direct match
    if sub in _REVERSE_LOOKUP:
        return _REVERSE_LOOKUP[sub]

    # Partial match
    for key, channel_id in _REVERSE_LOOKUP.items():
        if key in sub or sub in key:
            return channel_id

    # Already a parent channel?
    if sub in CHANNELS:
        return sub

    # Default to direct_organic for unmatched
    return "direct_organic"


def ensure_mece_classification(touchpoints_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure every touchpoint has exactly one of 13 channel IDs.
    Re-classifies any non-standard channel_ids.

    Args:
        touchpoints_df: DataFrame with 'channel_id' column.

    Returns:
        DataFrame with validated 'channel_id' values.
    """
    df = touchpoints_df.copy()

    # Check for any channel IDs not in standard set
    non_standard = df[~df["channel_id"].isin(CHANNELS)]
    if len(non_standard) > 0:
        df.loc[non_standard.index, "channel_id"] = (
            non_standard["channel_id"].apply(classify_channel)
        )

    return df
