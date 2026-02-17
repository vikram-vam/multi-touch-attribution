"""
Application theme — colors, fonts, and Plotly template.
"""

import plotly.graph_objects as go
import plotly.io as pio
import dash_bootstrap_components as dbc


# ── External Stylesheets ──
EXTERNAL_STYLESHEETS = [
    dbc.themes.DARKLY,
    "https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap",
]

# ── Color Palette ──
COLORS = {
    "bg_primary": "#0F1419",
    "bg_secondary": "#1A2332",
    "bg_tertiary": "#243044",
    "bg_card": "#1E2D3D",
    "text_primary": "#E8ECF1",
    "text_secondary": "#8899AA",
    "text_accent": "#FFFFFF",
    "highlight": "#00D4AA",
    "highlight_alt": "#00B4D8",
    "positive": "#2ECC71",
    "negative": "#E74C3C",
    "warning": "#F39C12",
    "info": "#3498DB",
    "border": "#2A3A4A",

    # Channel colors — distinct and accessible
    "channel_colors": {
        "independent_agent": "#2ECC71",
        "paid_search_brand": "#3498DB",
        "paid_search_nonbrand": "#2980B9",
        "organic_search": "#1ABC9C",
        "display_programmatic": "#E74C3C",
        "paid_social": "#9B59B6",
        "tv_radio": "#F39C12",
        "direct_mail": "#E67E22",
        "email_marketing": "#F1C40F",
        "call_center": "#16A085",
        "aggregator_comparator": "#95A5A6",
        "direct_organic": "#34495E",
        "video_ott_ctv": "#D35400",
    },

    # Tier colors
    "tier_colors": {
        "rule_based": "#95A5A6",
        "game_theoretic": "#2ECC71",
        "probabilistic": "#3498DB",
        "statistical": "#9B59B6",
        "deep_learning": "#E74C3C",
        "meta_model": "#F39C12",
    },
}

# ── Plotly Template ──
PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=COLORS["bg_secondary"],
        plot_bgcolor=COLORS["bg_secondary"],
        font=dict(
            family="Inter, sans-serif",
            color=COLORS["text_primary"],
            size=12,
        ),
        title=dict(
            font=dict(
                family="DM Sans, sans-serif",
                size=18,
                color=COLORS["text_accent"],
            ),
            x=0,
            xanchor="left",
        ),
        xaxis=dict(
            gridcolor=COLORS["border"],
            linecolor=COLORS["border"],
            zerolinecolor=COLORS["border"],
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        yaxis=dict(
            gridcolor=COLORS["border"],
            linecolor=COLORS["border"],
            zerolinecolor=COLORS["border"],
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        colorway=[
            COLORS["highlight"], COLORS["info"], COLORS["positive"],
            COLORS["warning"], "#9B59B6", "#E74C3C", "#F1C40F",
            "#1ABC9C", "#D35400", "#34495E", "#E67E22", "#95A5A6",
            "#2980B9",
        ],
        margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text_secondary"]),
        ),
        hoverlabel=dict(
            bgcolor=COLORS["bg_tertiary"],
            font_size=12,
            font_family="Inter",
        ),
    ),
)

# Register as default template
pio.templates["erie_dark"] = PLOTLY_TEMPLATE
pio.templates.default = "erie_dark"


def get_channel_color(channel_id: str) -> str:
    """Get the assigned color for a channel."""
    return COLORS["channel_colors"].get(channel_id, COLORS["highlight"])


def get_channel_colors_list(channels: list) -> list:
    """Get ordered color list for a list of channels."""
    return [get_channel_color(ch) for ch in channels]
