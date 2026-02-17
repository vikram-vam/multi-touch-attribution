"""
Page 5 — Identity Resolution (EP-5)
"""

from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from app.data_store import get_identity_graph, get_journeys
from app.components.components import section_header, chart_container, metric_card, insight_callout
from app.theme import COLORS
from src.utils.formatters import fmt_number, fmt_pct


def layout():
    identity = get_identity_graph()
    journeys = get_journeys()
    total_j = len(journeys)

    if len(identity) > 0:
        total_fragments = len(identity)
        unique_persistent = identity["persistent_id"].nunique() if "persistent_id" in identity.columns else 0
        avg_fragments = total_fragments / max(unique_persistent, 1)
        match_rate = identity["match_confidence"].mean() if "match_confidence" in identity.columns else 0
    else:
        total_fragments = 0
        unique_persistent = 0
        avg_fragments = 0
        match_rate = 0

    return html.Div([
        section_header(
            "Identity Resolution & Graph",
            "How fragmented user identities are resolved into unified customer profiles"
        ),

        dbc.Row([
            dbc.Col(metric_card("Fragment IDs", fmt_number(total_fragments)), md=3),
            dbc.Col(metric_card("Resolved Profiles", fmt_number(unique_persistent)), md=3),
            dbc.Col(metric_card("Avg. Fragments/User", f"{avg_fragments:.1f}"), md=3),
            dbc.Col(metric_card("Avg. Match Confidence", fmt_pct(match_rate)), md=3),
        ], className="mb-4"),

        insight_callout(
            f"Identity resolution merged {fmt_number(total_fragments)} device/cookie fragments "
            f"into {fmt_number(unique_persistent)} unified profiles ({avg_fragments:.1f} fragments per user). "
            f"Without resolution, attribution would over-count unique users and under-credit "
            f"channels that appear later in the journey.",
            icon="🔗"
        ),

        dbc.Row([
            dbc.Col(chart_container(
                "Match Tier Distribution", "identity-tier-dist", "400px"
            ), md=6),
            dbc.Col(chart_container(
                "Match Confidence Distribution", "identity-confidence", "400px"
            ), md=6),
        ]),

        dbc.Row([
            dbc.Col(chart_container(
                "Fragments per User", "identity-fragments", "350px"
            ), md=12),
        ]),

    ], className="page-container")


@callback(
    Output("identity-tier-dist", "figure"),
    Input("identity-tier-dist", "id"),
)
def update_tier_dist(_):
    identity = get_identity_graph()
    if len(identity) == 0 or "match_tier" not in identity.columns:
        return go.Figure()

    counts = identity["match_tier"].value_counts()
    tier_colors = {"deterministic": COLORS["positive"], "probabilistic": COLORS["warning"],
                   "fuzzy": COLORS["negative"]}

    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.5,
        marker=dict(colors=[tier_colors.get(t, COLORS["info"]) for t in counts.index]),
    ))
    fig.update_layout(title="Identity Match Tiers", height=400)
    return fig


@callback(
    Output("identity-confidence", "figure"),
    Input("identity-confidence", "id"),
)
def update_confidence(_):
    identity = get_identity_graph()
    if len(identity) == 0 or "match_confidence" not in identity.columns:
        return go.Figure()

    fig = go.Figure(go.Histogram(
        x=identity["match_confidence"],
        nbinsx=30,
        marker_color=COLORS["highlight"],
    ))
    fig.update_layout(
        title="Match Confidence Distribution",
        xaxis_title="Confidence Score",
        yaxis_title="Count",
        height=400,
    )
    return fig


@callback(
    Output("identity-fragments", "figure"),
    Input("identity-fragments", "id"),
)
def update_fragments(_):
    identity = get_identity_graph()
    if len(identity) == 0 or "persistent_id" not in identity.columns:
        return go.Figure()

    frags = identity.groupby("persistent_id").size().value_counts().sort_index()

    fig = go.Figure(go.Bar(
        x=frags.index.astype(str),
        y=frags.values,
        marker_color=COLORS["highlight_alt"],
    ))
    fig.update_layout(
        title="Distribution of Fragments per Resolved User",
        xaxis_title="Number of Fragments",
        yaxis_title="User Count",
        height=350,
    )
    return fig
