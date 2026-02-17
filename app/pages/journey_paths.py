"""
Page 3 — Journey Paths (EP-3)
Sankey diagram and path analysis.
"""

from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from app.data_store import get_journeys, get_channel_display_name
from app.components.components import (
    section_header, chart_container, metric_card, insight_callout
)
from app.theme import COLORS, get_channel_color


def layout():
    journeys = get_journeys()
    converting = journeys[journeys["is_converting"]] if "is_converting" in journeys.columns else journeys.head(0)
    total_c = len(converting)

    with_agent = journeys[journeys["has_agent_touch"]] if "has_agent_touch" in journeys.columns else journeys.head(0)
    without_agent = journeys[~journeys["has_agent_touch"]] if "has_agent_touch" in journeys.columns else journeys
    agent_rate = with_agent["is_converting"].mean() if len(with_agent) > 0 else 0
    no_agent_rate = without_agent["is_converting"].mean() if len(without_agent) > 0 else 0.001
    multiplier = agent_rate / max(no_agent_rate, 0.001)

    avg_touches = converting["touchpoint_count"].mean() if len(converting) > 0 and "touchpoint_count" in converting.columns else 0
    single_pct = len(converting[converting["touchpoint_count"] == 1]) / total_c if total_c > 0 and "touchpoint_count" in converting.columns else 0

    return html.Div([
        section_header(
            "Journey Path Analysis",
            "Understanding the customer journey from first awareness to bind"
        ),

        dbc.Row([
            dbc.Col(metric_card("Avg. Path Length", f"{avg_touches:.1f} touches"), md=3),
            dbc.Col(metric_card("Single-Touch %", f"{single_pct*100:.1f}%"), md=3),
            dbc.Col(metric_card("Agent Multiplier", f"{multiplier:.1f}×",
                                delta="vs. digital-only", delta_direction="positive"), md=3),
            dbc.Col(metric_card("Multi-Channel Journeys",
                                f'{len(converting[converting["distinct_channel_count"] > 1]) if "distinct_channel_count" in converting.columns and total_c > 0 else 0:,}'), md=3),
        ], className="mb-4"),

        # Sankey
        dbc.Row([
            dbc.Col(chart_container(
                "Channel Flow: Top Converting Paths (Sankey)", "journey-sankey", "550px"
            ), md=12),
        ]),

        insight_callout(
            f"Multi-touch journeys with {avg_touches:.1f} average touchpoints reveal that "
            f"only {single_pct*100:.0f}% of conversions are single-channel. "
            f"Last-touch attribution misses {(1-single_pct)*100:.0f}% of the story.",
            icon="🔗"
        ),

        # Top paths table
        dbc.Row([
            dbc.Col(chart_container(
                "Top 15 Converting Paths", "journey-top-paths", "400px"
            ), md=7),
            dbc.Col(chart_container(
                "Path Length Distribution", "journey-length-dist", "400px"
            ), md=5),
        ]),

    ], className="page-container")


@callback(
    Output("journey-sankey", "figure"),
    Input("journey-sankey", "id"),
)
def update_sankey(_):
    journeys = get_journeys()
    converting = journeys[journeys["is_converting"]] if "is_converting" in journeys.columns else journeys.head(0)

    if len(converting) == 0 or "channel_path" not in converting.columns:
        return go.Figure()

    # Build transition counts for first 4 positions
    from collections import Counter
    transitions = Counter()

    for _, row in converting.head(5000).iterrows():
        path = row["channel_path"]
        if isinstance(path, str):
            path = path.split("|")
        for i in range(min(len(path) - 1, 3)):
            src = f"Step {i+1}: {path[i]}"
            tgt = f"Step {i+2}: {path[i+1]}"
            transitions[(src, tgt)] += 1

    # Filter to top transitions
    top = transitions.most_common(50)
    if not top:
        return go.Figure()

    all_nodes = sorted(set([t[0][0] for t in top] + [t[0][1] for t in top]))
    node_idx = {n: i for i, n in enumerate(all_nodes)}

    # Node colors based on channel
    node_colors = []
    for n in all_nodes:
        ch = n.split(": ", 1)[1] if ": " in n else n
        node_colors.append(get_channel_color(ch))

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color=COLORS["border"], width=0.5),
            label=[n.replace("_", " ").title() for n in all_nodes],
            color=node_colors,
        ),
        link=dict(
            source=[node_idx[t[0][0]] for t in top],
            target=[node_idx[t[0][1]] for t in top],
            value=[t[1] for t in top],
            color=[f"rgba(0,212,170,0.15)"] * len(top),
        ),
    ))

    fig.update_layout(title="Channel Transition Flow", height=550)
    return fig


@callback(
    Output("journey-top-paths", "figure"),
    Input("journey-top-paths", "id"),
)
def update_top_paths(_):
    journeys = get_journeys()
    converting = journeys[journeys["is_converting"]] if "is_converting" in journeys.columns else journeys.head(0)

    if len(converting) == 0 or "channel_path_str" not in converting.columns:
        return go.Figure()

    top = converting["channel_path_str"].value_counts().head(15)

    fig = go.Figure(go.Bar(
        y=[p.replace("|", " → ").replace("_", " ")[:60] for p in top.index],
        x=top.values,
        orientation="h",
        marker_color=COLORS["highlight"],
    ))

    fig.update_layout(
        title="Top Converting Paths",
        xaxis_title="Count",
        height=400,
        margin=dict(l=250),
    )
    return fig


@callback(
    Output("journey-length-dist", "figure"),
    Input("journey-length-dist", "id"),
)
def update_length_dist(_):
    journeys = get_journeys()
    converting = journeys[journeys["is_converting"]] if "is_converting" in journeys.columns else journeys.head(0)

    if len(converting) == 0 or "touchpoint_count" not in converting.columns:
        return go.Figure()

    fig = go.Figure(go.Histogram(
        x=converting["touchpoint_count"],
        nbinsx=20,
        marker_color=COLORS["highlight"],
    ))

    fig.update_layout(
        title="Path Length Distribution",
        xaxis_title="Number of Touchpoints",
        yaxis_title="Journey Count",
        height=400,
    )
    return fig
