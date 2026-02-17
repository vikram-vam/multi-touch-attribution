"""
Page 1 — Executive Summary (EP-1)
The "money slide" that opens the Erie demo.
"""

from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from app.data_store import (
    get_attribution_results, get_journeys, get_channel_spend,
    get_channel_display_name, get_demo_narrative,
)
from app.components.components import metric_card, insight_callout, chart_container, section_header
from app.theme import COLORS, get_channel_color
from src.utils.formatters import fmt_number, fmt_pct, fmt_currency


def layout():
    journeys = get_journeys()
    results = get_attribution_results()
    spend = get_channel_spend()

    total_j = len(journeys)
    converting = journeys[journeys["is_converting"]] if "is_converting" in journeys.columns else journeys.head(0)
    total_c = len(converting)
    conv_rate = total_c / total_j if total_j > 0 else 0
    total_spend = spend["spend_dollars"].sum() if "spend_dollars" in spend.columns and len(spend) > 0 else 0
    avg_cost = total_spend / max(total_c, 1)
    avg_premium = 1200
    total_premium = total_c * avg_premium

    # Agent stats
    with_agent = journeys[journeys["has_agent_touch"]] if "has_agent_touch" in journeys.columns else journeys.head(0)
    agent_rate = with_agent["is_converting"].mean() if len(with_agent) > 0 else 0
    no_agent = journeys[~journeys["has_agent_touch"]] if "has_agent_touch" in journeys.columns else journeys
    no_agent_rate = no_agent["is_converting"].mean() if len(no_agent) > 0 else 0.001
    multiplier = agent_rate / max(no_agent_rate, 0.001)

    # Last-Touch vs Shapley for agent
    lt = results[results["model_name"] == "last_touch"] if len(results) > 0 else results
    sh = results[results["model_name"] == "shapley"] if len(results) > 0 else results
    lt_agent_pct = 0
    sh_agent_pct = 0
    if len(lt) > 0:
        lt_total = lt["attributed_conversions"].sum()
        lt_row = lt[lt["channel_id"] == "independent_agent"]
        lt_agent_pct = lt_row["attributed_conversions"].values[0] / lt_total if len(lt_row) > 0 and lt_total > 0 else 0
    if len(sh) > 0:
        sh_total = sh["attributed_conversions"].sum()
        sh_row = sh[sh["channel_id"] == "independent_agent"]
        sh_agent_pct = sh_row["attributed_conversions"].values[0] / sh_total if len(sh_row) > 0 and sh_total > 0 else 0
    shift = sh_agent_pct - lt_agent_pct

    return html.Div([
        section_header(
            "Executive Summary",
            "Multi-Channel Attribution Intelligence for Erie Insurance"
        ),

        # ── KPI Row ──
        dbc.Row([
            dbc.Col(metric_card("Total Journeys", fmt_number(total_j)), md=2),
            dbc.Col(metric_card("Conversions (Binds)", fmt_number(total_c),
                                subtitle=f"{fmt_pct(conv_rate)} conversion rate"), md=2),
            dbc.Col(metric_card("Total Media Spend", fmt_currency(total_spend)), md=2),
            dbc.Col(metric_card("Avg. Cost per Bind", fmt_currency(avg_cost)), md=2),
            dbc.Col(metric_card("Total Premium Value", fmt_currency(total_premium)), md=2),
            dbc.Col(metric_card("Agent Multiplier", f"{multiplier:.1f}×",
                                subtitle="vs. non-agent journeys",
                                delta=f"+{shift*100:.1f}pp credit shift",
                                delta_direction="positive"), md=2),
        ], className="mb-4"),

        # ── Key Insight ──
        insight_callout(
            f"Agent-involved journeys convert at {multiplier:.1f}× the rate of digital-only journeys, "
            f"yet last-touch gives agents only {fmt_pct(lt_agent_pct)} credit. "
            f"Shapley reveals their true contribution at {fmt_pct(sh_agent_pct)} — "
            f"a {abs(shift)*100:.1f}pp misallocation worth ${int(abs(shift) * total_c * avg_premium):,} in premium value.",
            icon="🔍"
        ),

        # ── Charts Row ──
        dbc.Row([
            dbc.Col(
                chart_container("Attribution by Model: Independent Agent", "exec-agent-chart"),
                md=6
            ),
            dbc.Col(
                chart_container("Last-Touch vs. Shapley: All Channels", "exec-comparison-chart"),
                md=6
            ),
        ]),

        # ── Recommendation ──
        dbc.Row([
            dbc.Col(
                insight_callout(
                    f"Recommendation: Reallocate {fmt_pct(0.18)} of brand search spend to agent support programs. "
                    f"Projected impact: +{int(total_c * 0.18):,} additional binds generating "
                    f"${int(total_c * 0.18 * avg_premium):,} in annual premium.",
                    icon="💰"
                ),
                md=12
            ),
        ]),

        # ── Journey Stats ──
        dbc.Row([
            dbc.Col(metric_card(
                "Avg. Touchpoints (Converting)",
                f"{converting['touchpoint_count'].mean():.1f}" if len(converting) > 0 and 'touchpoint_count' in converting.columns else "—",
            ), md=3),
            dbc.Col(metric_card(
                "Median Journey Duration",
                f"{converting['journey_duration_days'].median():.0f} days" if len(converting) > 0 and 'journey_duration_days' in converting.columns else "—",
            ), md=3),
            dbc.Col(metric_card(
                "Journeys with Agent",
                fmt_pct(len(with_agent) / total_j) if total_j > 0 else "—",
            ), md=3),
            dbc.Col(metric_card(
                "Multi-Touch Journeys",
                fmt_pct(len(converting[converting["touchpoint_count"] > 1]) / len(converting)) if len(converting) > 0 and 'touchpoint_count' in converting.columns else "—",
            ), md=3),
        ], className="mt-3"),

    ], className="page-container")


# ── Callbacks ──
@callback(
    Output("exec-agent-chart", "figure"),
    Input("exec-agent-chart", "id"),
)
def update_agent_chart(_):
    results = get_attribution_results()
    if len(results) == 0:
        return go.Figure()

    agent_data = results[results["channel_id"] == "independent_agent"].copy()
    if len(agent_data) == 0:
        return go.Figure()

    agent_data = agent_data.sort_values("attribution_pct", ascending=True)

    fig = go.Figure(go.Bar(
        y=agent_data["model_name"],
        x=agent_data["attribution_pct"],
        orientation="h",
        marker_color=[COLORS["highlight"] if "shapley" in m or "ensemble" in m
                     else COLORS["text_secondary"]
                     for m in agent_data["model_name"]],
        hovertemplate="<b>%{y}</b><br>Agent Credit: %{x:.1%}<extra></extra>",
    ))

    fig.update_layout(
        title="Agent Attribution % by Model",
        xaxis_title="Attribution Share",
        xaxis_tickformat=".0%",
        height=400,
        margin=dict(l=150),
    )

    return fig


@callback(
    Output("exec-comparison-chart", "figure"),
    Input("exec-comparison-chart", "id"),
)
def update_comparison_chart(_):
    results = get_attribution_results()
    if len(results) == 0:
        return go.Figure()

    lt = results[results["model_name"] == "last_touch"].sort_values("channel_id")
    sh = results[results["model_name"] == "shapley"].sort_values("channel_id")

    if len(lt) == 0 or len(sh) == 0:
        return go.Figure()

    channels = lt["channel_id"].tolist()
    display_names = [get_channel_display_name(ch) for ch in channels]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Last-Touch (GA4)",
        x=display_names,
        y=lt["attribution_pct"].values,
        marker_color=COLORS["text_secondary"],
    ))
    fig.add_trace(go.Bar(
        name="Shapley Value",
        x=display_names,
        y=sh["attribution_pct"].values,
        marker_color=COLORS["highlight"],
    ))

    fig.update_layout(
        title="Channel Attribution: Last-Touch vs. Shapley",
        barmode="group",
        yaxis_title="Attribution Share",
        yaxis_tickformat=".0%",
        xaxis_tickangle=-45,
        height=400,
        legend=dict(x=0.7, y=0.95),
    )

    return fig
