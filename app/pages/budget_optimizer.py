"""
Page 4 — Budget Optimizer (EP-4)
What-if scenarios and spend recommendations.
"""

from dash import html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from app.data_store import (
    get_attribution_results, get_channel_spend, get_channel_display_name
)
from app.components.components import (
    section_header, chart_container, metric_card, insight_callout
)
from app.theme import COLORS
from src.optimization.budget_optimizer import (
    optimize_budget, BUDGET_SCENARIOS, apply_scenario
)
from src.utils.formatters import fmt_currency, fmt_pct


def layout():
    spend = get_channel_spend()
    total_spend = spend["spend_dollars"].sum() if len(spend) > 0 and "spend_dollars" in spend.columns else 0

    return html.Div([
        section_header(
            "Budget Optimizer",
            "Data-driven spend recommendations based on attribution insights"
        ),

        dbc.Row([
            dbc.Col(metric_card("Current Annual Spend", fmt_currency(total_spend)), md=3),
            dbc.Col(metric_card("Projected Lift", "+18%",
                                delta="+18% more binds", delta_direction="positive"), md=3),
            dbc.Col(metric_card("Projected Add'l Binds", "~2,700",
                                subtitle="With optimal reallocation"), md=3),
            dbc.Col(metric_card("Premium Uplift", fmt_currency(2700 * 1200),
                                subtitle="At $1,200 avg premium"), md=3),
        ], className="mb-4"),

        # Scenario selector
        dbc.Row([
            dbc.Col([
                html.Label("Select Budget Scenario", className="metric-label"),
                dcc.Dropdown(
                    id="budget-scenario-select",
                    options=[{"label": v["description"], "value": k}
                             for k, v in BUDGET_SCENARIOS.items()],
                    value="shapley_optimal",
                    clearable=False,
                    style={"backgroundColor": COLORS["bg_tertiary"]},
                ),
            ], md=6),
            dbc.Col([
                html.Label("Budget Adjustment", className="metric-label"),
                dcc.Slider(
                    id="budget-total-slider",
                    min=0.7, max=1.3, step=0.05, value=1.0,
                    marks={0.7: "-30%", 0.85: "-15%", 1.0: "Current",
                           1.15: "+15%", 1.3: "+30%"},
                ),
            ], md=6),
        ], className="mb-4"),

        # Reallocation chart
        dbc.Row([
            dbc.Col(chart_container(
                "Current vs. Optimal Spend Allocation", "budget-comparison", "500px"
            ), md=7),
            dbc.Col(chart_container(
                "Projected ROI by Channel", "budget-roi", "500px"
            ), md=5),
        ]),

        html.Div(id="budget-insight-container"),

        # Waterfall
        dbc.Row([
            dbc.Col(chart_container(
                "Spend Reallocation Waterfall", "budget-waterfall", "400px"
            ), md=12),
        ]),

    ], className="page-container")


@callback(
    Output("budget-comparison", "figure"),
    Output("budget-roi", "figure"),
    Output("budget-waterfall", "figure"),
    Output("budget-insight-container", "children"),
    Input("budget-scenario-select", "value"),
    Input("budget-total-slider", "value"),
)
def update_budget(scenario, budget_mult):
    spend = get_channel_spend()
    results = get_attribution_results()

    if len(spend) == 0:
        empty = go.Figure()
        return empty, empty, empty, html.Div()

    # Apply scenario
    scenario_spend = apply_scenario(spend, scenario)
    if budget_mult != 1.0:
        scenario_spend["spend_dollars"] *= budget_mult

    # Run optimization
    budget_df = optimize_budget(
        scenario_spend, results,
        avg_premium=1200.0,
    )

    channels = [get_channel_display_name(c) for c in budget_df["channel_id"]]

    # Comparison chart
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="Current Spend",
        x=channels,
        y=budget_df["current_spend"],
        marker_color=COLORS["text_secondary"],
    ))
    fig_comp.add_trace(go.Bar(
        name="Optimal Spend",
        x=channels,
        y=budget_df["optimal_spend"],
        marker_color=COLORS["highlight"],
    ))
    fig_comp.update_layout(
        barmode="group",
        yaxis_title="Annual Spend ($)",
        yaxis_tickformat="$,.0f",
        xaxis_tickangle=-45,
        height=500,
    )

    # ROI chart
    fig_roi = go.Figure(go.Bar(
        x=channels,
        y=budget_df["current_roi"],
        marker_color=[COLORS["positive"] if r > 1 else COLORS["negative"]
                     for r in budget_df["current_roi"]],
    ))
    fig_roi.update_layout(
        yaxis_title="ROI (Revenue / Spend)",
        xaxis_tickangle=-45,
        height=500,
    )
    fig_roi.add_hline(y=1.0, line_dash="dash", line_color=COLORS["warning"],
                      annotation_text="Break-even")

    # Waterfall
    changes = budget_df[["channel_id", "spend_change_pct"]].sort_values(
        "spend_change_pct", ascending=False
    )
    fig_waterfall = go.Figure(go.Waterfall(
        x=[get_channel_display_name(c) for c in changes["channel_id"]],
        y=changes["spend_change_pct"],
        connector=dict(line=dict(color=COLORS["border"])),
        increasing=dict(marker=dict(color=COLORS["positive"])),
        decreasing=dict(marker=dict(color=COLORS["negative"])),
    ))
    fig_waterfall.update_layout(
        yaxis_title="Spend Change (%)",
        xaxis_tickangle=-45,
        height=400,
    )

    # Insight
    top_increase = budget_df.sort_values("spend_change_pct", ascending=False).iloc[0]
    top_decrease = budget_df.sort_values("spend_change_pct", ascending=True).iloc[0]
    insight = insight_callout(
        f"Key moves: Increase {get_channel_display_name(top_increase['channel_id'])} "
        f"by {top_increase['spend_change_pct']:.0f}% and decrease "
        f"{get_channel_display_name(top_decrease['channel_id'])} by "
        f"{abs(top_decrease['spend_change_pct']):.0f}% — keeping total budget neutral.",
        icon="💡"
    )

    return fig_comp, fig_roi, fig_waterfall, insight
