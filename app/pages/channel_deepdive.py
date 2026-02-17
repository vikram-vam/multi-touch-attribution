"""
Page 6 — Channel Deep-Dive (EP-6)
"""

from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from app.data_store import (
    get_attribution_results, get_journeys, get_channel_spend,
    get_channel_list, get_channel_display_name
)
from app.components.components import section_header, chart_container, metric_card, insight_callout
from app.theme import COLORS, get_channel_color
from src.utils.formatters import fmt_number, fmt_pct, fmt_currency


def layout():
    channels = get_channel_list()

    return html.Div([
        section_header(
            "Channel Deep-Dive",
            "Detailed performance analysis for each marketing channel"
        ),

        dbc.Row([
            dbc.Col([
                html.Label("Select Channel", className="metric-label"),
                dcc.Dropdown(
                    id="channel-select",
                    options=[{"label": get_channel_display_name(c), "value": c} for c in channels],
                    value=channels[0] if channels else None,
                    clearable=False,
                    style={"backgroundColor": COLORS["bg_tertiary"]},
                ),
            ], md=4),
        ], className="mb-4"),

        html.Div(id="channel-kpi-row"),
        html.Div(id="channel-insight-container"),

        dbc.Row([
            dbc.Col(chart_container(
                "Attribution Across Models", "channel-model-comparison", "400px"
            ), md=6),
            dbc.Col(chart_container(
                "Funnel Position Heatmap", "channel-funnel-position", "400px"
            ), md=6),
        ]),

        dbc.Row([
            dbc.Col(chart_container(
                "Channel Spend Over Time", "channel-spend-trend", "350px"
            ), md=12),
        ]),

    ], className="page-container")


@callback(
    Output("channel-kpi-row", "children"),
    Output("channel-insight-container", "children"),
    Output("channel-model-comparison", "figure"),
    Output("channel-funnel-position", "figure"),
    Output("channel-spend-trend", "figure"),
    Input("channel-select", "value"),
)
def update_channel(selected_channel):
    results = get_attribution_results()
    journeys = get_journeys()
    spend = get_channel_spend()
    empty = go.Figure()

    if not selected_channel or len(results) == 0:
        return html.Div(), html.Div(), empty, empty, empty

    display = get_channel_display_name(selected_channel)

    # Channel KPIs
    ch_results = results[results["channel_id"] == selected_channel]
    avg_pct = ch_results["attribution_pct"].mean() if len(ch_results) > 0 else 0
    total_credit = ch_results.groupby("model_name")["attributed_conversions"].first().mean() if len(ch_results) > 0 else 0

    ch_spend = spend[spend["channel_id"] == selected_channel]["spend_dollars"].sum() if "channel_id" in spend.columns else 0
    cost_per = ch_spend / max(total_credit, 1)

    # Journeys with this channel
    if "channel_set_str" in journeys.columns:
        with_ch = journeys[journeys["channel_set_str"].str.contains(selected_channel, na=False)]
    else:
        with_ch = journeys.head(0)

    ch_conv_rate = with_ch["is_converting"].mean() if len(with_ch) > 0 else 0

    kpi_row = dbc.Row([
        dbc.Col(metric_card(f"{display} — Avg. Attribution", fmt_pct(avg_pct)), md=3),
        dbc.Col(metric_card("Average Credited Conversions", fmt_number(int(total_credit))), md=3),
        dbc.Col(metric_card("Annual Spend", fmt_currency(ch_spend)), md=3),
        dbc.Col(metric_card("Conversion Rate (Journeys incl.)", fmt_pct(ch_conv_rate)), md=3),
    ], className="mb-4")

    # Insight
    lt_row = ch_results[ch_results["model_name"] == "last_touch"]
    sh_row = ch_results[ch_results["model_name"] == "shapley"]
    lt_pct = lt_row["attribution_pct"].values[0] if len(lt_row) > 0 else 0
    sh_pct = sh_row["attribution_pct"].values[0] if len(sh_row) > 0 else 0
    diff = sh_pct - lt_pct
    direction = "over-credits" if diff < 0 else "under-credits"
    insight = insight_callout(
        f"Last-touch {direction} {display} by {abs(diff)*100:.1f}pp compared to Shapley. "
        f"Cost per attributed bind: {fmt_currency(cost_per)}.",
        icon="🎯"
    )

    # Model comparison bar
    ch_models = ch_results.sort_values("attribution_pct", ascending=True)
    fig_models = go.Figure(go.Bar(
        y=ch_models["model_name"].str.replace("_", " ").str.title(),
        x=ch_models["attribution_pct"],
        orientation="h",
        marker_color=[get_channel_color(selected_channel)] * len(ch_models),
    ))
    fig_models.update_layout(
        xaxis_title="Attribution Share",
        xaxis_tickformat=".0%",
        height=400, margin=dict(l=150),
    )

    # Funnel position — simplified as path position histogram
    positions = []
    for _, row in with_ch.iterrows():
        path = row.get("channel_path", [])
        if isinstance(path, str):
            path = path.split("|")
        n = len(path)
        for i, ch in enumerate(path):
            if ch == selected_channel:
                positions.append(i / max(n - 1, 1))

    fig_funnel = go.Figure(go.Histogram(
        x=positions, nbinsx=10,
        marker_color=get_channel_color(selected_channel),
    ))
    fig_funnel.update_layout(
        xaxis_title="Position in Journey (0=first, 1=last)",
        yaxis_title="Count",
        height=400,
    )

    # Spend trend
    if len(spend) > 0 and "month" in spend.columns and "channel_id" in spend.columns:
        ch_monthly = spend[spend["channel_id"] == selected_channel].sort_values("month")
        fig_trend = go.Figure(go.Scatter(
            x=ch_monthly["month"],
            y=ch_monthly["spend_dollars"],
            mode="lines+markers",
            line=dict(color=get_channel_color(selected_channel), width=2),
            marker=dict(size=6),
        ))
        fig_trend.update_layout(
            yaxis_title="Monthly Spend ($)",
            yaxis_tickformat="$,.0f",
            height=350,
        )
    else:
        fig_trend = empty

    return kpi_row, insight, fig_models, fig_funnel, fig_trend
