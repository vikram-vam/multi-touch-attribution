"""
Page 2 — Attribution Model Comparison (EP-2)
Side-by-side comparison of all 17 attribution models.
"""

from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from app.data_store import (
    get_attribution_results, get_channel_display_name, get_available_models
)
from app.components.components import (
    section_header, chart_container, insight_callout
)
from app.theme import COLORS, get_channel_color


def layout():
    models = get_available_models()

    return html.Div([
        section_header(
            "Attribution Model Comparison",
            "17 models across 6 tiers — from rule-based heuristics to causal deep learning"
        ),

        # Model selector dropdown
        dbc.Row([
            dbc.Col([
                html.Label("Select Models to Compare", className="metric-label"),
                dcc.Dropdown(
                    id="attr-model-select",
                    options=[{"label": m.replace("_", " ").title(), "value": m} for m in models],
                    value=["last_touch", "shapley", "markov_order_2", "ensemble"] if len(models) > 3 else models[:3],
                    multi=True,
                    style={"backgroundColor": COLORS["bg_tertiary"], "color": COLORS["text_primary"]},
                ),
            ], md=12),
        ], className="mb-4"),

        # Main comparison chart
        dbc.Row([
            dbc.Col(chart_container(
                "Channel Attribution by Selected Models", "attr-grouped-bar", "500px"
            ), md=12),
        ]),

        # Heatmap and correlation
        dbc.Row([
            dbc.Col(chart_container(
                "Attribution Heatmap (Model × Channel)", "attr-heatmap", "450px"
            ), md=7),
            dbc.Col(chart_container(
                "Model Agreement (Spearman ρ)", "attr-correlation", "450px"
            ), md=5),
        ]),

        # Insight
        html.Div(id="attr-insight-container"),

    ], className="page-container")


@callback(
    Output("attr-grouped-bar", "figure"),
    Input("attr-model-select", "value"),
)
def update_grouped_bar(selected_models):
    results = get_attribution_results()
    if len(results) == 0 or not selected_models:
        return go.Figure()

    filtered = results[results["model_name"].isin(selected_models)]
    fig = go.Figure()

    for model in selected_models:
        model_data = filtered[filtered["model_name"] == model].sort_values("channel_id")
        if len(model_data) > 0:
            channels = [get_channel_display_name(c) for c in model_data["channel_id"]]
            fig.add_trace(go.Bar(
                name=model.replace("_", " ").title(),
                x=channels,
                y=model_data["attribution_pct"].values,
            ))

    fig.update_layout(
        barmode="group",
        yaxis_title="Attribution Share",
        yaxis_tickformat=".0%",
        xaxis_tickangle=-45,
        legend=dict(x=0, y=1.12, orientation="h"),
        height=500,
    )
    return fig


@callback(
    Output("attr-heatmap", "figure"),
    Input("attr-model-select", "value"),
)
def update_heatmap(selected_models):
    results = get_attribution_results()
    if len(results) == 0 or not selected_models:
        return go.Figure()

    filtered = results[results["model_name"].isin(selected_models)]
    pivot = filtered.pivot_table(
        index="channel_id", columns="model_name",
        values="attribution_pct", aggfunc="first",
    ).fillna(0)

    display_idx = [get_channel_display_name(c) for c in pivot.index]
    display_cols = [m.replace("_", " ").title() for m in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=display_cols,
        y=display_idx,
        colorscale="Viridis",
        hovertemplate="<b>%{y}</b> × %{x}<br>Share: %{z:.1%}<extra></extra>",
    ))

    fig.update_layout(height=450, margin=dict(l=150))
    return fig


@callback(
    Output("attr-correlation", "figure"),
    Input("attr-model-select", "value"),
)
def update_correlation(selected_models):
    results = get_attribution_results()
    if len(results) == 0 or not selected_models or len(selected_models) < 2:
        return go.Figure()

    from scipy.stats import spearmanr

    filtered = results[results["model_name"].isin(selected_models)]
    pivot = filtered.pivot_table(
        index="channel_id", columns="model_name",
        values="attribution_pct", aggfunc="first",
    ).fillna(0)

    n = len(selected_models)
    corr_matrix = np.ones((n, n))
    for i, m1 in enumerate(selected_models):
        for j, m2 in enumerate(selected_models):
            if m1 in pivot.columns and m2 in pivot.columns and i != j:
                rho, _ = spearmanr(pivot[m1], pivot[m2])
                corr_matrix[i, j] = rho

    display_names = [m.replace("_", " ").title() for m in selected_models]

    fig = go.Figure(go.Heatmap(
        z=corr_matrix,
        x=display_names,
        y=display_names,
        colorscale="RdYlGn",
        zmin=-1, zmax=1,
        text=np.around(corr_matrix, 2),
        texttemplate="%{text}",
        hovertemplate="<b>%{x} × %{y}</b><br>ρ = %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(height=450, title="Spearman Rank Correlation")
    return fig


@callback(
    Output("attr-insight-container", "children"),
    Input("attr-model-select", "value"),
)
def update_insight(selected_models):
    results = get_attribution_results()
    if len(results) == 0 or not selected_models:
        return html.Div()

    filtered = results[results["model_name"].isin(selected_models)]
    pivot = filtered.pivot_table(
        index="channel_id", columns="model_name",
        values="attribution_pct", aggfunc="first",
    ).fillna(0)

    if len(pivot.columns) > 1:
        ranges = pivot.max(axis=1) - pivot.min(axis=1)
        top_ch = ranges.idxmax()
        top_range = ranges.max()
        return insight_callout(
            f"Largest model disagreement: {get_channel_display_name(top_ch)} "
            f"({top_range*100:.1f}pp range). This channel's value depends heavily "
            f"on the modeling approach — exactly the type of insight MCA provides.",
            icon="📊"
        )

    return html.Div()
