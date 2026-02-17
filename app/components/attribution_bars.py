"""Side-by-side attribution comparison bar chart component."""
from dash import dcc
import plotly.graph_objects as go

from app.theme import COLORS, CHART_COLORS


def attribution_bars(
    data: dict,
    chart_id: str = "attribution-bars",
    title: str = "Attribution by Channel",
    height: str = "400px",
):
    """
    Grouped bar chart comparing attribution across models.

    Args:
        data: Dict of {model_name: {channel_id: attribution_pct}}.
        chart_id: HTML id for the graph.
        title: Chart title.
        height: CSS height string.
    """
    fig = go.Figure()

    for i, (model, channel_data) in enumerate(data.items()):
        channels = sorted(channel_data.keys())
        values = [channel_data[ch] for ch in channels]
        color = CHART_COLORS[i % len(CHART_COLORS)]

        fig.add_trace(go.Bar(
            name=model.replace("_", " ").title(),
            x=channels,
            y=values,
            marker_color=color,
            opacity=0.85,
        ))

    fig.update_layout(
        title=title,
        barmode="group",
        xaxis_title="Channel",
        yaxis_title="Attribution %",
        yaxis_tickformat=".0%",
        font=dict(color=COLORS.get("text", "#e0e0e0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=20, t=50, b=80),
        legend=dict(orientation="h", y=1.12),
    )

    return dcc.Graph(
        id=chart_id,
        figure=fig,
        style={"height": height},
        config={"displayModeBar": False},
    )
