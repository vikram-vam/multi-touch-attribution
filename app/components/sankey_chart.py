"""Plotly Sankey diagram wrapper component."""
from dash import dcc
import plotly.graph_objects as go

from app.theme import COLORS


def sankey_chart(
    sources: list,
    targets: list,
    values: list,
    labels: list,
    chart_id: str = "sankey-chart",
    title: str = "Journey Flow",
    height: str = "500px",
):
    """
    Create a Plotly Sankey diagram with Erie theming.

    Args:
        sources: List of source node indices.
        targets: List of target node indices.
        values: List of flow values.
        labels: List of node labels.
        chart_id: HTML id for the graph.
        title: Chart title.
        height: CSS height string.
    """
    n_nodes = len(labels)
    node_colors = [COLORS["primary"]] * n_nodes

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color=COLORS["border"], width=1),
            label=labels,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(100, 149, 237, 0.3)",
        ),
    ))

    fig.update_layout(
        title=title,
        font=dict(size=12, color=COLORS.get("text", "#e0e0e0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return dcc.Graph(
        id=chart_id,
        figure=fig,
        style={"height": height},
        config={"displayModeBar": False},
    )
