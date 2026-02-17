"""Identity resolution network visualization component."""
from dash import dcc
import plotly.graph_objects as go
import numpy as np

from app.theme import COLORS


def identity_graph_viz(
    identity_data: dict,
    chart_id: str = "identity-graph",
    height: str = "450px",
):
    """
    Identity resolution network visualization.
    Shows fragment-to-profile merging as a scatter network.

    Args:
        identity_data: Dict with 'nodes', 'edges' lists.
        chart_id: HTML id for the graph.
        height: CSS height string.
    """
    nodes = identity_data.get("nodes", [])
    edges = identity_data.get("edges", [])

    if not nodes:
        # Generate placeholder visualization
        return _placeholder_viz(chart_id, height)

    # Node positions (force-directed layout approximation)
    n = len(nodes)
    rng = np.random.default_rng(42)
    x_pos = rng.normal(0, 1, n)
    y_pos = rng.normal(0, 1, n)

    # Color by node type
    colors = []
    sizes = []
    labels = []
    for node in nodes:
        node_type = node.get("type", "fragment")
        if node_type == "persistent_id":
            colors.append(COLORS["primary"])
            sizes.append(15)
        else:
            colors.append(COLORS.get("accent", "#64b5f6"))
            sizes.append(8)
        labels.append(node.get("label", ""))

    # Edge traces
    edge_x = []
    edge_y = []
    for edge in edges:
        src = edge.get("source", 0)
        tgt = edge.get("target", 0)
        if src < n and tgt < n:
            edge_x.extend([x_pos[src], x_pos[tgt], None])
            edge_y.extend([y_pos[src], y_pos[tgt], None])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="rgba(100,149,237,0.3)"),
        hoverinfo="none",
    ))

    fig.add_trace(go.Scatter(
        x=x_pos, y=y_pos,
        mode="markers+text",
        marker=dict(size=sizes, color=colors, opacity=0.8),
        text=labels,
        textposition="top center",
        textfont=dict(size=8, color=COLORS.get("text", "#e0e0e0")),
        hoverinfo="text",
    ))

    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=10),
    )

    return dcc.Graph(
        id=chart_id,
        figure=fig,
        style={"height": height},
        config={"displayModeBar": False},
    )


def _placeholder_viz(chart_id, height):
    """Create a placeholder identity graph."""
    fig = go.Figure()
    fig.add_annotation(text="Identity graph visualization", showarrow=False)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return dcc.Graph(id=chart_id, figure=fig, style={"height": height})
