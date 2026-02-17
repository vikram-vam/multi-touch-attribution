"""Chart transition animations component."""
from dash import html, dcc, clientside_callback, Output, Input


def animated_chart(chart_id: str, height: str = "400px"):
    """
    Chart container with CSS transition animation support.

    Wraps a dcc.Graph with fade-in animation on update.
    """
    return html.Div(
        dcc.Graph(
            id=chart_id,
            style={"height": height},
            config={"displayModeBar": False},
            animate=True,
            animation_options={"frame": {"redraw": True}, "transition": {"duration": 500}},
        ),
        className="animated-chart-container animate-in",
    )


def chart_with_loading(chart_id: str, height: str = "400px"):
    """Chart container with loading spinner overlay."""
    return dcc.Loading(
        dcc.Graph(
            id=chart_id,
            style={"height": height},
            config={"displayModeBar": False},
        ),
        type="circle",
        color="#6366f1",
    )


# Clientside callback for smooth Plotly frame transitions
TRANSITION_CONFIG = {
    "transition": {"duration": 500, "easing": "cubic-in-out"},
    "frame": {"duration": 500, "redraw": True},
}
