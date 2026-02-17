"""Sidebar parameter controls component."""
from dash import html, dcc
import dash_bootstrap_components as dbc


def parameter_panel(controls: list, panel_id: str = "param-panel"):
    """
    Sidebar parameter panel with sliders and dropdowns.

    Args:
        controls: List of dicts with keys: label, component (dcc element).
        panel_id: HTML id for the panel container.
    """
    items = []
    for ctrl in controls:
        items.append(
            html.Div(
                [
                    html.Label(ctrl["label"], className="param-label"),
                    ctrl["component"],
                ],
                className="param-control",
            )
        )

    return html.Div(
        items,
        id=panel_id,
        className="parameter-panel",
    )


def make_slider(id: str, min_val: float, max_val: float,
                value: float, step: float = None, marks: dict = None):
    """Create a styled slider for parameter controls."""
    return dcc.Slider(
        id=id,
        min=min_val,
        max=max_val,
        value=value,
        step=step,
        marks=marks,
        className="param-slider",
    )


def make_dropdown(id: str, options: list, value=None, multi: bool = False):
    """Create a styled dropdown for parameter controls."""
    return dcc.Dropdown(
        id=id,
        options=[{"label": o, "value": o} for o in options],
        value=value,
        multi=multi,
        className="param-dropdown",
    )
