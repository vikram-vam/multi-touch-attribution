"""
Erie Insurance — Multi-Channel Attribution Intelligence
Plotly Dash Application Entry Point

Usage: python app/app.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc

from app.theme import EXTERNAL_STYLESHEETS, COLORS
from app.components.components import navbar

# Import page layouts
from app.pages import (
    executive_summary,
    attribution_comparison,
    journey_paths,
    budget_optimizer,
    identity_resolution,
    channel_deepdive,
    validation,
    technical_appendix,
)


# ── Create Dash App ──
app = Dash(
    __name__,
    external_stylesheets=EXTERNAL_STYLESHEETS,
    suppress_callback_exceptions=True,
    title="Erie MCA Intelligence",
    update_title="Loading...",
    assets_folder=str(PROJECT_ROOT / "app" / "assets"),
)

server = app.server  # For production deployment


# ── App Layout ──
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    navbar(),
    html.Div(id="page-content"),
], style={"backgroundColor": COLORS["bg_primary"], "minHeight": "100vh"})


# ── Page Routing ──
@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname):
    pages = {
        "/": executive_summary.layout,
        "/attribution": attribution_comparison.layout,
        "/journeys": journey_paths.layout,
        "/budget": budget_optimizer.layout,
        "/identity": identity_resolution.layout,
        "/channels": channel_deepdive.layout,
        "/validation": validation.layout,
        "/technical": technical_appendix.layout,
    }

    page_func = pages.get(pathname, executive_summary.layout)
    return page_func()


# ── Run ──
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
