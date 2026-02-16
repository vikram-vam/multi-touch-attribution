"""
Erie MCA Demo - Main Dash Application
Multi-page interactive dashboard for multi-channel attribution analysis
"""

import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from loguru import logger
import sys

# Configure app
app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

# Server for deployment
server = app.server

# App title
app.title = "Erie Insurance - Multi-Channel Attribution Demo"

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                background-color: #1a1a1a;
            }
            .navbar-brand {
                font-weight: 600;
                font-size: 1.3rem;
            }
            .card {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 8px;
                margin-bottom: 1.5rem;
            }
            .metric-card {
                text-align: center;
                padding: 1.5rem;
            }
            .metric-value {
                font-size: 2.5rem;
                font-weight: 700;
                color: #1f4788;
            }
            .metric-label {
                font-size: 0.9rem;
                color: #999;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .insight-box {
                background: linear-gradient(135deg, #1f4788 0%, #2d5aa8 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 8px;
                margin-bottom: 1rem;
            }
            .chart-container {
                background-color: #2d2d2d;
                padding: 1rem;
                border-radius: 8px;
            }
            h1, h2, h3 {
                color: #ffffff;
            }
            .nav-link.active {
                background-color: #1f4788 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Navigation bar
navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.A(
                            dbc.Row(
                                [
                                    dbc.Col(html.I(className="fas fa-chart-line me-2")),
                                    dbc.Col(
                                        dbc.NavbarBrand(
                                            "Erie Insurance - Multi-Channel Attribution",
                                            className="ms-2"
                                        )
                                    ),
                                ],
                                align="center",
                                className="g-0",
                            ),
                            href="/",
                            style={"textDecoration": "none"},
                        )
                    ),
                ],
                align="center",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Nav(
                            [
                                dbc.NavItem(dbc.NavLink("Executive Summary", href="/", active="exact")),
                                dbc.NavItem(dbc.NavLink("Model Comparison", href="/model-comparison", active="exact")),
                                dbc.NavItem(dbc.NavLink("Journey Explorer", href="/journey-explorer", active="exact")),
                                dbc.NavItem(dbc.NavLink("Budget Optimizer", href="/budget-optimizer", active="exact")),
                                dbc.NavItem(dbc.NavLink("Technical Appendix", href="/technical-appendix", active="exact")),
                            ],
                            navbar=True,
                        )
                    )
                ],
                align="center",
            ),
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
    className="mb-4",
)

# App layout
app.layout = dbc.Container(
    [
        navbar,
        dash.page_container
    ],
    fluid=True,
    className="px-4"
)


if __name__ == "__main__":
    logger.info("Starting Erie MCA Demo application")
    app.run_server(debug=True, host="0.0.0.0", port=8050)
