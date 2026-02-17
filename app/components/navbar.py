"""Top navigation bar component."""
from dash import html
import dash_bootstrap_components as dbc


def navbar():
    """Top navigation bar with Erie branding and page links."""
    nav_items = [
        dbc.NavItem(dbc.NavLink("Executive Summary", href="/", active="exact")),
        dbc.NavItem(dbc.NavLink("Attribution", href="/attribution", active="exact")),
        dbc.NavItem(dbc.NavLink("Journeys", href="/journeys", active="exact")),
        dbc.NavItem(dbc.NavLink("Budget", href="/budget", active="exact")),
        dbc.NavItem(dbc.NavLink("Identity", href="/identity", active="exact")),
        dbc.NavItem(dbc.NavLink("Channels", href="/channels", active="exact")),
        dbc.NavItem(dbc.NavLink("Validation", href="/validation", active="exact")),
        dbc.NavItem(dbc.NavLink("Technical", href="/technical", active="exact")),
    ]

    return dbc.Navbar(
        dbc.Container(
            [
                html.Div(
                    [
                        html.Div("Erie Insurance", className="navbar-brand-text"),
                        html.Div("Multi-Channel Attribution Intelligence",
                                 className="navbar-brand-sub"),
                    ],
                ),
                dbc.Nav(nav_items, className="ms-auto", navbar=True),
            ],
            fluid=True,
        ),
        className="navbar-custom",
        dark=True,
    )
