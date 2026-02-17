"""Budget reallocation slider interface component."""
from dash import html, dcc
import dash_bootstrap_components as dbc


def budget_sliders(
    channels: list,
    current_spend: dict,
    component_id: str = "budget-sliders",
):
    """
    Budget reallocation slider interface.

    Args:
        channels: List of channel_ids.
        current_spend: Dict of channel_id → current spend.
        component_id: HTML id prefix.
    """
    sliders = []
    for ch in channels:
        spend = current_spend.get(ch, 0)
        display_name = ch.replace("_", " ").title()

        sliders.append(
            html.Div(
                [
                    html.Label(display_name, className="budget-channel-label"),
                    html.Div(
                        [
                            dcc.Slider(
                                id={"type": "budget-slider", "channel": ch},
                                min=0,
                                max=spend * 3,
                                value=spend,
                                step=spend * 0.05,
                                marks={
                                    0: "0",
                                    int(spend): f"${spend/1000:.0f}K",
                                    int(spend * 2): f"${spend*2/1000:.0f}K",
                                },
                                tooltip={"placement": "bottom"},
                                className="budget-slider",
                            ),
                        ],
                        className="budget-slider-container",
                    ),
                ],
                className="budget-slider-row",
            )
        )

    return html.Div(
        [
            html.H4("Channel Budget Allocation", className="component-title"),
            html.Div(sliders, id=component_id, className="budget-sliders"),
        ]
    )
