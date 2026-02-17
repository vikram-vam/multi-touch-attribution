"""Model selector toggle pills component."""
from dash import html, dcc


def model_selector(available_models: list, starred_models: list = None,
                    component_id: str = "model-selector"):
    """
    Toggleable model selection pills.
    Starred models are highlighted with a star marker.
    """
    starred = set(starred_models or [])
    pills = []

    for model in available_models:
        display_name = model.replace("_", " ").title()
        class_name = "model-pill active"
        if model in starred:
            class_name += " starred"

        pills.append(
            html.Button(
                display_name,
                id={"type": "model-pill", "model": model},
                className=class_name,
                n_clicks=0,
            )
        )

    return html.Div(
        [
            html.Div(pills, className="model-selector-container"),
            dcc.Store(id=f"{component_id}-store", data=available_models),
        ]
    )
