"""Interactive journey path explorer component."""
from dash import html, dcc
import dash_bootstrap_components as dbc


def journey_explorer(
    top_paths: list,
    component_id: str = "journey-explorer",
):
    """
    Interactive journey path explorer showing top converting paths.

    Args:
        top_paths: List of dicts with 'path', 'count', 'pct' keys.
        component_id: HTML id prefix.
    """
    rows = []
    for i, path_data in enumerate(top_paths[:15]):
        path_str = path_data.get("path", "")
        count = path_data.get("count", 0)
        pct = path_data.get("pct", 0)

        # Split path into channel badges
        channels = path_str.split("|") if isinstance(path_str, str) else path_str
        badges = []
        for j, ch in enumerate(channels):
            badges.append(html.Span(
                ch.replace("_", " ").title(),
                className="path-channel-badge",
            ))
            if j < len(channels) - 1:
                badges.append(html.Span(" → ", className="path-arrow"))

        rows.append(
            html.Div(
                [
                    html.Div(
                        [html.Span(f"#{i+1}", className="path-rank")] + badges,
                        className="path-channels",
                    ),
                    html.Div(
                        [
                            html.Span(f"{count:,}", className="path-count"),
                            html.Span(f"({pct:.1%})", className="path-pct"),
                        ],
                        className="path-stats",
                    ),
                ],
                className="journey-path-row",
            )
        )

    return html.Div(
        [
            html.H4("Top Converting Paths", className="component-title"),
            html.Div(rows, id=component_id, className="journey-explorer"),
        ]
    )
