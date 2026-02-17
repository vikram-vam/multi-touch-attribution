"""KPI metric card component."""
from dash import html


def metric_card(label: str, value: str, delta: str = None,
                delta_direction: str = None, subtitle: str = None):
    """
    Premium metric card with optional delta indicator.

    Args:
        label: Metric name (e.g., "Total Journeys").
        value: Pre-formatted metric value (e.g., "150,000").
        delta: Pre-formatted delta string (e.g., "+26.4pp").
        delta_direction: 'positive' or 'negative' for coloring.
        subtitle: Optional small text below value.
    """
    children = [
        html.Div(label, className="metric-label"),
        html.Div(value, className="metric-value"),
    ]

    if delta:
        delta_class = f"metric-delta {delta_direction or ''}"
        children.append(html.Div(delta, className=delta_class))

    if subtitle:
        children.append(html.Div(subtitle, className="metric-subtitle"))

    return html.Div(children, className="metric-card animate-in")
