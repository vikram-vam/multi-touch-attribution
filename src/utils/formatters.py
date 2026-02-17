"""
Number formatting utilities for the UI layer.
Every formatted string in the Dash app should come through these functions.
"""


def fmt_number(value: int | float) -> str:
    """Format a number with comma separators. E.g., 150000 → '150,000'."""
    if isinstance(value, float) and value == int(value):
        value = int(value)
    return f"{value:,}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    """Format a fraction (0.0–1.0) as a percentage string. E.g., 0.342 → '34.2%'."""
    return f"{value * 100:.{decimals}f}%"


def fmt_pct_pp(value: float, decimals: int = 1) -> str:
    """Format a percentage-point change. E.g., 0.264 → '+26.4pp'."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.{decimals}f}pp"


def fmt_currency(value: float, prefix: str = "$", decimals: int = 0) -> str:
    """Format a dollar amount. E.g., 2400000 → '$2,400,000'."""
    if abs(value) >= 1_000_000:
        return f"{prefix}{value / 1_000_000:,.{max(0, decimals)}f}M"
    elif abs(value) >= 1_000:
        return f"{prefix}{value:,.{decimals}f}"
    else:
        return f"{prefix}{value:,.2f}"


def fmt_currency_full(value: float, prefix: str = "$") -> str:
    """Format a full dollar amount without abbreviation. E.g., 2400000 → '$2,400,000'."""
    return f"{prefix}{value:,.0f}"


def fmt_multiplier(value: float) -> str:
    """Format a multiplier value. E.g., 4.8 → '4.8×'."""
    return f"{value:.1f}×"


def fmt_delta(value: float, direction: str = "auto") -> tuple[str, str]:
    """
    Format a delta value and determine direction.

    Returns:
        (formatted_string, direction) where direction is 'positive' or 'negative'.
    """
    if direction == "auto":
        direction = "positive" if value >= 0 else "negative"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%", direction


def fmt_rank(rank: int) -> str:
    """Format a rank with ordinal suffix. E.g., 1 → '1st', 2 → '2nd'."""
    if 11 <= rank % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank % 10, "th")
    return f"{rank}{suffix}"


def fmt_spend_change(current: float, proposed: float) -> str:
    """Format a spend change with direction. E.g., '$15M → $12M (−20%)'."""
    pct = (proposed - current) / current * 100 if current != 0 else 0
    sign = "+" if pct >= 0 else ""
    return f"{fmt_currency(current)} → {fmt_currency(proposed)} ({sign}{pct:.1f}%)"
