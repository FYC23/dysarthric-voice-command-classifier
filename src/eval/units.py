"""Human-readable counts for tables (9.2k, 321k, 89.1M). No dependencies, so the demo can use it."""

UNITS = ("", "k", "M", "G", "T")


def human_count(n: float) -> str:
    """9232 -> 9.2k, 321068 -> 321k, 89.1e6 -> 89.1M, 999950 -> 1.0M."""
    value, unit = float(n), 0
    while True:
        decimals = 0 if unit == 0 or abs(round(value, 1)) >= 100 else 1
        shown = round(value, decimals)
        # Round first, then move up a unit if rounding reached 1000 ("1000k" -> "1.0M")
        if abs(shown) < 1000 or unit == len(UNITS) - 1:
            break
        value, unit = value / 1000, unit + 1
    return f"{shown:.{decimals}f}{UNITS[unit]}"
