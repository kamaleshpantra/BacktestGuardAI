"""
validate_prices.py
------------------
Validates a list of raw price values before they enter the backtesting engine.

Each check is intentionally simple and deterministic. The function never
silently ignores bad data — it collects every problem it finds and returns
them all so the caller can fix everything in one pass.
"""

import math


def validate_prices(prices: list) -> list[str]:
    """
    Check a list of price values for common data quality problems.

    Parameters
    ----------
    prices : list
        A flat list of numeric price values (e.g. daily closing prices).

    Returns
    -------
    list[str]
        A list of human-readable error strings describing every problem found.
        An empty list means the data passed all checks.

    Examples
    --------
    >>> validate_prices([100.0, 101.5, 99.0])
    []
    >>> validate_prices([100.0, -5.0, float('inf')])
    ['index 1: price must be positive, got -5.0',
     'index 2: price is not finite (inf or -inf)']
    """
    errors: list[str] = []

    # Rule 1: input must be a list
    if not isinstance(prices, list):
        errors.append(
            f"prices must be a list, got {type(prices).__name__}"
        )
        return errors  # cannot iterate further; return immediately

    # Rule 2: list must not be empty
    if len(prices) == 0:
        errors.append("prices list is empty")
        return errors  # nothing to iterate over

    # Rule 3: check each element
    for index, value in enumerate(prices):

        # 3a: must be a real number (int or float — not str, None, list, etc.)
        if not isinstance(value, (int, float)):
            errors.append(
                f"index {index}: expected a number, got {type(value).__name__}"
            )
            continue  # skip further checks for this element

        # 3b: must be finite (rules out float('inf'), float('-inf'), float('nan'))
        if not math.isfinite(value):
            errors.append(
                f"index {index}: price is not finite (inf or -inf or nan)"
            )
            continue  # a non-finite value cannot be meaningfully compared to 0

        # 3c: must be strictly positive
        if value <= 0:
            errors.append(
                f"index {index}: price must be positive, got {value}"
            )

    return errors
