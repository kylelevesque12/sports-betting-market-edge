"""Conversions between American odds, decimal odds, and implied probability.

Implied probabilities computed here include the sportsbook's vig; use
``src.betting.vig_removal`` to obtain fair (no-vig) probabilities.
"""


def _validate_american(odds: float) -> None:
    if odds == 0:
        raise ValueError("American odds cannot be 0.")


def _validate_decimal(decimal_odds: float) -> None:
    if decimal_odds <= 1:
        raise ValueError(f"Decimal odds must be greater than 1, got {decimal_odds}.")


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds.

    Args:
        odds: American odds (e.g. -150 or +130). Cannot be 0.

    Returns:
        Decimal odds (total return per 1 unit staked, including stake).

    Raises:
        ValueError: If ``odds`` is 0.
    """
    _validate_american(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def american_to_implied_probability(odds: float) -> float:
    """Convert American odds to the implied probability (vig included).

    Args:
        odds: American odds (e.g. -150 or +130). Cannot be 0.

    Returns:
        Implied probability in (0, 1).

    Raises:
        ValueError: If ``odds`` is 0.
    """
    return decimal_to_implied_probability(american_to_decimal(odds))


def decimal_to_implied_probability(decimal_odds: float) -> float:
    """Convert decimal odds to the implied probability (vig included).

    Args:
        decimal_odds: Decimal odds. Must be greater than 1.

    Returns:
        Implied probability in (0, 1).

    Raises:
        ValueError: If ``decimal_odds`` is not greater than 1.
    """
    _validate_decimal(decimal_odds)
    return 1.0 / decimal_odds
