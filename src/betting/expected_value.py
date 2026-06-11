"""Expected value and edge calculations for flat 1-unit stakes.

These are research utilities for historical analysis only — they make no
claims about future profitability.
"""


def _validate_probability(name: str, prob: float) -> None:
    if not 0 <= prob <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {prob}.")


def expected_value_per_unit(prob_win: float, decimal_odds: float) -> float:
    """Expected profit of a 1-unit stake at the given decimal odds.

    EV = prob_win * (decimal_odds - 1) - (1 - prob_win)

    Args:
        prob_win: Model-estimated probability the bet wins, in [0, 1].
        decimal_odds: Decimal odds offered. Must be greater than 1.

    Returns:
        Expected profit in units (negative means expected loss).

    Raises:
        ValueError: If ``prob_win`` is outside [0, 1] or ``decimal_odds`` <= 1.
    """
    _validate_probability("prob_win", prob_win)
    if decimal_odds <= 1:
        raise ValueError(f"Decimal odds must be greater than 1, got {decimal_odds}.")
    return prob_win * (decimal_odds - 1.0) - (1.0 - prob_win)


def edge(model_prob: float, fair_market_prob: float) -> float:
    """Difference between model probability and fair (no-vig) market probability.

    Positive values mean the model thinks the outcome is more likely than the
    market does.

    Args:
        model_prob: Model-estimated probability, in [0, 1].
        fair_market_prob: Vig-removed market probability, in [0, 1].

    Returns:
        ``model_prob - fair_market_prob``.

    Raises:
        ValueError: If either probability is outside [0, 1].
    """
    _validate_probability("model_prob", model_prob)
    _validate_probability("fair_market_prob", fair_market_prob)
    return model_prob - fair_market_prob
