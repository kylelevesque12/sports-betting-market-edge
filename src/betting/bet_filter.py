"""Bet decision rules: whether an edge/EV combination qualifies as a bet.

Research utilities for historical analysis only — not betting advice and no
claim of profitability. Flat 1-unit staking is assumed elsewhere; this module
makes no staking decisions.
"""

from src.betting.expected_value import edge, expected_value_per_unit


def _validate_threshold(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")


def should_bet(
    model_prob: float,
    fair_market_prob: float,
    ev: float,
    min_edge: float = 0.03,
    min_ev: float = 0.02,
) -> bool:
    """Decide whether a single side qualifies as a historical-EV bet.

    Qualifies only if both the model edge and the expected value meet their
    minimum thresholds (inclusive).

    Args:
        model_prob: Model-estimated win probability, in [0, 1].
        fair_market_prob: Vig-removed market probability, in [0, 1].
        ev: Expected profit per 1-unit stake for this side.
        min_edge: Minimum required edge (model_prob - fair_market_prob). Must be >= 0.
        min_ev: Minimum required expected value per unit. Must be >= 0.

    Returns:
        True if edge >= min_edge and ev >= min_ev, else False.

    Raises:
        ValueError: If a probability is outside [0, 1] or a threshold is negative.
    """
    _validate_threshold("min_edge", min_edge)
    _validate_threshold("min_ev", min_ev)
    side_edge = edge(model_prob, fair_market_prob)  # validates both probabilities
    return side_edge >= min_edge and ev >= min_ev


def classify_bet_side(
    model_home_prob: float,
    fair_home_prob: float,
    home_decimal_odds: float,
    away_decimal_odds: float,
    min_edge: float = 0.03,
    min_ev: float = 0.02,
) -> dict:
    """Evaluate both sides of a two-way moneyline and pick the qualifying side.

    Away probabilities are the complements of the home probabilities. Each
    side's edge and EV are computed, then tested with :func:`should_bet`.

    Args:
        model_home_prob: Model-estimated home win probability, in [0, 1].
        fair_home_prob: Vig-removed market home win probability, in [0, 1].
        home_decimal_odds: Decimal odds offered on the home side. Must be > 1.
        away_decimal_odds: Decimal odds offered on the away side. Must be > 1.
        min_edge: Minimum required edge. Must be >= 0.
        min_ev: Minimum required EV per unit. Must be >= 0.

    Returns:
        Dict with keys ``bet_flag``, ``side`` ("home", "away", or "no_bet"),
        ``model_prob``, ``fair_market_prob``, ``edge``, ``expected_value``,
        and ``decimal_odds``. When no side qualifies, ``bet_flag`` is False,
        ``side`` is "no_bet", and the side-specific fields are None.

    Raises:
        ValueError: If inputs are invalid, or if both sides qualify
            (opposite sides of the same game cannot both hold an edge unless
            inputs are inconsistent).
    """
    model_away_prob = 1.0 - model_home_prob
    fair_away_prob = 1.0 - fair_home_prob

    # EV computation validates probabilities and decimal odds.
    home_ev = expected_value_per_unit(model_home_prob, home_decimal_odds)
    away_ev = expected_value_per_unit(model_away_prob, away_decimal_odds)

    home_qualifies = should_bet(model_home_prob, fair_home_prob, home_ev, min_edge, min_ev)
    away_qualifies = should_bet(model_away_prob, fair_away_prob, away_ev, min_edge, min_ev)

    if home_qualifies and away_qualifies:
        raise ValueError(
            "Both sides qualify as bets; inputs are inconsistent "
            "(check fair probabilities and odds)."
        )

    if home_qualifies:
        return {
            "bet_flag": True,
            "side": "home",
            "model_prob": model_home_prob,
            "fair_market_prob": fair_home_prob,
            "edge": edge(model_home_prob, fair_home_prob),
            "expected_value": home_ev,
            "decimal_odds": home_decimal_odds,
        }
    if away_qualifies:
        return {
            "bet_flag": True,
            "side": "away",
            "model_prob": model_away_prob,
            "fair_market_prob": fair_away_prob,
            "edge": edge(model_away_prob, fair_away_prob),
            "expected_value": away_ev,
            "decimal_odds": away_decimal_odds,
        }
    return {
        "bet_flag": False,
        "side": "no_bet",
        "model_prob": None,
        "fair_market_prob": None,
        "edge": None,
        "expected_value": None,
        "decimal_odds": None,
    }
