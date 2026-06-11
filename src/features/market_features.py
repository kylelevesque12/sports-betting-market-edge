"""Market-derived features from sportsbook moneyline odds.

Pre-game safety: every feature here is derived purely from the posted odds in
the input row. Provided the odds row itself was captured before game start
(the odds schema carries ``timestamp`` for that check), these features are
knowable before tipoff by construction.

Formulas are not re-derived here — they come from the tested utilities in
``src.betting`` (CLAUDE.md architecture rule).
"""

import polars as pl

from src.betting.odds_conversion import (
    american_to_decimal,
    american_to_implied_probability,
)
from src.betting.vig_removal import remove_vig_two_way

REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "home_american_odds",
    "away_american_odds",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "home_implied_prob_raw",
    "away_implied_prob_raw",
    "home_fair_market_prob",
    "away_fair_market_prob",
    "sportsbook_vig",
    "home_decimal_odds",
    "away_decimal_odds",
)


def add_moneyline_market_features(df: pl.DataFrame) -> pl.DataFrame:
    """Append implied-probability, fair-probability, vig, and decimal-odds columns.

    Applies the scalar betting math utilities row by row. Row-wise application
    is deliberate: it reuses the unit-tested functions (including their input
    validation) instead of re-deriving formulas as Polars expressions.

    Args:
        df: DataFrame containing ``home_american_odds`` and
            ``away_american_odds`` (one row per game per sportsbook per
            timestamp).

    Returns:
        The input DataFrame with the seven market feature columns appended.
        Row count and existing columns are unchanged.

    Raises:
        ValueError: If required input columns are missing, or if any odds
            value is invalid (propagated from the betting math utilities,
            e.g. American odds of 0).
    """
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"odds frame is missing required columns: {missing}")

    home_odds = df.get_column("home_american_odds").to_list()
    away_odds = df.get_column("away_american_odds").to_list()

    home_raw = [american_to_implied_probability(o) for o in home_odds]
    away_raw = [american_to_implied_probability(o) for o in away_odds]
    fair_pairs = [remove_vig_two_way(h, a) for h, a in zip(home_raw, away_raw)]

    return df.with_columns(
        pl.Series("home_implied_prob_raw", home_raw, dtype=pl.Float64),
        pl.Series("away_implied_prob_raw", away_raw, dtype=pl.Float64),
        pl.Series(
            "home_fair_market_prob", [p[0] for p in fair_pairs], dtype=pl.Float64
        ),
        pl.Series(
            "away_fair_market_prob", [p[1] for p in fair_pairs], dtype=pl.Float64
        ),
        pl.Series(
            "sportsbook_vig",
            [h + a - 1.0 for h, a in zip(home_raw, away_raw)],
            dtype=pl.Float64,
        ),
        pl.Series(
            "home_decimal_odds",
            [american_to_decimal(o) for o in home_odds],
            dtype=pl.Float64,
        ),
        pl.Series(
            "away_decimal_odds",
            [american_to_decimal(o) for o in away_odds],
            dtype=pl.Float64,
        ),
    )
