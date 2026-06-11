"""Market baseline: the sportsbook's vig-removed fair probability as prediction.

This is the bar every trained model must beat. It involves no training — the
market's fair probability for a team is simply assigned to that team's row.
Per docs/tech_stack.md, evaluation priority is probability quality (log loss,
Brier, calibration) before any ROI consideration.
"""

import polars as pl

REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "is_home",
    "home_fair_market_prob",
    "away_fair_market_prob",
)


def predict_market_baseline(df: pl.DataFrame) -> pl.DataFrame:
    """Assign the fair market win probability to each team-game row.

    Home rows (``is_home == 1``) receive ``home_fair_market_prob``; away rows
    (``is_home == 0``) receive ``away_fair_market_prob``.

    Args:
        df: Team-game rows joined with market features, containing the
            required columns.

    Returns:
        The input DataFrame with ``predicted_win_prob`` appended. Row count
        and existing columns are unchanged.

    Raises:
        ValueError: If required columns are missing, ``is_home`` contains
            values other than 0 or 1, or a fair market probability is null or
            outside [0, 1].
    """
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"input frame is missing required columns: {missing}")

    bad_is_home = df.filter(~pl.col("is_home").is_in([0, 1]) | pl.col("is_home").is_null())
    if bad_is_home.height > 0:
        bad_values = bad_is_home.get_column("is_home").unique().to_list()
        raise ValueError(f"is_home must contain only 0 or 1, found: {bad_values}")

    for col in ("home_fair_market_prob", "away_fair_market_prob"):
        invalid = df.filter(
            pl.col(col).is_null() | (pl.col(col) < 0) | (pl.col(col) > 1)
        )
        if invalid.height > 0:
            bad_values = invalid.get_column(col).unique().to_list()
            raise ValueError(f"{col} must be within [0, 1], found: {bad_values}")

    return df.with_columns(
        pl.when(pl.col("is_home") == 1)
        .then(pl.col("home_fair_market_prob"))
        .otherwise(pl.col("away_fair_market_prob"))
        .alias("predicted_win_prob")
    )
