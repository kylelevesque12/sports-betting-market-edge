"""Transform one-row-per-game data into one-row-per-team-game data.

The team-game row is the unit of prediction for this project (CLAUDE.md).

Leakage note: the output contains outcome columns (``points_for``,
``points_against``, ``point_diff``, ``team_win``). These are labels and
inputs for *historical* aggregations (e.g. future rolling features that must
shift past the current game) — they must never be used directly as pre-game
features for the same game.
"""

import polars as pl

REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "game_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_win",
)

OUTPUT_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "game_date",
    "team",
    "opponent",
    "is_home",
    "points_for",
    "points_against",
    "point_diff",
    "team_win",
)


def create_team_game_rows(games: pl.DataFrame) -> pl.DataFrame:
    """Expand each game row into one home-team row and one away-team row.

    Args:
        games: DataFrame with one row per game and the required input columns.

    Returns:
        DataFrame with exactly two rows per game and columns
        ``game_id, season, game_date, team, opponent, is_home, points_for,
        points_against, point_diff, team_win``, sorted by ``game_date``,
        ``game_id``, then ``is_home`` descending (home row first).

    Raises:
        ValueError: If required input columns are missing.
    """
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in games.columns]
    if missing:
        raise ValueError(f"games is missing required columns: {missing}")

    shared = ["game_id", "season", "game_date"]

    home_rows = games.select(
        *shared,
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.lit(1).alias("is_home"),
        pl.col("home_score").alias("points_for"),
        pl.col("away_score").alias("points_against"),
        (pl.col("home_score") - pl.col("away_score")).alias("point_diff"),
        pl.col("home_win").alias("team_win"),
    )

    away_rows = games.select(
        *shared,
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.lit(0).alias("is_home"),
        pl.col("away_score").alias("points_for"),
        pl.col("home_score").alias("points_against"),
        (pl.col("away_score") - pl.col("home_score")).alias("point_diff"),
        (1 - pl.col("home_win")).alias("team_win"),
    )

    return pl.concat([home_rows, away_rows]).sort(
        ["game_date", "game_id", "is_home"], descending=[False, False, True]
    )
