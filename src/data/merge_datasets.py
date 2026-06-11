"""Load and merge toy game and odds CSV files.

Structural pipeline step validating the games <-> odds join before any real
data collection. Per docs/tech_stack.md, this module uses Polars; CSV is
acceptable here only because the inputs are tiny toy samples.
"""

import polars as pl

REQUIRED_GAME_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "game_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_win",
)

REQUIRED_ODDS_COLUMNS: tuple[str, ...] = (
    "game_id",
    "sportsbook",
    "market",
    "timestamp",
    "home_team",
    "away_team",
    "home_american_odds",
    "away_american_odds",
    "is_opening_line",
    "is_closing_line",
)


def _validate_columns(
    df: pl.DataFrame, required: tuple[str, ...], frame_name: str
) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def load_games(path: str) -> pl.DataFrame:
    """Load a games CSV and validate its schema.

    Args:
        path: Path to a CSV with one row per game.

    Returns:
        Polars DataFrame with at least the required game columns.

    Raises:
        ValueError: If required game columns are missing.
    """
    games = pl.read_csv(path)
    _validate_columns(games, REQUIRED_GAME_COLUMNS, "games")
    return games


def load_odds(path: str) -> pl.DataFrame:
    """Load an odds CSV and validate its schema.

    Args:
        path: Path to a CSV with one row per game per sportsbook per timestamp.

    Returns:
        Polars DataFrame with at least the required odds columns.

    Raises:
        ValueError: If required odds columns are missing.
    """
    odds = pl.read_csv(path)
    _validate_columns(odds, REQUIRED_ODDS_COLUMNS, "odds")
    return odds


def merge_games_and_odds(games: pl.DataFrame, odds: pl.DataFrame) -> pl.DataFrame:
    """Join odds rows to their games on ``game_id``.

    Preserves one row per game per sportsbook per timestamp (i.e. one output
    row per input odds row). Game columns that share a name with odds columns
    (``home_team``, ``away_team``) are kept from the odds frame as-is and from
    the games frame with a ``_games`` suffix, so downstream checks can verify
    consistency.

    Args:
        games: DataFrame with the required game columns.
        odds: DataFrame with the required odds columns.

    Returns:
        Merged DataFrame with one row per odds row.

    Raises:
        ValueError: If required columns are missing from either frame, or if
            any odds row has a ``game_id`` not present in ``games``.
    """
    _validate_columns(games, REQUIRED_GAME_COLUMNS, "games")
    _validate_columns(odds, REQUIRED_ODDS_COLUMNS, "odds")

    unmatched = odds.join(games.select("game_id"), on="game_id", how="anti")
    if unmatched.height > 0:
        orphan_ids = sorted(unmatched.get_column("game_id").unique().to_list())
        raise ValueError(f"Odds rows reference unknown game_ids: {orphan_ids}")

    return odds.join(games, on="game_id", how="inner", suffix="_games")
