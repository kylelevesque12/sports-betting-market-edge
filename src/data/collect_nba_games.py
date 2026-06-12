"""Real NBA game schedule/results ingestion (research plan milestone M3).

Source: ``nba_api`` LeagueGameLog, which returns one row per **team**-game.
``normalize_nba_games`` pairs those team rows into one row per game matching
the project schema, canonicalizes team names, and applies the M1 validation
utilities. Unit tests use mocked frames only — no network calls.

Raw API responses are saved untouched to ``data/raw/`` by collection
scripts before normalization (CLAUDE.md immutability rule); cleaned output
goes to ``data/processed/`` as Parquet.
"""

import time
from pathlib import Path

import polars as pl

from src.data.schema_validation import (
    parse_date_column,
    validate_binary_column,
    validate_required_columns,
)
from src.data.team_mapping import canonicalize_team_columns

# nba_api LeagueGameLog columns this module depends on.
REQUIRED_RAW_COLUMNS: tuple[str, ...] = (
    "GAME_ID",
    "SEASON_ID",
    "GAME_DATE",
    "TEAM_ABBREVIATION",
    "MATCHUP",
    "PTS",
)

CLEAN_GAMES_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "game_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_win",
)

_REQUEST_PAUSE_SECONDS = 1.5  # polite rate limiting between season requests


def _to_polars(raw_df) -> pl.DataFrame:
    """Accept a Polars or pandas DataFrame at the API boundary."""
    if isinstance(raw_df, pl.DataFrame):
        return raw_df
    if hasattr(raw_df, "to_dict"):  # pandas without requiring pyarrow
        return pl.DataFrame(raw_df.to_dict(orient="list"))
    raise ValueError(
        f"raw_df must be a Polars or pandas DataFrame, got {type(raw_df).__name__}."
    )


def _season_label(season_id: str) -> str:
    """Convert an nba_api SEASON_ID (e.g. ``'22023'``) to ``'2023-24'``.

    The leading digit is the season type (2 = regular season); the remainder
    is the season start year.
    """
    digits = str(season_id).strip()
    if len(digits) != 5 or not digits.isdigit():
        raise ValueError(
            f"unrecognized SEASON_ID {season_id!r}; expected 5 digits like '22023'."
        )
    start_year = int(digits[1:])
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def normalize_nba_games(raw_df) -> pl.DataFrame:
    """Pair raw team-game rows into the clean one-row-per-game schema.

    Args:
        raw_df: Raw LeagueGameLog-shaped data (Polars or pandas), one row
            per team per game, with home teams identified by ``'vs.'`` and
            away teams by ``'@'`` in ``MATCHUP``.

    Returns:
        Polars DataFrame with columns ``game_id, season, game_date,
        home_team, away_team, home_score, away_score, home_win``, sorted by
        ``game_date`` then ``game_id``.

    Raises:
        ValueError: If required raw columns are missing; game_id, date,
            team, or score values are null; teams are unknown to the
            mapping; or any game lacks exactly one home and one away row.
    """
    raw = _to_polars(raw_df)
    validate_required_columns(raw, list(REQUIRED_RAW_COLUMNS), "raw NBA games")

    for col in ("GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "PTS", "SEASON_ID"):
        null_count = raw.get_column(col).null_count()
        if null_count > 0:
            raise ValueError(f"raw NBA games column {col!r} contains "
                             f"{null_count} null value(s).")

    is_home = pl.col("MATCHUP").str.contains("vs.", literal=True)
    is_away = pl.col("MATCHUP").str.contains("@", literal=True)
    unparseable = raw.filter(~is_home & ~is_away)
    if unparseable.height > 0:
        bad = unparseable.get_column("MATCHUP").unique().to_list()[:5]
        raise ValueError(f"MATCHUP values not recognizable as home/away: {bad}")

    home_rows = raw.filter(is_home).select(
        pl.col("GAME_ID").alias("game_id"),
        pl.col("SEASON_ID").alias("season_id"),
        pl.col("GAME_DATE").alias("game_date"),
        pl.col("TEAM_ABBREVIATION").alias("home_team"),
        pl.col("PTS").alias("home_score"),
    )
    away_rows = raw.filter(is_away).select(
        pl.col("GAME_ID").alias("game_id"),
        pl.col("TEAM_ABBREVIATION").alias("away_team"),
        pl.col("PTS").alias("away_score"),
    )

    # Every game must have exactly one home and one away row.
    for name, side in (("home", home_rows), ("away", away_rows)):
        dupes = side.group_by("game_id").len().filter(pl.col("len") > 1)
        if dupes.height > 0:
            raise ValueError(
                f"games with multiple {name} rows: "
                f"{dupes.get_column('game_id').to_list()[:5]}"
            )
    unmatched = home_rows.join(away_rows, on="game_id", how="anti")
    unmatched_away = away_rows.join(home_rows, on="game_id", how="anti")
    if unmatched.height > 0 or unmatched_away.height > 0:
        bad = (
            unmatched.get_column("game_id").to_list()
            + unmatched_away.get_column("game_id").to_list()
        )[:5]
        raise ValueError(f"games missing a home or away row: {bad}")

    games = home_rows.join(away_rows, on="game_id", how="inner")

    season_labels = {
        sid: _season_label(sid)
        for sid in games.get_column("season_id").unique().to_list()
    }
    # NBA games cannot end tied; equal scores mean corrupted source data.
    ties = games.filter(pl.col("home_score") == pl.col("away_score"))
    if ties.height > 0:
        raise ValueError(
            f"games with tied scores (impossible in the NBA — corrupted "
            f"data): {ties.get_column('game_id').to_list()[:5]}"
        )

    games = games.with_columns(
        pl.col("season_id").replace_strict(season_labels).alias("season"),
        (pl.col("home_score") > pl.col("away_score")).cast(pl.Int64).alias("home_win"),
    ).drop("season_id")

    games = canonicalize_team_columns(games, ["home_team", "away_team"])
    games = parse_date_column(games, "game_date")
    validate_binary_column(games, "home_win")

    games = games.select(CLEAN_GAMES_COLUMNS).sort(["game_date", "game_id"])
    validate_required_columns(games, list(CLEAN_GAMES_COLUMNS), "clean games")
    return games


def save_games_parquet(
    games: pl.DataFrame,
    output_path: str,
    overwrite: bool = False,
) -> None:
    """Write clean games to Parquet, creating parent directories.

    Args:
        games: Clean games DataFrame.
        output_path: Destination ``.parquet`` path.
        overwrite: If False (default), refuse to replace an existing file.

    Raises:
        FileExistsError: If the file exists and ``overwrite`` is False.
    """
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass overwrite=True to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    games.write_parquet(path)


def collect_nba_games_for_seasons(seasons: list[str]) -> pl.DataFrame:
    """Fetch raw regular-season team-game logs from nba_api for each season.

    Live network function — never called from unit tests. Callers (scripts)
    are responsible for saving the returned raw frame to ``data/raw/``
    before normalizing.

    Args:
        seasons: Season strings in nba_api format, e.g. ``["2022-23"]``.

    Returns:
        Concatenated raw team-game rows (Polars) across the seasons.
    """
    from nba_api.stats.endpoints import leaguegamelog  # imported lazily

    frames: list[pl.DataFrame] = []
    for i, season in enumerate(seasons):
        if i > 0:
            time.sleep(_REQUEST_PAUSE_SECONDS)
        log = leaguegamelog.LeagueGameLog(
            season=season, season_type_all_star="Regular Season"
        )
        frames.append(_to_polars(log.get_data_frames()[0]))
    return pl.concat(frames)
