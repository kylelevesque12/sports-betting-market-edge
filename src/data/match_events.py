"""Match normalized odds rows to internal NBA game IDs (milestone M5).

Provider event IDs are not internal game IDs. Matching uses what the two
datasets actually share: the event's start date and the canonicalized
home/away teams. The odds snapshot ``timestamp`` is when a price was
captured, not when the game is played (a Tuesday-night game's opening line
is often captured Monday), so it is never used as the event date —
``event_date``, ``commence_time``, or ``event_datetime`` is required.

Matching is exact and strict: every odds row must match exactly one game,
and failures raise rather than silently dropping rows. See
docs/event_matching.md.
"""

import polars as pl

from src.data.schema_validation import (
    parse_date_column,
    parse_datetime_column,
    validate_american_odds_column,
    validate_required_columns,
)
from src.data.team_mapping import canonicalize_team_columns

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
    "provider_event_id",
    "sportsbook",
    "market",
    "timestamp",
    "home_team",
    "away_team",
    "home_american_odds",
    "away_american_odds",
)

ODDS_DUPLICATE_KEY: tuple[str, ...] = (
    "provider_event_id",
    "sportsbook",
    "market",
    "timestamp",
)

MATCHED_DUPLICATE_KEY: tuple[str, ...] = (
    "game_id",
    "sportsbook",
    "market",
    "timestamp",
)


def derive_event_date_from_odds(
    odds: pl.DataFrame,
    event_timezone: str = "America/New_York",
) -> pl.DataFrame:
    """Return ``odds`` with a normalized ``event_date`` (Polars ``Date``).

    Sources, in priority order: an existing ``event_date`` column, the date
    part of ``commence_time``, or the date part of ``event_datetime``.

    Timezone rule: providers commonly report event start times in UTC, but
    NBA schedule dates follow the US convention — a late tip reported as
    e.g. ``2025-11-03T00:30:00Z`` is an evening game on **Nov 2** in
    ``America/New_York``. Timezone-aware datetimes are therefore converted
    to ``event_timezone`` before the date is taken; timezone-naive
    datetimes are treated as already being project-local time. The source
    datetime column is preserved unchanged alongside the derived date.

    Args:
        odds: Normalized odds rows.
        event_timezone: IANA timezone for the NBA schedule date convention.
            Defaults to ``America/New_York``.

    Returns:
        ``odds`` with ``event_date`` parsed/derived as ``pl.Date``.

    Raises:
        ValueError: If none of the event-date columns is present (the
            snapshot ``timestamp`` records when odds were captured, not
            when the game starts, and is never used), or if
            ``event_timezone`` is not a valid timezone name.
    """
    if "event_date" in odds.columns:
        return parse_date_column(odds, "event_date")
    for source in ("commence_time", "event_datetime"):
        if source in odds.columns:
            parsed = parse_datetime_column(odds, source)
            source_dtype = parsed.schema[source]
            try:
                date_expr = pl.col(source)
                if getattr(source_dtype, "time_zone", None) is not None:
                    date_expr = date_expr.dt.convert_time_zone(event_timezone)
                return parsed.with_columns(
                    date_expr.dt.date().alias("event_date")
                )
            except pl.exceptions.PolarsError as exc:
                raise ValueError(
                    f"could not derive event_date from {source!r} with "
                    f"timezone {event_timezone!r}: {exc}"
                ) from exc
    raise ValueError(
        "odds has no event_date, commence_time, or event_datetime column. "
        "An event start date is required for matching; the snapshot "
        "'timestamp' records when odds were captured, not when the game is "
        "played, and cannot be used as the event date."
    )


def _check_duplicates(
    df: pl.DataFrame, key: tuple[str, ...], frame_name: str
) -> None:
    dupes = df.group_by(list(key)).len().filter(pl.col("len") > 1)
    if dupes.height > 0:
        examples = dupes.select(list(key)).head(3).to_dicts()
        raise ValueError(
            f"{frame_name} contains {dupes.height} duplicate key(s) for "
            f"{list(key)}; first examples: {examples}"
        )


def match_odds_to_games(
    odds: pl.DataFrame,
    games: pl.DataFrame,
) -> pl.DataFrame:
    """Attach internal ``game_id`` to odds rows by date + canonical teams.

    Match keys: ``event_date == game_date``, canonical ``home_team``, and
    canonical ``away_team``. Exact and strict — no fuzzy matching, no
    silent drops.

    Args:
        odds: Normalized odds with an event-date column (see
            :func:`derive_event_date_from_odds`).
        games: Clean games table (M3 schema).

    Returns:
        All odds rows with ``game_id`` populated from the matched game and
        ``provider_event_id`` preserved.

    Raises:
        ValueError: If required columns are missing; duplicate odds rows
            exist for (provider_event_id, sportsbook, market, timestamp);
            any odds row matches no game; or any odds row matches more
            than one game.
    """
    validate_required_columns(games, list(REQUIRED_GAME_COLUMNS), "games")
    validate_required_columns(odds, list(REQUIRED_ODDS_COLUMNS), "odds")

    _check_duplicates(odds, ODDS_DUPLICATE_KEY, "odds")

    games_clean = canonicalize_team_columns(games, ["home_team", "away_team"])
    games_clean = parse_date_column(games_clean, "game_date")

    odds_clean = canonicalize_team_columns(odds, ["home_team", "away_team"])
    odds_clean = derive_event_date_from_odds(odds_clean)

    game_keys = games_clean.select(
        pl.col("game_id").alias("_matched_game_id"),
        pl.col("game_date").alias("event_date"),
        "home_team",
        "away_team",
    )

    matched = (
        odds_clean.with_row_index("_odds_row")
        .join(game_keys, on=["event_date", "home_team", "away_team"], how="left")
    )

    multi = matched.group_by("_odds_row").len().filter(pl.col("len") > 1)
    if multi.height > 0:
        bad_rows = matched.join(multi.select("_odds_row"), on="_odds_row").select(
            "provider_event_id", "event_date", "home_team", "away_team"
        )
        raise ValueError(
            f"{multi.height} odds row(s) match multiple games (duplicate "
            f"date/team keys in games?): {bad_rows.head(3).to_dicts()}"
        )

    unmatched = matched.filter(pl.col("_matched_game_id").is_null())
    if unmatched.height > 0:
        examples = unmatched.select(
            "provider_event_id", "event_date", "home_team", "away_team"
        ).head(5).to_dicts()
        raise ValueError(
            f"{unmatched.height} odds row(s) failed to match any game; "
            f"first examples: {examples}"
        )

    result = matched.with_columns(
        pl.col("_matched_game_id").alias("game_id")
    ).drop("_matched_game_id", "_odds_row")
    return result


def validate_matched_odds(matched_odds: pl.DataFrame) -> None:
    """Verify matched odds are ready for real market feature creation.

    Args:
        matched_odds: Output of :func:`match_odds_to_games`.

    Raises:
        ValueError: If game_id, provider_event_id, timestamp, or team
            columns contain nulls; odds fail American-odds validation; or
            duplicate (game_id, sportsbook, market, timestamp) rows exist.
    """
    validate_required_columns(
        matched_odds,
        ["game_id", *REQUIRED_ODDS_COLUMNS],
        "matched odds",
    )

    for col in ("game_id", "provider_event_id", "timestamp", "home_team", "away_team"):
        null_count = matched_odds.get_column(col).null_count()
        if null_count > 0:
            raise ValueError(
                f"matched odds column {col!r} contains {null_count} null value(s)."
            )

    validate_american_odds_column(matched_odds, "home_american_odds")
    validate_american_odds_column(matched_odds, "away_american_odds")

    _check_duplicates(matched_odds, MATCHED_DUPLICATE_KEY, "matched odds")
