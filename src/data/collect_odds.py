"""Historical odds ingestion normalization (research plan milestone M4).

Provider for v1: The Odds API. The pre-game moneyline is its ``h2h`` market.
This module normalizes raw snapshot responses (live or saved JSON) into the
clean odds schema; it does NOT match provider events to internal NBA game
IDs — that is milestone M5, so ``game_id`` is null here by design.

No API keys in code (THE_ODDS_API_KEY environment variable only); unit
tests use mocked raw dictionaries and never call the network. See
docs/odds_ingestion.md.
"""

import json
import os
import time
from pathlib import Path

import polars as pl

from src.data.schema_validation import (
    parse_datetime_column,
    validate_american_odds_column,
    validate_required_columns,
)
from src.data.team_mapping import canonicalize_team_name

CLEAN_ODDS_COLUMNS: tuple[str, ...] = (
    "provider_event_id",
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

_HISTORICAL_ENDPOINT = (
    "https://api.the-odds-api.com/v4/historical/sports/basketball_nba/odds"
)
_REQUEST_PAUSE_SECONDS = 1.0


def _american_int(price, event_id: str, side: str) -> int:
    """Coerce a provider price to an integer American odds value."""
    try:
        as_float = float(price)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"event {event_id!r}: {side} price {price!r} is not numeric."
        ) from exc
    if as_float != int(as_float):
        raise ValueError(
            f"event {event_id!r}: {side} price {price!r} is not a whole-number "
            f"American odds value (is the snapshot in decimal format?)."
        )
    return int(as_float)


def normalize_the_odds_api_h2h_snapshot(
    raw_snapshot: dict,
    snapshot_timestamp: str,
    market: str = "h2h",
) -> pl.DataFrame:
    """Normalize one The Odds API historical snapshot into clean odds rows.

    Produces one row per event per sportsbook for the ``h2h`` (moneyline)
    market, stamped with the snapshot time. ``game_id`` is null until event
    matching (M5). ``is_opening_line``/``is_closing_line`` default to False;
    use :func:`mark_opening_and_closing_lines` across snapshots.

    Args:
        raw_snapshot: Parsed provider response. Historical responses wrap
            events in a ``"data"`` key; a dict with ``"data"`` is expected.
        snapshot_timestamp: ISO datetime the snapshot represents.
        market: Provider market key. v1 supports ``"h2h"`` only.

    Returns:
        Polars DataFrame with the clean odds schema.

    Raises:
        ValueError: If the snapshot shape is wrong, an event is missing
            teams, a bookmaker lacks the market, an outcome is missing for
            either team, odds are invalid, or the timestamp is malformed.
    """
    if not isinstance(raw_snapshot, dict) or "data" not in raw_snapshot:
        raise ValueError(
            "raw_snapshot must be a The Odds API historical response dict "
            "with a 'data' key containing events."
        )

    rows: list[dict] = []
    for event in raw_snapshot["data"]:
        event_id = event.get("id")
        if not event_id:
            raise ValueError(f"event is missing an id: {event!r}")

        raw_home = event.get("home_team")
        raw_away = event.get("away_team")
        if not raw_home or not raw_away:
            raise ValueError(f"event {event_id!r} is missing home/away teams.")
        home = canonicalize_team_name(raw_home)
        away = canonicalize_team_name(raw_away)

        for bookmaker in event.get("bookmakers", []):
            book = bookmaker.get("key") or bookmaker.get("title")
            if not book:
                raise ValueError(f"event {event_id!r}: bookmaker has no key/title.")

            h2h = next(
                (m for m in bookmaker.get("markets", []) if m.get("key") == market),
                None,
            )
            if h2h is None:
                raise ValueError(
                    f"event {event_id!r}, bookmaker {book!r}: market "
                    f"{market!r} is missing."
                )

            outcomes = h2h.get("outcomes", [])
            if len(outcomes) != 2:
                raise ValueError(
                    f"event {event_id!r}, bookmaker {book!r}: h2h market must "
                    f"have exactly 2 outcomes, got {len(outcomes)}."
                )

            prices: dict[str, object] = {}
            for outcome in outcomes:
                name = outcome.get("name")
                if name:
                    prices[canonicalize_team_name(name)] = outcome.get("price")
            if home not in prices:
                raise ValueError(
                    f"event {event_id!r}, bookmaker {book!r}: no h2h outcome "
                    f"for home team {home!r}."
                )
            if away not in prices:
                raise ValueError(
                    f"event {event_id!r}, bookmaker {book!r}: no h2h outcome "
                    f"for away team {away!r}."
                )

            rows.append(
                {
                    "provider_event_id": event_id,
                    "game_id": None,
                    "sportsbook": book,
                    "market": market,
                    "timestamp": snapshot_timestamp,
                    "home_team": home,
                    "away_team": away,
                    "home_american_odds": _american_int(prices[home], event_id, "home"),
                    "away_american_odds": _american_int(prices[away], event_id, "away"),
                    "is_opening_line": False,
                    "is_closing_line": False,
                }
            )

    odds = pl.DataFrame(
        rows,
        schema={
            "provider_event_id": pl.String,
            "game_id": pl.String,
            "sportsbook": pl.String,
            "market": pl.String,
            "timestamp": pl.String,
            "home_team": pl.String,
            "away_team": pl.String,
            "home_american_odds": pl.Int64,
            "away_american_odds": pl.Int64,
            "is_opening_line": pl.Boolean,
            "is_closing_line": pl.Boolean,
        },
    )
    odds = parse_datetime_column(odds, "timestamp")
    if odds.height > 0:
        validate_american_odds_column(odds, "home_american_odds")
        validate_american_odds_column(odds, "away_american_odds")
    return odds.select(CLEAN_ODDS_COLUMNS)


def mark_opening_and_closing_lines(odds: pl.DataFrame) -> pl.DataFrame:
    """Mark earliest/latest snapshot per event-book-market as opening/closing.

    Args:
        odds: Clean odds rows from one or more snapshots.

    Returns:
        Same rows with ``is_opening_line`` True on each group's earliest
        timestamp and ``is_closing_line`` True on its latest. A group with a
        single snapshot is both opening and closing.

    Raises:
        ValueError: If required columns are missing or ``timestamp`` has
            nulls.
    """
    validate_required_columns(
        odds,
        ["provider_event_id", "sportsbook", "market", "timestamp"],
        "odds",
    )
    null_count = odds.get_column("timestamp").null_count()
    if null_count > 0:
        raise ValueError(f"column 'timestamp' contains {null_count} null value(s).")

    group = ["provider_event_id", "sportsbook", "market"]
    return odds.with_columns(
        (pl.col("timestamp") == pl.col("timestamp").min().over(group)).alias(
            "is_opening_line"
        ),
        (pl.col("timestamp") == pl.col("timestamp").max().over(group)).alias(
            "is_closing_line"
        ),
    )


def _safe_write(path_str: str, overwrite: bool) -> Path:
    path = Path(path_str)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path_str} already exists; pass overwrite=True to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_raw_odds_json(
    raw_snapshot: dict,
    output_path: str,
    overwrite: bool = False,
) -> None:
    """Save a raw provider response untouched as JSON (CLAUDE.md raw rule).

    Raises:
        FileExistsError: If the file exists and ``overwrite`` is False.
    """
    path = _safe_write(output_path, overwrite)
    path.write_text(json.dumps(raw_snapshot, indent=2))


def save_odds_parquet(
    odds: pl.DataFrame,
    output_path: str,
    overwrite: bool = False,
) -> None:
    """Save normalized odds as Parquet, creating parent directories.

    Raises:
        FileExistsError: If the file exists and ``overwrite`` is False.
    """
    path = _safe_write(output_path, overwrite)
    odds.write_parquet(path)


def fetch_the_odds_api_historical_snapshot(snapshot_iso_datetime: str) -> dict:
    """Fetch one historical NBA h2h snapshot from The Odds API.

    Live network function — never called from unit tests. The API key comes
    from the ``THE_ODDS_API_KEY`` environment variable; it is never
    hard-coded. Historical snapshots cost paid quota — callers should save
    every response via :func:`save_raw_odds_json` immediately.

    Args:
        snapshot_iso_datetime: Point-in-time to query, ISO format (UTC),
            e.g. ``"2023-10-24T22:45:00Z"``.

    Returns:
        Parsed JSON response (``timestamp``, ``previous_timestamp``,
        ``next_timestamp``, ``data`` keys).

    Raises:
        KeyError: If THE_ODDS_API_KEY is not set.
        requests.HTTPError: On API errors.
    """
    import requests  # imported lazily; justified in docs/tech_stack.md

    api_key = os.environ["THE_ODDS_API_KEY"]
    time.sleep(_REQUEST_PAUSE_SECONDS)
    response = requests.get(
        _HISTORICAL_ENDPOINT,
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "american",
            "date": snapshot_iso_datetime,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()
