"""Collect a TINY real historical odds sample from The Odds API (M8).

Connectivity / key / data-shape test only — NOT a full historical pull.
Requests two historical h2h snapshots for one game date that overlaps the
already-collected games table, limited to two bookmakers, saves raw JSON
untouched under ``data/raw/odds/the_odds_api/``, normalizes through the M4
path, marks opening/closing lines, and writes
``data/processed/odds_sample.parquet``.

Credit budget: 2 snapshot requests, 1 market, 2 bookmakers. The API key
comes only from the THE_ODDS_API_KEY environment variable and is never
printed. Never run from tests. Educational research data — no betting
claims of any kind.

Usage:
    THE_ODDS_API_KEY=... python scripts/collect_real_odds_sample.py [--overwrite]
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from src.data.collect_odds import (
    _HISTORICAL_ENDPOINT,
    mark_opening_and_closing_lines,
    normalize_the_odds_api_h2h_snapshot,
    save_odds_parquet,
    save_raw_odds_json,
)

GAMES_PATH = REPO_ROOT / "data" / "processed" / "nba_games.parquet"
RAW_DIR = REPO_ROOT / "data" / "raw" / "odds" / "the_odds_api"
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "odds_sample.parquet"

BOOKMAKERS = "draftkings,fanduel"
# Two snapshots on one game day: an afternoon (UTC) "early" line and a
# near-tip "late" line, so opening/closing marking has real movement.
SNAPSHOT_UTC_TIMES = ("18:00:00Z", "23:30:00Z")


def pick_sample_date() -> str:
    """Choose a busy game date from the collected games table."""
    if not GAMES_PATH.exists():
        raise SystemExit(
            "ERROR: data/processed/nba_games.parquet not found. Run "
            "scripts/collect_real_nba_games.py first."
        )
    games = pl.read_parquet(GAMES_PATH)
    print(
        f"games table date range: {games.get_column('game_date').min()} "
        f"to {games.get_column('game_date').max()}"
    )
    # Busiest date in the first collected season -> most events per credit.
    busiest = (
        games.group_by("game_date")
        .len()
        .sort(["len", "game_date"], descending=[True, False])
        .row(0, named=True)
    )
    print(f"sample date: {busiest['game_date']} ({busiest['len']} games)")
    return str(busiest["game_date"])


def fetch_snapshot(api_key: str, snapshot_iso: str) -> dict:
    """Request one historical h2h snapshot, limited to two bookmakers."""
    import requests

    response = requests.get(
        _HISTORICAL_ENDPOINT,
        params={
            "apiKey": api_key,
            "bookmakers": BOOKMAKERS,
            "markets": "h2h",
            "oddsFormat": "american",
            "dateFormat": "iso",
            "date": snapshot_iso,
        },
        timeout=30,
    )
    if not response.ok:
        # Never echo the request URL or params — the key is in them.
        raise SystemExit(
            f"ERROR: The Odds API returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )
    remaining = response.headers.get("x-requests-remaining")
    if remaining is not None:
        print(f"  (API credits remaining: {remaining})")
    return response.json()


def main() -> None:
    """Collect, normalize, and summarize the tiny odds sample."""
    overwrite = "--overwrite" in sys.argv
    if overwrite:
        print("OVERWRITE MODE: existing raw and processed files will be replaced.")

    api_key = os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        raise SystemExit(
            "ERROR: THE_ODDS_API_KEY is not set. Export it in your shell "
            "(never commit it): export THE_ODDS_API_KEY=..."
        )

    sample_date = pick_sample_date()
    requested = [f"{sample_date}T{t}" for t in SNAPSHOT_UTC_TIMES]

    frames: list[pl.DataFrame] = []
    raw_paths: list[Path] = []
    for snapshot_iso in requested:
        raw_path = RAW_DIR / f"h2h_{snapshot_iso.replace(':', '')}.json"
        if raw_path.exists() and not overwrite:
            print(f"loading saved raw snapshot {raw_path.name} (no API credit spent)")
            raw = json.loads(raw_path.read_text())
        else:
            print(f"requesting historical snapshot at {snapshot_iso} ...")
            raw = fetch_snapshot(api_key, snapshot_iso)
            save_raw_odds_json(raw, str(raw_path), overwrite=overwrite)
        raw_paths.append(raw_path)

        events = raw.get("data", [])
        if not events:
            raise SystemExit(
                f"ERROR: snapshot {snapshot_iso} contains no events. Try a "
                f"different date/time (check the season calendar), and "
                f"verify your plan includes historical data."
            )
        frames.append(
            normalize_the_odds_api_h2h_snapshot(raw, raw["timestamp"])
        )

    odds = mark_opening_and_closing_lines(pl.concat(frames))
    if odds.height == 0:
        raise SystemExit(
            "ERROR: snapshots normalized to zero odds rows (no h2h prices "
            "from the requested bookmakers). Try different bookmakers or times."
        )

    save_odds_parquet(odds, str(PROCESSED_PATH), overwrite=overwrite)

    books = sorted(odds.get_column("sportsbook").unique().to_list())
    timestamps = odds.get_column("timestamp")
    print("-" * 64)
    print("Real odds sample summary (tiny connectivity/shape test only)")
    print(f"requested snapshots:   {', '.join(requested)}")
    print("raw outputs:           " + ", ".join(
        str(p.relative_to(REPO_ROOT)) for p in raw_paths
    ))
    print(f"normalized odds rows:  {odds.height}")
    print(f"unique events:         {odds.get_column('provider_event_id').n_unique()}")
    print(f"sportsbooks found:     {', '.join(books)}")
    print(f"timestamp range:       {timestamps.min()} to {timestamps.max()}")
    print(f"opening line rows:     {odds.filter(pl.col('is_opening_line')).height}")
    print(f"closing line rows:     {odds.filter(pl.col('is_closing_line')).height}")
    print(f"processed output:      {PROCESSED_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
