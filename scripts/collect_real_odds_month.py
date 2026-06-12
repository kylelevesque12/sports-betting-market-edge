"""Collect one month of real historical NBA moneyline odds (credit-conscious).

First scaled odds collection: two h2h snapshots per game date (early
18:00Z, near-tip 23:30Z) for one month of regular-season dates taken from
the processed games table, limited to two bookmakers.

Safety model:
- Default run is a DRY PLAN: prints date range, snapshot count, and the
  estimated credit cost, then exits without any network call.
- ``--confirm`` performs live requests, but only for snapshots whose raw
  JSON is not already saved — reruns are resume-safe and spend nothing on
  existing data.
- ``--confirm --overwrite`` refetches everything and replaces outputs.
- THE_ODDS_API_KEY comes from the environment only and is never printed.

Collection only: no matching, features, evaluation, or backtesting here.
Educational research data — no betting claims of any kind.
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
RAW_DIR = REPO_ROOT / "data" / "raw" / "odds" / "the_odds_api" / "month_sample"
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "odds_month_sample.parquet"

BOOKMAKERS = "draftkings,fanduel"
SNAPSHOT_UTC_TIMES = ("18:00:00Z", "23:30:00Z")
# Mid-season month, safely away from opening week, All-Star logistics, and
# the play-in/playoff boundary. Only dates with games in the table are used.
MONTH_START = "2024-01-01"
MONTH_END = "2024-01-31"
CREDITS_PER_SNAPSHOT = 10  # 1 market x <=10 bookmakers (The Odds API pricing)


def plan_snapshots() -> list[str]:
    """Build planned snapshot timestamps from real game dates in the month."""
    if not GAMES_PATH.exists():
        raise SystemExit(
            "ERROR: data/processed/nba_games.parquet not found. Run "
            "scripts/collect_real_nba_games.py first."
        )
    games = pl.read_parquet(GAMES_PATH)
    month_dates = sorted(
        games.filter(
            (pl.col("game_date") >= pl.lit(MONTH_START).str.to_date())
            & (pl.col("game_date") <= pl.lit(MONTH_END).str.to_date())
        )
        .get_column("game_date")
        .unique()
        .to_list()
    )
    if not month_dates:
        raise SystemExit(
            f"ERROR: no games found between {MONTH_START} and {MONTH_END} "
            f"in the games table."
        )
    return [f"{d}T{t}" for d in month_dates for t in SNAPSHOT_UTC_TIMES]


def fetch_snapshot(api_key: str, snapshot_iso: str) -> tuple[dict, str | None]:
    """Request one historical h2h snapshot; return (response, credits left)."""
    import requests

    response = requests.get(
        _HISTORICAL_ENDPOINT,
        params={
            "apiKey": api_key,
            "regions": "us",
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
            f"ERROR: The Odds API returned HTTP {response.status_code} for "
            f"snapshot {snapshot_iso}: {response.text[:300]}"
        )
    return response.json(), response.headers.get("x-requests-remaining")


def main() -> None:
    """Plan, optionally collect, normalize, and summarize one month of odds."""
    confirm = "--confirm" in sys.argv
    overwrite = "--overwrite" in sys.argv
    if overwrite:
        print("OVERWRITE MODE: existing raw and processed files will be replaced.")

    planned = plan_snapshots()
    game_dates = len(planned) // len(SNAPSHOT_UTC_TIMES)
    estimated_credits = len(planned) * CREDITS_PER_SNAPSHOT

    print("One-month odds collection plan")
    print("-" * 64)
    print(f"date range:           {MONTH_START} to {MONTH_END}")
    print(f"game dates:           {game_dates}")
    print(f"snapshot timestamps:  {len(planned)} "
          f"({len(SNAPSHOT_UTC_TIMES)} per game date)")
    print(f"estimated credits:    {estimated_credits} "
          f"(~{CREDITS_PER_SNAPSHOT}/snapshot; existing raw files cost 0)")
    print(f"bookmakers:           {BOOKMAKERS}")
    print(f"raw output dir:       {RAW_DIR.relative_to(REPO_ROOT)}")
    print(f"processed output:     {PROCESSED_PATH.relative_to(REPO_ROOT)}")

    if not confirm:
        print("-" * 64)
        print("DRY PLAN ONLY — no requests made. Re-run with --confirm to "
              "collect missing snapshots.")
        return

    api_key = None
    frames: list[pl.DataFrame] = []
    fetched, skipped = 0, 0
    credits_remaining: str | None = None

    for snapshot_iso in planned:
        raw_path = RAW_DIR / f"h2h_{snapshot_iso.replace(':', '')}.json"
        if raw_path.exists() and not overwrite:
            raw = json.loads(raw_path.read_text())
            skipped += 1
        else:
            if api_key is None:
                api_key = os.environ.get("THE_ODDS_API_KEY")
                if not api_key:
                    raise SystemExit(
                        "ERROR: THE_ODDS_API_KEY is not set. Export it in "
                        "your shell (never commit it)."
                    )
            print(f"fetching {snapshot_iso} ...")
            raw, credits_remaining = fetch_snapshot(api_key, snapshot_iso)
            save_raw_odds_json(raw, str(raw_path), overwrite=overwrite)
            fetched += 1
            if credits_remaining is not None:
                print(f"  (credits remaining: {credits_remaining})")

        if not raw.get("data"):
            raise SystemExit(
                f"ERROR: snapshot {snapshot_iso} contains no events — "
                f"malformed or empty provider response; investigate before "
                f"continuing."
            )
        frames.append(normalize_the_odds_api_h2h_snapshot(raw, raw["timestamp"]))

    odds = mark_opening_and_closing_lines(pl.concat(frames))
    if odds.height == 0:
        raise SystemExit("ERROR: snapshots normalized to zero odds rows.")
    books = sorted(odds.get_column("sportsbook").unique().to_list())
    if not books:
        raise SystemExit("ERROR: no sportsbooks found in normalized odds.")
    opening_rows = odds.filter(pl.col("is_opening_line")).height
    closing_rows = odds.filter(pl.col("is_closing_line")).height
    if opening_rows == 0 or closing_rows == 0:
        raise SystemExit("ERROR: opening/closing line marking produced no rows.")

    save_odds_parquet(odds, str(PROCESSED_PATH), overwrite=overwrite)

    timestamps = odds.get_column("timestamp")
    print("-" * 64)
    print("One-month odds collection summary")
    print(f"requested snapshots:   {len(planned)}")
    print(f"skipped (existing):    {skipped}")
    print(f"fetched (live):        {fetched}")
    print(f"normalized odds rows:  {odds.height}")
    print(f"unique provider events: {odds.get_column('provider_event_id').n_unique()}")
    print(f"sportsbooks found:     {', '.join(books)}")
    print(f"timestamp range:       {timestamps.min()} to {timestamps.max()}")
    print(f"opening line rows:     {opening_rows}")
    print(f"closing line rows:     {closing_rows}")
    print(f"processed output:      {PROCESSED_PATH.relative_to(REPO_ROOT)}")
    if credits_remaining is not None:
        print(f"API credits remaining: {credits_remaining}")


if __name__ == "__main__":
    main()
