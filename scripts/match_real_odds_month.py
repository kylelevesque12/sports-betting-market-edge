"""Match the one-month real odds sample to real NBA games.

Same pattern as scripts/match_real_odds_sample.py at month scale: an
explicit, counted, printed scope filter (snapshots contain every upcoming
event at capture time, including games outside the table's coverage),
followed by the strict M5 matcher — exact event date + canonical teams, no
fuzzy matching, no silent drops, snapshot timestamp never used as game
date. Data integration step only: no models, no evaluation, no backtests.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from src.data.match_events import (
    derive_event_date_from_odds,
    match_odds_to_games,
    validate_matched_odds,
)
from src.data.schema_validation import parse_date_column
from src.data.team_mapping import canonicalize_team_columns

GAMES_PATH = REPO_ROOT / "data" / "processed" / "nba_games.parquet"
ODDS_PATH = REPO_ROOT / "data" / "processed" / "odds_month_sample.parquet"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "matched_odds_month_sample.parquet"


def exclude_out_of_scope_events(
    odds: pl.DataFrame, games: pl.DataFrame
) -> tuple[pl.DataFrame, int, int]:
    """Explicitly exclude odds events outside the games table's key space.

    Returns (in-scope odds in original schema, excluded event count,
    excluded row count). Every exclusion is printed — never silent.
    """
    odds_keyed = derive_event_date_from_odds(
        canonicalize_team_columns(odds, ["home_team", "away_team"])
    )
    game_keys = parse_date_column(
        canonicalize_team_columns(games, ["home_team", "away_team"]), "game_date"
    ).select(
        pl.col("game_date").alias("event_date"), "home_team", "away_team"
    ).unique()

    in_scope = odds_keyed.join(
        game_keys, on=["event_date", "home_team", "away_team"], how="semi"
    )
    out_of_scope = odds_keyed.join(
        game_keys, on=["event_date", "home_team", "away_team"], how="anti"
    )

    excluded_events = 0
    if out_of_scope.height > 0:
        excluded = (
            out_of_scope.select(
                "provider_event_id", "event_date", "home_team", "away_team"
            )
            .unique()
            .sort("event_date")
        )
        excluded_events = excluded.height
        print("\nSCOPE EXCLUSIONS (reason: outside current games table scope):")
        print(f"  provider events excluded: {excluded_events}")
        print(f"  odds rows excluded:       {out_of_scope.height}")
        with pl.Config(tbl_rows=100, fmt_str_lengths=40):
            print(excluded)
        print()

    filtered = odds.join(
        in_scope.select("provider_event_id", "sportsbook", "timestamp").unique(),
        on=["provider_event_id", "sportsbook", "timestamp"],
        how="semi",
    )
    return filtered, excluded_events, out_of_scope.height


def print_unmatched_diagnostics(odds: pl.DataFrame, games: pl.DataFrame) -> None:
    """Show which in-scope odds events found no game, plus nearby games."""
    odds_keyed = derive_event_date_from_odds(
        canonicalize_team_columns(odds, ["home_team", "away_team"])
    )
    games_keyed = parse_date_column(
        canonicalize_team_columns(games, ["home_team", "away_team"]), "game_date"
    )
    game_keys = games_keyed.select(
        pl.col("game_date").alias("event_date"), "home_team", "away_team"
    ).with_columns(pl.lit(True).alias("_found"))

    unmatched = (
        odds_keyed.join(
            game_keys, on=["event_date", "home_team", "away_team"], how="left"
        )
        .filter(pl.col("_found").is_null())
        .select(
            "provider_event_id", "commence_time", "event_date",
            "home_team", "away_team",
        )
        .unique()
        .sort("event_date")
    )
    print("\nUNMATCHED IN-SCOPE ODDS EVENTS:")
    with pl.Config(tbl_rows=100, fmt_str_lengths=40):
        print(unmatched)


def main() -> None:
    """Scope-filter, strictly match, validate, and save the month sample."""
    overwrite = "--overwrite" in sys.argv
    if OUTPUT_PATH.exists() and not overwrite:
        raise SystemExit(
            f"ERROR: {OUTPUT_PATH.relative_to(REPO_ROOT)} already exists; "
            f"pass --overwrite to replace it."
        )
    for path, hint in (
        (GAMES_PATH, "scripts/collect_real_nba_games.py"),
        (ODDS_PATH, "scripts/collect_real_odds_month.py --confirm"),
    ):
        if not path.exists():
            raise SystemExit(
                f"ERROR: {path.relative_to(REPO_ROOT)} not found. Run {hint} first."
            )

    games = pl.read_parquet(GAMES_PATH)
    odds = pl.read_parquet(ODDS_PATH)

    print("Match one-month real odds to real games")
    print("-" * 66)
    print(f"games rows:             {games.height}")
    print(f"odds rows:              {odds.height}")
    print(f"unique provider events: {odds.get_column('provider_event_id').n_unique()}")
    print(
        f"commence_time range:    {odds.get_column('commence_time').min()} "
        f"to {odds.get_column('commence_time').max()}"
    )
    print(f"sportsbooks:            "
          f"{sorted(odds.get_column('sportsbook').unique().to_list())}")
    print(f"opening line rows:      "
          f"{odds.filter(pl.col('is_opening_line')).height}")
    print(f"closing line rows:      "
          f"{odds.filter(pl.col('is_closing_line')).height}")

    odds_in_scope, excluded_events, excluded_rows = exclude_out_of_scope_events(
        odds, games
    )

    try:
        matched = match_odds_to_games(odds_in_scope, games)
    except ValueError as exc:
        print(f"\nSTRICT MATCH FAILED: {exc}")
        print_unmatched_diagnostics(odds_in_scope, games)
        raise

    validate_matched_odds(matched)

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()  # only reachable with --overwrite
    matched.write_parquet(OUTPUT_PATH)

    dupes = (
        matched.group_by(["game_id", "sportsbook", "market", "timestamp"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    print("-" * 66)
    print("matched output summary")
    print(f"matched rows:             {matched.height}")
    print(f"unique game_ids:          {matched.get_column('game_id').n_unique()}")
    print(f"unique provider events:   "
          f"{matched.get_column('provider_event_id').n_unique()}")
    print(f"scope-excluded events:    {excluded_events}")
    print(f"scope-excluded rows:      {excluded_rows}")
    print(f"unmatched after filter:   0 (strict matching: failures raise)")
    print(f"duplicate match keys:     {dupes}")
    print(f"opening line rows:        "
          f"{matched.filter(pl.col('is_opening_line')).height}")
    print(f"closing line rows:        "
          f"{matched.filter(pl.col('is_closing_line')).height}")
    print(f"output:                   {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
