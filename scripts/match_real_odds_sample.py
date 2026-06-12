"""Match the tiny real odds sample to real NBA games (M9 diagnostic).

Joins data/processed/odds_sample.parquet to data/processed/nba_games.parquet
via the strict M5 matcher (exact event-date + canonical teams, no fuzzy
matching, no silent drops, snapshot timestamp never used as game date).

A real provider snapshot contains every upcoming event at capture time —
possibly spanning several days and including games outside the games table
(e.g. play-in games when the table is regular-season only). If strict
matching fails, this script prints a diagnostic table of the unmatched
events plus nearby games, then re-raises — failure with a useful diagnosis
is a successful outcome for this milestone.

Integration test only: no models, no backtests, no betting claims.
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
ODDS_PATH = REPO_ROOT / "data" / "processed" / "odds_sample.parquet"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "matched_odds_sample.parquet"


def exclude_out_of_scope_events(
    odds: pl.DataFrame, games: pl.DataFrame
) -> pl.DataFrame:
    """Explicitly exclude odds events outside the games table's scope.

    A historical snapshot contains every upcoming event at capture time —
    including games beyond the collected table's coverage (e.g. play-in
    games when the table is regular-season only). Those events cannot match
    by definition, so they are removed here as an explicit, counted,
    printed exclusion BEFORE strict matching. This is a scope filter, not a
    weakening of the matcher: everything that survives must still match
    exactly one game or the strict matcher raises.
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

    if out_of_scope.height > 0:
        excluded_events = (
            out_of_scope.select(
                "provider_event_id", "event_date", "home_team", "away_team"
            )
            .unique()
            .sort("event_date")
        )
        print("\nSCOPE EXCLUSIONS (outside current games table scope):")
        print(f"  provider events excluded: {excluded_events.height}")
        print(f"  odds rows excluded:       {out_of_scope.height}")
        with pl.Config(tbl_rows=50, fmt_str_lengths=40):
            print(excluded_events)
        print("  reason: outside current games table scope (games table is "
              "regular-season only)\n")

    # Return original-schema rows (drop the derived event_date; the strict
    # matcher re-derives it from commence_time as usual).
    return odds.join(
        in_scope.select("provider_event_id", "sportsbook", "timestamp").unique(),
        on=["provider_event_id", "sportsbook", "timestamp"],
        how="semi",
    )


def print_unmatched_diagnostics(odds: pl.DataFrame, games: pl.DataFrame) -> None:
    """Show which odds events found no game, and what games were nearby."""
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

    print("\nUNMATCHED ODDS EVENTS:")
    with pl.Config(tbl_rows=50, fmt_str_lengths=40):
        print(unmatched)

    if unmatched.height > 0:
        lo = unmatched.get_column("event_date").min()
        hi = unmatched.get_column("event_date").max()
        nearby = games_keyed.filter(
            (pl.col("game_date") >= lo - pl.duration(days=1))
            & (pl.col("game_date") <= hi + pl.duration(days=1))
        ).select("game_id", "game_date", "home_team", "away_team")
        print("\nGAMES TABLE ROWS ON SAME/ADJACENT DATES:")
        with pl.Config(tbl_rows=50):
            print(nearby.sort("game_date"))
        print(
            f"\ngames table coverage: "
            f"{games_keyed.get_column('game_date').min()} to "
            f"{games_keyed.get_column('game_date').max()} "
            f"(regular season only — play-in/playoff events in a snapshot "
            f"will not match by design)"
        )


def main() -> None:
    """Match the odds sample to games; diagnose loudly if strict match fails."""
    overwrite = "--overwrite" in sys.argv
    if OUTPUT_PATH.exists() and not overwrite:
        raise SystemExit(
            f"ERROR: {OUTPUT_PATH.relative_to(REPO_ROOT)} already exists; "
            f"pass --overwrite to replace it."
        )

    games = pl.read_parquet(GAMES_PATH)
    odds = pl.read_parquet(ODDS_PATH)

    print("Match real odds sample to real games (diagnostic integration step)")
    print("-" * 66)
    print(f"games rows:            {games.height}")
    print(f"odds rows:             {odds.height}")
    print(f"unique provider events: {odds.get_column('provider_event_id').n_unique()}")
    print(
        f"commence_time range:   {odds.get_column('commence_time').min()} "
        f"to {odds.get_column('commence_time').max()}"
    )
    print(f"sportsbooks:           "
          f"{sorted(odds.get_column('sportsbook').unique().to_list())}")

    odds_in_scope = exclude_out_of_scope_events(odds, games)

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
    print(f"matched rows:            {matched.height}")
    print(f"unique game_ids:         {matched.get_column('game_id').n_unique()}")
    print(f"unique provider events:  "
          f"{matched.get_column('provider_event_id').n_unique()}")
    print(
        f"scope-excluded rows:     {odds.height - odds_in_scope.height} "
        f"(reported above; never silent)"
    )
    print(f"unmatched events:        0 (strict matching: failures raise)")
    print(f"opening line rows:       "
          f"{matched.filter(pl.col('is_opening_line')).height}")
    print(f"closing line rows:       "
          f"{matched.filter(pl.col('is_closing_line')).height}")
    print(f"duplicate match keys:    {dupes}")
    print(f"output:                  {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
