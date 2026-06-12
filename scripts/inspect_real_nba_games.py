"""Inspect the processed real NBA games table before odds ingestion.

Read-only sanity report on ``data/processed/nba_games.parquet``: counts,
coverage, distributions, and hard validation of the schema invariants the
odds-matching milestone will depend on. Modifies nothing.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from src.data.collect_nba_games import CLEAN_GAMES_COLUMNS
from src.data.schema_validation import (
    validate_binary_column,
    validate_required_columns,
)
from src.data.team_mapping import CANONICAL_TEAM_IDS

GAMES_PATH = REPO_ROOT / "data" / "processed" / "nba_games.parquet"


def validate_games(games: pl.DataFrame) -> None:
    """Hard checks; each failure raises with a clear message."""
    validate_required_columns(games, list(CLEAN_GAMES_COLUMNS), "processed games")

    if games.get_column("game_id").null_count() > 0:
        raise ValueError("game_id contains null values.")

    duplicates = games.group_by("game_id").len().filter(pl.col("len") > 1)
    if duplicates.height > 0:
        raise ValueError(
            f"game_id is not unique; duplicates: "
            f"{duplicates.get_column('game_id').to_list()[:5]}"
        )

    if games.schema["game_date"] != pl.Date:
        raise ValueError(
            f"game_date must be pl.Date, got {games.schema['game_date']}"
        )

    for col in ("home_team", "away_team"):
        bad = games.filter(~pl.col(col).is_in(list(CANONICAL_TEAM_IDS)))
        if bad.height > 0:
            raise ValueError(
                f"{col} contains non-canonical teams: "
                f"{bad.get_column(col).unique().to_list()}"
            )

    for col in ("home_score", "away_score"):
        if games.get_column(col).null_count() > 0:
            raise ValueError(f"{col} contains null values.")

    validate_binary_column(games, "home_win")

    self_games = games.filter(pl.col("home_team") == pl.col("away_team"))
    if self_games.height > 0:
        raise ValueError(
            f"games where home_team equals away_team: "
            f"{self_games.get_column('game_id').to_list()[:5]}"
        )

    teams_present = set(games.get_column("home_team").to_list()) | set(
        games.get_column("away_team").to_list()
    )
    missing_teams = sorted(set(CANONICAL_TEAM_IDS) - teams_present)
    if missing_teams:
        raise ValueError(
            f"complete seasons should include all 30 NBA teams; "
            f"missing: {missing_teams}"
        )


def main() -> None:
    """Print the inspection report and run validation."""
    if not GAMES_PATH.exists():
        raise SystemExit(
            f"ERROR: {GAMES_PATH.relative_to(REPO_ROOT)} not found. "
            "Run scripts/collect_real_nba_games.py first."
        )

    games = pl.read_parquet(GAMES_PATH)

    seasons = sorted(games.get_column("season").unique().to_list())
    teams = sorted(
        set(games.get_column("home_team").to_list())
        | set(games.get_column("away_team").to_list())
    )
    duplicate_count = (
        games.group_by("game_id").len().filter(pl.col("len") > 1).height
    )
    null_counts = {
        col: games.get_column(col).null_count() for col in CLEAN_GAMES_COLUMNS
    }
    home_wins = int(games.get_column("home_win").sum())
    home_win_pct = home_wins / games.height

    print("Real NBA games inspection")
    print("=" * 64)
    print(f"total games:          {games.height}")
    print(f"seasons included:     {', '.join(seasons)}")
    print(
        f"date range:           {games.get_column('game_date').min()} "
        f"to {games.get_column('game_date').max()}"
    )
    print(f"unique teams:         {len(teams)}")
    print(f"teams found:          {', '.join(teams)}")
    print(f"duplicate game_ids:   {duplicate_count}")
    print("null counts:          " + ", ".join(
        f"{col}={n}" for col, n in null_counts.items() if n > 0
    ) + ("(none)" if not any(null_counts.values()) else ""))
    print(
        f"home_win distribution: {home_wins} home wins / "
        f"{games.height - home_wins} away wins "
        f"(home win rate {home_win_pct:.3f})"
    )

    validate_games(games)
    print("-" * 64)
    print("all validation checks passed")


if __name__ == "__main__":
    main()
