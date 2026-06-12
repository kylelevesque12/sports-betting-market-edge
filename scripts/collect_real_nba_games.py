"""Collect real NBA game results and produce the processed games table (M7).

Fetches regular-season game logs from nba_api for a small initial season
range, saves raw responses untouched under ``data/raw/nba_games/``
(immutable, per CLAUDE.md), normalizes them into the project games schema,
validates, and writes ``data/processed/nba_games.parquet``.

Idempotent and quota-polite: if a season's raw file already exists it is
loaded instead of refetched. Pass ``--overwrite`` to refetch raw data and
replace the processed output. Games/results only — no odds, features,
models, or backtests.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from src.data.collect_nba_games import (
    collect_nba_games_for_seasons,
    normalize_nba_games,
    save_games_parquet,
)
from src.data.schema_validation import validate_binary_column
from src.data.team_mapping import CANONICAL_TEAM_IDS

SEASONS = ["2023-24", "2024-25"]
RAW_DIR = REPO_ROOT / "data" / "raw" / "nba_games"
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "nba_games.parquet"


def collect_or_load_raw_season(season: str, overwrite: bool) -> pl.DataFrame:
    """Fetch one season's raw game log, or load the saved raw file.

    Raw files are immutable once saved: an existing file is reused unless
    ``--overwrite`` is passed, so API quota is never spent twice.
    """
    raw_path = RAW_DIR / f"leaguegamelog_{season}.parquet"
    if raw_path.exists() and not overwrite:
        print(f"  {season}: loading saved raw file ({raw_path.name})")
        return pl.read_parquet(raw_path)

    print(f"  {season}: fetching from nba_api ...")
    try:
        raw = collect_nba_games_for_seasons([season])
    except Exception as exc:  # noqa: BLE001 - report clearly, never partial
        raise SystemExit(
            f"ERROR: nba_api fetch failed for season {season}: {exc}\n"
            "No partial data was saved. Check network access to "
            "stats.nba.com and retry."
        ) from exc

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        raw_path.unlink()  # only reachable with --overwrite
    raw.write_parquet(raw_path)
    print(f"  {season}: saved raw ({raw.height} team-game rows)")
    return raw


def exclude_neutral_site_games(raw: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Remove games with no home-team designation, explicitly and counted.

    NBA.com lists BOTH teams with '@' for neutral-site games (international
    games, NBA Cup semifinals in Las Vegas), so no home team exists in this
    source. Excluding them is a documented, counted exclusion per the
    research plan — never a silent drop. Revisit if a future milestone
    needs neutral-site games (official home designations exist in other
    nba_api endpoints).
    """
    home_counts = (
        raw.with_columns(
            pl.col("MATCHUP").str.contains("vs.", literal=True).alias("_is_home")
        )
        .group_by("GAME_ID")
        .agg(pl.col("_is_home").sum().alias("n_home"))
    )
    neutral_ids = (
        home_counts.filter(pl.col("n_home") != 1).get_column("GAME_ID").to_list()
    )
    return raw.filter(~pl.col("GAME_ID").is_in(neutral_ids)), sorted(neutral_ids)


def validate_processed_games(games: pl.DataFrame) -> int:
    """Run M7 validation checks; return duplicate game_id count (must be 0)."""
    if games.schema["game_id"] != pl.String:
        raise ValueError(f"game_id must be string, got {games.schema['game_id']}")
    if games.get_column("game_id").null_count() > 0:
        raise ValueError("game_id contains nulls.")
    if games.schema["game_date"] != pl.Date:
        raise ValueError(f"game_date must be pl.Date, got {games.schema['game_date']}")

    for col in ("home_team", "away_team"):
        bad = games.filter(~pl.col(col).is_in(list(CANONICAL_TEAM_IDS)))
        if bad.height > 0:
            raise ValueError(
                f"{col} contains non-canonical teams: "
                f"{bad.get_column(col).unique().to_list()}"
            )
    for col in ("home_score", "away_score"):
        if games.get_column(col).null_count() > 0:
            raise ValueError(f"{col} contains nulls.")
    validate_binary_column(games, "home_win")

    duplicates = games.group_by("game_id").len().filter(pl.col("len") > 1)
    if duplicates.height > 0:
        raise ValueError(
            f"duplicate game_id rows: {duplicates.get_column('game_id').to_list()[:5]}"
        )
    return duplicates.height


def main() -> None:
    """Collect, normalize, validate, save, and summarize real NBA games."""
    overwrite = "--overwrite" in sys.argv

    print(f"Collecting NBA games for seasons: {', '.join(SEASONS)}")
    raw_frames = [collect_or_load_raw_season(s, overwrite) for s in SEASONS]
    raw = pl.concat(raw_frames)

    raw_with_home, neutral_ids = exclude_neutral_site_games(raw)
    if neutral_ids:
        print(
            f"  excluded {len(neutral_ids)} neutral-site game(s) with no "
            f"home designation: {neutral_ids}"
        )

    games = normalize_nba_games(raw_with_home)
    duplicate_count = validate_processed_games(games)

    save_games_parquet(games, str(PROCESSED_PATH), overwrite=overwrite)

    dates = games.get_column("game_date")
    teams = sorted(
        set(games.get_column("home_team").to_list())
        | set(games.get_column("away_team").to_list())
    )
    print("-" * 64)
    print("Real NBA games collection summary")
    print(f"seasons collected:    {', '.join(SEASONS)}")
    print(f"raw team-game rows:   {raw.height}")
    print(f"neutral-site games excluded: {len(neutral_ids)}")
    print(f"normalized games:     {games.height}")
    print(f"date range:           {dates.min()} to {dates.max()}")
    print(f"teams found:          {len(teams)} ({', '.join(teams[:6])} ...)")
    print(f"duplicate game_ids:   {duplicate_count}")
    print(f"output:               {PROCESSED_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
