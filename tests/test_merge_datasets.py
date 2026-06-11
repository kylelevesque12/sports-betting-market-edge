"""Unit tests for src.data.merge_datasets using the toy sample data."""

from pathlib import Path

import polars as pl
import pytest

from src.data.merge_datasets import (
    REQUIRED_GAME_COLUMNS,
    REQUIRED_ODDS_COLUMNS,
    load_games,
    load_odds,
    merge_games_and_odds,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
GAMES_CSV = REPO_ROOT / "data" / "external" / "sample_games.csv"
ODDS_CSV = REPO_ROOT / "data" / "external" / "sample_odds.csv"


@pytest.fixture()
def games() -> pl.DataFrame:
    return load_games(str(GAMES_CSV))


@pytest.fixture()
def odds() -> pl.DataFrame:
    return load_odds(str(ODDS_CSV))


class TestLoadGames:
    def test_loads_toy_sample(self, games: pl.DataFrame) -> None:
        assert games.height == 10
        for col in REQUIRED_GAME_COLUMNS:
            assert col in games.columns

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad_games.csv"
        bad.write_text("game_id,season\nG001,2025-26\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_games(str(bad))


class TestLoadOdds:
    def test_loads_toy_sample(self, odds: pl.DataFrame) -> None:
        assert odds.height == 40
        for col in REQUIRED_ODDS_COLUMNS:
            assert col in odds.columns

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad_odds.csv"
        bad.write_text("game_id,sportsbook\nG001,AlphaBook\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_odds(str(bad))


class TestMergeGamesAndOdds:
    def test_preserves_one_row_per_odds_row(
        self, games: pl.DataFrame, odds: pl.DataFrame
    ) -> None:
        merged = merge_games_and_odds(games, odds)
        assert merged.height == odds.height

    def test_one_row_per_game_per_book_per_timestamp(
        self, games: pl.DataFrame, odds: pl.DataFrame
    ) -> None:
        merged = merge_games_and_odds(games, odds)
        keys = merged.select(["game_id", "sportsbook", "timestamp"])
        assert keys.unique().height == merged.height

    def test_teams_consistent_after_merge(
        self, games: pl.DataFrame, odds: pl.DataFrame
    ) -> None:
        merged = merge_games_and_odds(games, odds)
        assert merged.filter(
            (pl.col("home_team") != pl.col("home_team_games"))
            | (pl.col("away_team") != pl.col("away_team_games"))
        ).height == 0

    def test_game_columns_present_after_merge(
        self, games: pl.DataFrame, odds: pl.DataFrame
    ) -> None:
        merged = merge_games_and_odds(games, odds)
        for col in ("season", "game_date", "home_score", "away_score", "home_win"):
            assert col in merged.columns

    def test_unmatched_game_id_raises(
        self, games: pl.DataFrame, odds: pl.DataFrame
    ) -> None:
        orphan = odds.head(1).with_columns(pl.lit("G999").alias("game_id"))
        bad_odds = pl.concat([odds, orphan])
        with pytest.raises(ValueError, match="G999"):
            merge_games_and_odds(games, bad_odds)

    def test_missing_game_columns_raise(self, odds: pl.DataFrame) -> None:
        bad_games = pl.DataFrame({"game_id": ["G001"], "season": ["2025-26"]})
        with pytest.raises(ValueError, match="games is missing"):
            merge_games_and_odds(bad_games, odds)

    def test_missing_odds_columns_raise(self, games: pl.DataFrame) -> None:
        bad_odds = pl.DataFrame({"game_id": ["G001"], "sportsbook": ["AlphaBook"]})
        with pytest.raises(ValueError, match="odds is missing"):
            merge_games_and_odds(games, bad_odds)
