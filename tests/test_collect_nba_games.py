"""Unit tests for src.data.collect_nba_games (mocked data only, no network)."""

from pathlib import Path

import polars as pl
import pytest

from src.data.collect_nba_games import (
    CLEAN_GAMES_COLUMNS,
    normalize_nba_games,
    save_games_parquet,
)


def raw_team_game_rows() -> pl.DataFrame:
    """Two games in nba_api LeagueGameLog shape (one row per team-game)."""
    return pl.DataFrame(
        {
            "SEASON_ID": ["22023"] * 4,
            "GAME_ID": ["0022300001", "0022300001", "0022300002", "0022300002"],
            "GAME_DATE": ["2023-10-24", "2023-10-24", "2023-10-25", "2023-10-25"],
            "TEAM_ABBREVIATION": ["BOS", "NYK", "GS", "Phoenix Suns"],
            "MATCHUP": ["BOS vs. NYK", "NYK @ BOS", "GS vs. PHX", "PHX @ GS"],
            "PTS": [108, 104, 99, 112],
        }
    )


class TestNormalizeNbaGames:
    def test_creates_required_schema(self) -> None:
        clean = normalize_nba_games(raw_team_game_rows())
        assert tuple(clean.columns) == CLEAN_GAMES_COLUMNS
        assert clean.height == 2  # two games from four team rows

    def test_team_names_canonicalized(self) -> None:
        clean = normalize_nba_games(raw_team_game_rows())
        g2 = clean.filter(pl.col("game_id") == "0022300002").row(0, named=True)
        assert g2["home_team"] == "GSW"  # 'GS' alias
        assert g2["away_team"] == "PHX"  # full name 'Phoenix Suns'

    def test_game_date_parsed_to_date(self) -> None:
        clean = normalize_nba_games(raw_team_game_rows())
        assert clean.schema["game_date"] == pl.Date

    def test_season_label_derived(self) -> None:
        clean = normalize_nba_games(raw_team_game_rows())
        assert clean.get_column("season").unique().to_list() == ["2023-24"]

    def test_home_win_computed_correctly(self) -> None:
        clean = normalize_nba_games(raw_team_game_rows())
        g1 = clean.filter(pl.col("game_id") == "0022300001").row(0, named=True)
        assert (g1["home_score"], g1["away_score"], g1["home_win"]) == (108, 104, 1)
        g2 = clean.filter(pl.col("game_id") == "0022300002").row(0, named=True)
        assert (g2["home_score"], g2["away_score"], g2["home_win"]) == (99, 112, 0)

    def test_accepts_pandas_input(self) -> None:
        pd = pytest.importorskip("pandas")
        raw_pd = pd.DataFrame(raw_team_game_rows().to_dict(as_series=False))
        clean = normalize_nba_games(raw_pd)
        assert clean.height == 2

    def test_null_game_id_raises(self) -> None:
        raw = raw_team_game_rows().with_columns(
            pl.when(pl.arange(0, pl.len()) == 0)
            .then(None)
            .otherwise(pl.col("GAME_ID"))
            .alias("GAME_ID")
        )
        with pytest.raises(ValueError, match="'GAME_ID'.*null"):
            normalize_nba_games(raw)

    def test_null_date_raises(self) -> None:
        raw = raw_team_game_rows().with_columns(
            pl.when(pl.arange(0, pl.len()) == 0)
            .then(None)
            .otherwise(pl.col("GAME_DATE"))
            .alias("GAME_DATE")
        )
        with pytest.raises(ValueError, match="'GAME_DATE'.*null"):
            normalize_nba_games(raw)

    def test_null_score_raises(self) -> None:
        raw = raw_team_game_rows().with_columns(
            pl.when(pl.arange(0, pl.len()) == 0)
            .then(None)
            .otherwise(pl.col("PTS"))
            .alias("PTS")
        )
        with pytest.raises(ValueError, match="'PTS'.*null"):
            normalize_nba_games(raw)

    def test_unknown_team_raises(self) -> None:
        raw = raw_team_game_rows().with_columns(
            pl.when(pl.col("TEAM_ABBREVIATION") == "BOS")
            .then(pl.lit("Seattle SuperSonics"))
            .otherwise(pl.col("TEAM_ABBREVIATION"))
            .alias("TEAM_ABBREVIATION")
        )
        with pytest.raises(ValueError, match="unknown team"):
            normalize_nba_games(raw)

    def test_missing_raw_columns_raise(self) -> None:
        raw = raw_team_game_rows().drop("MATCHUP")
        with pytest.raises(ValueError, match="raw NBA games is missing.*MATCHUP"):
            normalize_nba_games(raw)

    def test_game_missing_away_row_raises(self) -> None:
        raw = raw_team_game_rows().filter(pl.col("MATCHUP") != "NYK @ BOS")
        with pytest.raises(ValueError, match="missing a home or away row"):
            normalize_nba_games(raw)

    def test_bad_season_id_raises(self) -> None:
        raw = raw_team_game_rows().with_columns(pl.lit("xyz").alias("SEASON_ID"))
        with pytest.raises(ValueError, match="unrecognized SEASON_ID"):
            normalize_nba_games(raw)

    def test_tied_scores_raise(self) -> None:
        raw = raw_team_game_rows().with_columns(pl.lit(100).alias("PTS"))
        with pytest.raises(ValueError, match="tied scores"):
            normalize_nba_games(raw)

    def test_unparseable_matchup_raises(self) -> None:
        raw = raw_team_game_rows().with_columns(
            pl.when(pl.arange(0, pl.len()) == 0)
            .then(pl.lit("BOS - NYK"))
            .otherwise(pl.col("MATCHUP"))
            .alias("MATCHUP")
        )
        with pytest.raises(ValueError, match="not recognizable as home/away"):
            normalize_nba_games(raw)


class TestSaveGamesParquet:
    def test_writes_parquet(self, tmp_path: Path) -> None:
        clean = normalize_nba_games(raw_team_game_rows())
        out = tmp_path / "processed" / "games.parquet"
        save_games_parquet(clean, str(out))
        assert out.exists()
        round_trip = pl.read_parquet(out)
        assert round_trip.equals(clean)

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        clean = normalize_nba_games(raw_team_game_rows())
        out = tmp_path / "a" / "b" / "c" / "games.parquet"
        save_games_parquet(clean, str(out))
        assert out.exists()

    def test_refuses_overwrite_by_default(self, tmp_path: Path) -> None:
        clean = normalize_nba_games(raw_team_game_rows())
        out = tmp_path / "games.parquet"
        save_games_parquet(clean, str(out))
        with pytest.raises(FileExistsError, match="already exists"):
            save_games_parquet(clean, str(out))

    def test_overwrite_true_replaces(self, tmp_path: Path) -> None:
        clean = normalize_nba_games(raw_team_game_rows())
        out = tmp_path / "games.parquet"
        save_games_parquet(clean, str(out))
        save_games_parquet(clean.head(1), str(out), overwrite=True)
        assert pl.read_parquet(out).height == 1
