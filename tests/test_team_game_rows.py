"""Unit tests for src.features.team_game_rows."""

from pathlib import Path

import polars as pl
import pytest

from src.features.team_game_rows import (
    OUTPUT_COLUMNS,
    create_team_game_rows,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
GAMES_CSV = REPO_ROOT / "data" / "external" / "sample_games.csv"


@pytest.fixture()
def two_games() -> pl.DataFrame:
    """One home win (G1) and one away win (G2)."""
    return pl.DataFrame(
        {
            "game_id": ["G1", "G2"],
            "season": ["2025-26", "2025-26"],
            "game_date": ["2025-11-01", "2025-11-02"],
            "home_team": ["Anvils", "Lynx"],
            "away_team": ["Krakens", "Comets"],
            "home_score": [112, 98],
            "away_score": [104, 101],
            "home_win": [1, 0],
        }
    )


class TestStructure:
    def test_two_rows_per_game(self, two_games: pl.DataFrame) -> None:
        result = create_team_game_rows(two_games)
        assert result.height == 2 * two_games.height
        counts = result.group_by("game_id").len()
        assert counts.get_column("len").to_list() == [2, 2]

    def test_output_columns(self, two_games: pl.DataFrame) -> None:
        result = create_team_game_rows(two_games)
        assert tuple(result.columns) == OUTPUT_COLUMNS

    def test_home_row_first_within_game(self, two_games: pl.DataFrame) -> None:
        result = create_team_game_rows(two_games)
        assert result.get_column("is_home").to_list() == [1, 0, 1, 0]
        assert result.get_column("game_date").to_list() == sorted(
            result.get_column("game_date").to_list()
        )


class TestHomeRow:
    def test_home_win_game(self, two_games: pl.DataFrame) -> None:
        result = create_team_game_rows(two_games)
        home = result.filter((pl.col("game_id") == "G1") & (pl.col("is_home") == 1))
        row = home.row(0, named=True)
        assert row["team"] == "Anvils"
        assert row["opponent"] == "Krakens"
        assert row["points_for"] == 112
        assert row["points_against"] == 104
        assert row["point_diff"] == 8
        assert row["team_win"] == 1


class TestAwayRow:
    def test_away_win_game(self, two_games: pl.DataFrame) -> None:
        result = create_team_game_rows(two_games)
        away = result.filter((pl.col("game_id") == "G2") & (pl.col("is_home") == 0))
        row = away.row(0, named=True)
        assert row["team"] == "Comets"
        assert row["opponent"] == "Lynx"
        assert row["points_for"] == 101
        assert row["points_against"] == 98
        assert row["point_diff"] == 3
        assert row["team_win"] == 1


class TestWinAssignment:
    def test_home_win_sides(self, two_games: pl.DataFrame) -> None:
        result = create_team_game_rows(two_games).filter(pl.col("game_id") == "G1")
        wins = {r["is_home"]: r["team_win"] for r in result.iter_rows(named=True)}
        assert wins == {1: 1, 0: 0}

    def test_away_win_sides(self, two_games: pl.DataFrame) -> None:
        result = create_team_game_rows(two_games).filter(pl.col("game_id") == "G2")
        wins = {r["is_home"]: r["team_win"] for r in result.iter_rows(named=True)}
        assert wins == {1: 0, 0: 1}

    def test_exactly_one_winner_per_game(self, two_games: pl.DataFrame) -> None:
        result = create_team_game_rows(two_games)
        win_sums = result.group_by("game_id").agg(pl.col("team_win").sum())
        assert win_sums.get_column("team_win").to_list() == [1, 1]


class TestValidation:
    @pytest.mark.parametrize("dropped", ["game_id", "home_score", "home_win"])
    def test_missing_column_raises(
        self, two_games: pl.DataFrame, dropped: str
    ) -> None:
        with pytest.raises(ValueError, match="missing required columns"):
            create_team_game_rows(two_games.drop(dropped))


class TestToySample:
    def test_runs_on_sample_games_csv(self) -> None:
        games = pl.read_csv(GAMES_CSV)
        result = create_team_game_rows(games)
        assert result.height == 20
        # Point diffs of the two rows of each game must cancel out.
        diff_sums = result.group_by("game_id").agg(pl.col("point_diff").sum())
        assert diff_sums.get_column("point_diff").to_list() == [0] * 10
        # Every game has exactly one winning team-row.
        win_sums = result.group_by("game_id").agg(pl.col("team_win").sum())
        assert win_sums.get_column("team_win").to_list() == [1] * 10
