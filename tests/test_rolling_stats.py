"""Unit tests for src.features.rolling_stats — leakage checks above all."""

import math
from pathlib import Path

import polars as pl
import pytest

from src.features.rolling_stats import add_rolling_team_features
from src.features.team_game_rows import create_team_game_rows

REPO_ROOT = Path(__file__).resolve().parent.parent
GAMES_CSV = REPO_ROOT / "data" / "external" / "sample_games.csv"


@pytest.fixture()
def two_team_games() -> pl.DataFrame:
    """Two teams, four games each, with easily hand-checked values.

    Team A: points_for 100, 110, 120, 130; wins 1, 0, 1, 1.
    Team B: points_for 90, 92, 94, 96; wins 0, 1, 0, 0.
    """
    return pl.DataFrame(
        {
            "team": ["A", "B"] * 4,
            "game_date": [
                "2025-11-01", "2025-11-01",
                "2025-11-03", "2025-11-03",
                "2025-11-05", "2025-11-05",
                "2025-11-07", "2025-11-07",
            ],
            "points_for": [100, 90, 110, 92, 120, 94, 130, 96],
            "points_against": [95, 95, 115, 90, 110, 99, 120, 100],
            "point_diff": [5, -5, -5, 2, 10, -5, 10, -4],
            "team_win": [1, 0, 0, 1, 1, 0, 1, 0],
        }
    )


def team_rows(df: pl.DataFrame, team: str) -> list[dict]:
    return df.filter(pl.col("team") == team).sort("game_date").to_dicts()


class TestStructure:
    def test_rolling_columns_created(self, two_team_games: pl.DataFrame) -> None:
        result = add_rolling_team_features(two_team_games)
        for w in (3, 5):
            for stem in (
                "rolling_win_pct",
                "rolling_points_for",
                "rolling_points_against",
                "rolling_point_diff",
            ):
                assert f"{stem}_{w}" in result.columns

    def test_original_columns_and_rows_preserved(
        self, two_team_games: pl.DataFrame
    ) -> None:
        result = add_rolling_team_features(two_team_games)
        assert result.height == two_team_games.height
        for col in two_team_games.columns:
            assert col in result.columns


class TestLeakageGuards:
    def test_first_game_has_null_features(self, two_team_games: pl.DataFrame) -> None:
        result = add_rolling_team_features(two_team_games)
        for team in ("A", "B"):
            first = team_rows(result, team)[0]
            assert first["rolling_win_pct_3"] is None
            assert first["rolling_points_for_3"] is None
            assert first["rolling_points_against_3"] is None
            assert first["rolling_point_diff_3"] is None

    def test_second_game_uses_only_first_game(
        self, two_team_games: pl.DataFrame
    ) -> None:
        result = add_rolling_team_features(two_team_games)
        second_a = team_rows(result, "A")[1]
        assert math.isclose(second_a["rolling_points_for_3"], 100.0)
        assert math.isclose(second_a["rolling_win_pct_3"], 1.0)
        second_b = team_rows(result, "B")[1]
        assert math.isclose(second_b["rolling_points_for_3"], 90.0)
        assert math.isclose(second_b["rolling_win_pct_3"], 0.0)

    def test_current_game_excluded(self, two_team_games: pl.DataFrame) -> None:
        # Team A game 3: prior games scored 100, 110 -> mean 105.
        # Including the current game (120) would give 110 — must not happen.
        result = add_rolling_team_features(two_team_games)
        third_a = team_rows(result, "A")[2]
        assert math.isclose(third_a["rolling_points_for_3"], 105.0)
        # Team A game 3 prior wins: 1, 0 -> 0.5 (current win=1 excluded).
        assert math.isclose(third_a["rolling_win_pct_3"], 0.5)

    def test_no_leakage_across_teams(self, two_team_games: pl.DataFrame) -> None:
        # Team B's features must come only from B's games (90s range),
        # never contaminated by A's (100s range).
        result = add_rolling_team_features(two_team_games)
        for row in team_rows(result, "B")[1:]:
            assert row["rolling_points_for_3"] < 100

    def test_window_caps_lookback(self, two_team_games: pl.DataFrame) -> None:
        # Team A game 4 with window 2: mean of games 2-3 (110, 120) = 115,
        # not the all-history mean (110).
        result = add_rolling_team_features(two_team_games, windows=[2])
        fourth_a = team_rows(result, "A")[3]
        assert math.isclose(fourth_a["rolling_points_for_2"], 115.0)


class TestWindows:
    def test_multiple_windows(self, two_team_games: pl.DataFrame) -> None:
        result = add_rolling_team_features(two_team_games, windows=[2, 3])
        fourth_a = team_rows(result, "A")[3]
        assert math.isclose(fourth_a["rolling_points_for_2"], 115.0)
        assert math.isclose(fourth_a["rolling_points_for_3"], 110.0)

    @pytest.mark.parametrize("bad_windows", [[0], [-1], [3, 0]])
    def test_invalid_windows_raise(
        self, two_team_games: pl.DataFrame, bad_windows: list[int]
    ) -> None:
        with pytest.raises(ValueError, match="windows must be >= 1"):
            add_rolling_team_features(two_team_games, windows=bad_windows)

    def test_empty_windows_raise(self, two_team_games: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="at least one window"):
            add_rolling_team_features(two_team_games, windows=[])


class TestValidation:
    @pytest.mark.parametrize("dropped", ["team", "game_date", "team_win"])
    def test_missing_columns_raise(
        self, two_team_games: pl.DataFrame, dropped: str
    ) -> None:
        with pytest.raises(ValueError, match="missing required columns"):
            add_rolling_team_features(two_team_games.drop(dropped))


class TestToySample:
    def test_works_on_sample_team_game_rows(self) -> None:
        games = pl.read_csv(GAMES_CSV)
        team_games = create_team_game_rows(games)
        result = add_rolling_team_features(team_games)
        assert result.height == 20
        # 4 teams -> exactly 4 first-game rows with null rolling features.
        nulls = result.filter(pl.col("rolling_win_pct_3").is_null())
        assert nulls.height == 4
        assert nulls.get_column("team").n_unique() == 4
        # All non-null win pcts are valid probabilities.
        valid = result.drop_nulls("rolling_win_pct_3")
        assert valid.filter(
            (pl.col("rolling_win_pct_3") < 0) | (pl.col("rolling_win_pct_3") > 1)
        ).height == 0
