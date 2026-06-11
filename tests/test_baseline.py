"""Unit tests for src.models.baseline."""

import math

import polars as pl
import pytest

from src.models.baseline import predict_market_baseline


@pytest.fixture()
def team_game_rows() -> pl.DataFrame:
    """Two games as team-game rows (home + away each) with market features."""
    return pl.DataFrame(
        {
            "game_id": ["G1", "G1", "G2", "G2"],
            "team": ["Anvils", "Krakens", "Lynx", "Comets"],
            "is_home": [1, 0, 1, 0],
            "home_fair_market_prob": [0.60, 0.60, 0.45, 0.45],
            "away_fair_market_prob": [0.40, 0.40, 0.55, 0.55],
        }
    )


class TestPredictionAssignment:
    def test_home_rows_get_home_prob(self, team_game_rows: pl.DataFrame) -> None:
        result = predict_market_baseline(team_game_rows)
        home = result.filter(pl.col("is_home") == 1)
        assert home.get_column("predicted_win_prob").to_list() == [0.60, 0.45]

    def test_away_rows_get_away_prob(self, team_game_rows: pl.DataFrame) -> None:
        result = predict_market_baseline(team_game_rows)
        away = result.filter(pl.col("is_home") == 0)
        assert away.get_column("predicted_win_prob").to_list() == [0.40, 0.55]

    def test_game_probs_sum_to_one(self, team_game_rows: pl.DataFrame) -> None:
        result = predict_market_baseline(team_game_rows)
        sums = result.group_by("game_id").agg(pl.col("predicted_win_prob").sum())
        assert all(
            math.isclose(s, 1.0) for s in sums.get_column("predicted_win_prob").to_list()
        )


class TestStructure:
    def test_row_count_preserved(self, team_game_rows: pl.DataFrame) -> None:
        result = predict_market_baseline(team_game_rows)
        assert result.height == team_game_rows.height

    def test_existing_columns_preserved(self, team_game_rows: pl.DataFrame) -> None:
        result = predict_market_baseline(team_game_rows)
        for col in team_game_rows.columns:
            assert col in result.columns
        assert result.get_column("team").to_list() == ["Anvils", "Krakens", "Lynx", "Comets"]


class TestValidation:
    @pytest.mark.parametrize(
        "dropped", ["is_home", "home_fair_market_prob", "away_fair_market_prob"]
    )
    def test_missing_columns_raise(
        self, team_game_rows: pl.DataFrame, dropped: str
    ) -> None:
        with pytest.raises(ValueError, match="missing required columns"):
            predict_market_baseline(team_game_rows.drop(dropped))

    @pytest.mark.parametrize("bad_value", [2, -1])
    def test_invalid_is_home_raises(
        self, team_game_rows: pl.DataFrame, bad_value: int
    ) -> None:
        bad = team_game_rows.with_columns(
            pl.when(pl.col("team") == "Anvils")
            .then(pl.lit(bad_value))
            .otherwise(pl.col("is_home"))
            .alias("is_home")
        )
        with pytest.raises(ValueError, match="is_home must contain only 0 or 1"):
            predict_market_baseline(bad)

    def test_null_is_home_raises(self, team_game_rows: pl.DataFrame) -> None:
        bad = team_game_rows.with_columns(
            pl.when(pl.col("team") == "Anvils")
            .then(None)
            .otherwise(pl.col("is_home"))
            .alias("is_home")
        )
        with pytest.raises(ValueError, match="is_home"):
            predict_market_baseline(bad)

    @pytest.mark.parametrize("bad_prob", [-0.1, 1.1])
    @pytest.mark.parametrize(
        "col", ["home_fair_market_prob", "away_fair_market_prob"]
    )
    def test_out_of_range_probabilities_raise(
        self, team_game_rows: pl.DataFrame, col: str, bad_prob: float
    ) -> None:
        bad = team_game_rows.with_columns(
            pl.when(pl.col("team") == "Anvils")
            .then(pl.lit(bad_prob))
            .otherwise(pl.col(col))
            .alias(col)
        )
        with pytest.raises(ValueError, match="must be within"):
            predict_market_baseline(bad)

    def test_null_probability_raises(self, team_game_rows: pl.DataFrame) -> None:
        bad = team_game_rows.with_columns(
            pl.when(pl.col("team") == "Anvils")
            .then(None)
            .otherwise(pl.col("home_fair_market_prob"))
            .alias("home_fair_market_prob")
        )
        with pytest.raises(ValueError, match="home_fair_market_prob"):
            predict_market_baseline(bad)


class TestMixedRows:
    def test_works_with_home_and_away_rows(self) -> None:
        df = pl.DataFrame(
            {
                "is_home": [1, 0],
                "home_fair_market_prob": [0.7, 0.7],
                "away_fair_market_prob": [0.3, 0.3],
            }
        )
        result = predict_market_baseline(df)
        assert result.get_column("predicted_win_prob").to_list() == [0.7, 0.3]
