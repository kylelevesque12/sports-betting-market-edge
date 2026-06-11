"""Unit tests for src.backtesting.simple_backtest."""

import math

import polars as pl
import pytest

from src.backtesting.simple_backtest import run_flat_stake_backtest


def frame(
    team_win: list[int],
    bet_flag: list[bool],
    decimal_odds: list[float],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "team_win": team_win,
            "bet_flag": bet_flag,
            "decimal_odds": decimal_odds,
        }
    )


class TestNoBets:
    def test_zero_bets_returns_zeroed_metrics(self) -> None:
        result = run_flat_stake_backtest(
            frame([1, 0], [False, False], [1.9, 2.1])
        )
        assert result == {
            "total_bets": 0,
            "total_staked": 0.0,
            "total_profit": 0.0,
            "roi": 0.0,
            "win_rate": 0.0,
            "average_odds": None,
        }


class TestSingleBets:
    def test_single_winning_bet(self) -> None:
        result = run_flat_stake_backtest(frame([1], [True], [2.5]))
        assert result["total_bets"] == 1
        assert math.isclose(result["total_staked"], 1.0)
        assert math.isclose(result["total_profit"], 1.5)  # 1 * (2.5 - 1)
        assert math.isclose(result["roi"], 1.5)
        assert math.isclose(result["win_rate"], 1.0)
        assert math.isclose(result["average_odds"], 2.5)

    def test_single_losing_bet(self) -> None:
        result = run_flat_stake_backtest(frame([0], [True], [2.5]))
        assert math.isclose(result["total_profit"], -1.0)
        assert math.isclose(result["roi"], -1.0)
        assert math.isclose(result["win_rate"], 0.0)

    def test_custom_stake_scales_profit(self) -> None:
        result = run_flat_stake_backtest(frame([1], [True], [3.0]), stake=2.0)
        assert math.isclose(result["total_staked"], 2.0)
        assert math.isclose(result["total_profit"], 4.0)  # 2 * (3.0 - 1)
        assert math.isclose(result["roi"], 2.0)


class TestMixedResults:
    def test_mixed_wins_and_losses(self) -> None:
        # Win at 2.0 (+1), loss (-1), win at 3.0 (+2), loss (-1) -> +1 total.
        result = run_flat_stake_backtest(
            frame([1, 0, 1, 0], [True] * 4, [2.0, 1.9, 3.0, 2.2])
        )
        assert result["total_bets"] == 4
        assert math.isclose(result["total_staked"], 4.0)
        assert math.isclose(result["total_profit"], 1.0)
        assert math.isclose(result["roi"], 0.25)
        assert math.isclose(result["win_rate"], 0.5)
        assert math.isclose(result["average_odds"], (2.0 + 1.9 + 3.0 + 2.2) / 4)

    def test_unflagged_rows_ignored(self) -> None:
        # Identical to the mixed case plus two no-bet rows that would change
        # everything if counted (a huge win and a loss).
        result = run_flat_stake_backtest(
            frame(
                [1, 0, 1, 0, 1, 0],
                [True, True, True, True, False, False],
                [2.0, 1.9, 3.0, 2.2, 10.0, 1.5],
            )
        )
        assert result["total_bets"] == 4
        assert math.isclose(result["total_profit"], 1.0)
        assert math.isclose(result["average_odds"], (2.0 + 1.9 + 3.0 + 2.2) / 4)


class TestValidation:
    @pytest.mark.parametrize("dropped", ["team_win", "bet_flag", "decimal_odds"])
    def test_missing_columns_raise(self, dropped: str) -> None:
        df = frame([1], [True], [2.0]).drop(dropped)
        with pytest.raises(ValueError, match="missing required columns"):
            run_flat_stake_backtest(df)

    @pytest.mark.parametrize("bad_stake", [0.0, -1.0])
    def test_non_positive_stake_raises(self, bad_stake: float) -> None:
        with pytest.raises(ValueError, match="stake must be positive"):
            run_flat_stake_backtest(frame([1], [True], [2.0]), stake=bad_stake)

    @pytest.mark.parametrize("bad_odds", [1.0, 0.5, -2.0])
    def test_invalid_odds_on_placed_bets_raise(self, bad_odds: float) -> None:
        with pytest.raises(ValueError, match="greater than 1 for placed bets"):
            run_flat_stake_backtest(frame([1], [True], [bad_odds]))

    def test_invalid_odds_on_unplaced_rows_allowed(self) -> None:
        # Odds quality only matters for settled bets.
        result = run_flat_stake_backtest(
            frame([1, 1], [True, False], [2.0, 1.0])
        )
        assert result["total_bets"] == 1

    @pytest.mark.parametrize("bad_win", [2, -1])
    def test_invalid_team_win_raises(self, bad_win: int) -> None:
        with pytest.raises(ValueError, match="only 0 and 1"):
            run_flat_stake_backtest(frame([bad_win], [True], [2.0]))

    def test_non_boolean_bet_flag_raises(self) -> None:
        df = pl.DataFrame(
            {"team_win": [1], "bet_flag": [1], "decimal_odds": [2.0]}
        )
        with pytest.raises(ValueError, match="bet_flag must be a boolean"):
            run_flat_stake_backtest(df)

    @pytest.mark.parametrize("col", ["team_win", "bet_flag", "decimal_odds"])
    def test_nulls_raise(self, col: str) -> None:
        df = frame([1, 0], [True, True], [2.0, 2.0]).with_columns(
            pl.when(pl.arange(0, pl.len()) == 0)
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )
        with pytest.raises(ValueError, match="null value"):
            run_flat_stake_backtest(df)
