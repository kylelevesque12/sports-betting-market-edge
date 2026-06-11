"""Unit tests for src.features.market_features."""

import math
from pathlib import Path

import polars as pl
import pytest

from src.features.market_features import (
    FEATURE_COLUMNS,
    add_moneyline_market_features,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ODDS_CSV = REPO_ROOT / "data" / "external" / "sample_odds.csv"


def make_odds(home: int, away: int) -> pl.DataFrame:
    return pl.DataFrame(
        {"home_american_odds": [home], "away_american_odds": [away]}
    )


class TestOutputStructure:
    def test_feature_columns_created(self) -> None:
        result = add_moneyline_market_features(make_odds(-150, 130))
        for col in FEATURE_COLUMNS:
            assert col in result.columns

    def test_original_columns_preserved(self) -> None:
        df = make_odds(-150, 130).with_columns(pl.lit("G001").alias("game_id"))
        result = add_moneyline_market_features(df)
        assert "game_id" in result.columns
        assert result.get_column("home_american_odds").to_list() == [-150]


class TestStandardJuiceMarket:
    def test_minus_110_both_sides(self) -> None:
        row = add_moneyline_market_features(make_odds(-110, -110)).row(0, named=True)
        # Raw implied probabilities include vig, so each side is above 0.5.
        assert row["home_implied_prob_raw"] > 0.5
        assert row["away_implied_prob_raw"] > 0.5
        assert math.isclose(row["home_implied_prob_raw"], 110 / 210)
        # Fair probabilities are 0.5 / 0.5 after vig removal.
        assert math.isclose(row["home_fair_market_prob"], 0.5)
        assert math.isclose(row["away_fair_market_prob"], 0.5)

    def test_vig_positive(self) -> None:
        row = add_moneyline_market_features(make_odds(-110, -110)).row(0, named=True)
        assert row["sportsbook_vig"] > 0
        assert math.isclose(row["sportsbook_vig"], 2 * (110 / 210) - 1)


class TestDecimalOdds:
    def test_known_values(self) -> None:
        row = add_moneyline_market_features(make_odds(-150, 130)).row(0, named=True)
        assert math.isclose(row["home_decimal_odds"], 1 + 100 / 150)
        assert math.isclose(row["away_decimal_odds"], 2.3)

    def test_even_money(self) -> None:
        row = add_moneyline_market_features(make_odds(100, -100)).row(0, named=True)
        assert math.isclose(row["home_decimal_odds"], 2.0)
        assert math.isclose(row["away_decimal_odds"], 2.0)


class TestFairProbabilities:
    def test_sum_to_one(self) -> None:
        row = add_moneyline_market_features(make_odds(-200, 175)).row(0, named=True)
        assert math.isclose(
            row["home_fair_market_prob"] + row["away_fair_market_prob"], 1.0
        )

    def test_favorite_has_higher_fair_prob(self) -> None:
        row = add_moneyline_market_features(make_odds(-200, 175)).row(0, named=True)
        assert row["home_fair_market_prob"] > row["away_fair_market_prob"]


class TestValidation:
    def test_zero_american_odds_raise(self) -> None:
        with pytest.raises(ValueError, match="American odds cannot be 0"):
            add_moneyline_market_features(make_odds(0, -110))
        with pytest.raises(ValueError, match="American odds cannot be 0"):
            add_moneyline_market_features(make_odds(-110, 0))

    def test_missing_columns_raise(self) -> None:
        df = pl.DataFrame({"home_american_odds": [-110]})
        with pytest.raises(ValueError, match="missing required columns"):
            add_moneyline_market_features(df)
        with pytest.raises(ValueError, match="missing required columns"):
            add_moneyline_market_features(pl.DataFrame({"x": [1]}))


class TestToySample:
    def test_runs_on_sample_odds_csv(self) -> None:
        odds = pl.read_csv(ODDS_CSV)
        result = add_moneyline_market_features(odds)
        assert result.height == odds.height  # row count preserved
        # All vigs positive (every toy book charges margin); all fair pairs sum to 1.
        assert result.filter(pl.col("sportsbook_vig") <= 0).height == 0
        sums = result.get_column("home_fair_market_prob") + result.get_column(
            "away_fair_market_prob"
        )
        assert all(math.isclose(s, 1.0) for s in sums.to_list())
