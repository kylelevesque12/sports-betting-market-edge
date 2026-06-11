"""Unit tests for src.betting.odds_conversion."""

import math

import pytest

from src.betting.odds_conversion import (
    american_to_decimal,
    american_to_implied_probability,
    decimal_to_implied_probability,
)


class TestAmericanToDecimal:
    def test_positive_odds(self) -> None:
        assert math.isclose(american_to_decimal(150), 2.5)
        assert math.isclose(american_to_decimal(100), 2.0)
        assert math.isclose(american_to_decimal(250), 3.5)

    def test_negative_odds(self) -> None:
        assert math.isclose(american_to_decimal(-100), 2.0)
        assert math.isclose(american_to_decimal(-200), 1.5)
        assert math.isclose(american_to_decimal(-150), 1 + 100 / 150)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            american_to_decimal(0)


class TestAmericanToImpliedProbability:
    def test_even_odds(self) -> None:
        assert math.isclose(american_to_implied_probability(100), 0.5)
        assert math.isclose(american_to_implied_probability(-100), 0.5)

    def test_favorite(self) -> None:
        # -150 implies 150 / (150 + 100) = 0.6
        assert math.isclose(american_to_implied_probability(-150), 0.6)

    def test_underdog(self) -> None:
        # +150 implies 100 / (150 + 100) = 0.4
        assert math.isclose(american_to_implied_probability(150), 0.4)

    def test_standard_juice(self) -> None:
        # -110 implies 110 / 210
        assert math.isclose(american_to_implied_probability(-110), 110 / 210)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            american_to_implied_probability(0)


class TestDecimalToImpliedProbability:
    def test_known_values(self) -> None:
        assert math.isclose(decimal_to_implied_probability(2.0), 0.5)
        assert math.isclose(decimal_to_implied_probability(4.0), 0.25)
        assert math.isclose(decimal_to_implied_probability(1.25), 0.8)

    @pytest.mark.parametrize("bad_odds", [1.0, 0.9, 0.0, -2.0])
    def test_invalid_odds_raise(self, bad_odds: float) -> None:
        with pytest.raises(ValueError):
            decimal_to_implied_probability(bad_odds)


class TestRoundTrip:
    @pytest.mark.parametrize("american", [-300, -110, -101, 100, 120, 450])
    def test_american_decimal_probability_consistency(self, american: float) -> None:
        via_decimal = decimal_to_implied_probability(american_to_decimal(american))
        direct = american_to_implied_probability(american)
        assert math.isclose(via_decimal, direct)
