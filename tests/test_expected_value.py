"""Unit tests for src.betting.expected_value."""

import math

import pytest

from src.betting.expected_value import edge, expected_value_per_unit


class TestExpectedValuePerUnit:
    def test_fair_odds_zero_ev(self) -> None:
        # 50% at decimal 2.0 is exactly fair.
        assert math.isclose(expected_value_per_unit(0.5, 2.0), 0.0)

    def test_positive_ev(self) -> None:
        # 60% at decimal 2.0: 0.6 * 1 - 0.4 = 0.2
        assert math.isclose(expected_value_per_unit(0.6, 2.0), 0.2)

    def test_negative_ev(self) -> None:
        # 40% at decimal 2.0: 0.4 * 1 - 0.6 = -0.2
        assert math.isclose(expected_value_per_unit(0.4, 2.0), -0.2)

    def test_longshot(self) -> None:
        # 25% at decimal 5.0: 0.25 * 4 - 0.75 = 0.25
        assert math.isclose(expected_value_per_unit(0.25, 5.0), 0.25)

    def test_boundary_probabilities_allowed(self) -> None:
        # Certain win profits (odds - 1); certain loss loses the stake.
        assert math.isclose(expected_value_per_unit(1.0, 2.5), 1.5)
        assert math.isclose(expected_value_per_unit(0.0, 2.5), -1.0)

    @pytest.mark.parametrize("bad_prob", [-0.01, 1.01, 2.0])
    def test_invalid_probability_raises(self, bad_prob: float) -> None:
        with pytest.raises(ValueError):
            expected_value_per_unit(bad_prob, 2.0)

    @pytest.mark.parametrize("bad_odds", [1.0, 0.5, 0.0, -3.0])
    def test_invalid_odds_raise(self, bad_odds: float) -> None:
        with pytest.raises(ValueError):
            expected_value_per_unit(0.5, bad_odds)


class TestEdge:
    def test_positive_edge(self) -> None:
        assert math.isclose(edge(0.55, 0.50), 0.05)

    def test_negative_edge(self) -> None:
        assert math.isclose(edge(0.45, 0.50), -0.05)

    def test_zero_edge(self) -> None:
        assert math.isclose(edge(0.5, 0.5), 0.0)

    def test_boundaries_allowed(self) -> None:
        assert math.isclose(edge(1.0, 0.0), 1.0)
        assert math.isclose(edge(0.0, 1.0), -1.0)

    @pytest.mark.parametrize(
        ("model_prob", "fair_market_prob"),
        [(-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, 1.1)],
    )
    def test_invalid_inputs_raise(self, model_prob: float, fair_market_prob: float) -> None:
        with pytest.raises(ValueError):
            edge(model_prob, fair_market_prob)
