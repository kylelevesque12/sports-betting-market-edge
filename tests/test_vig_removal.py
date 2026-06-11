"""Unit tests for src.betting.vig_removal."""

import math

import pytest

from src.betting.vig_removal import remove_vig_two_way


class TestRemoveVigTwoWay:
    def test_standard_minus_110_market(self) -> None:
        # Both sides -110: implied 110/210 each, fair = 50/50.
        implied = 110 / 210
        fair_a, fair_b = remove_vig_two_way(implied, implied)
        assert math.isclose(fair_a, 0.5)
        assert math.isclose(fair_b, 0.5)

    def test_asymmetric_market(self) -> None:
        # -200 / +170 example: implied 2/3 and 100/270.
        prob_a, prob_b = 200 / 300, 100 / 270
        fair_a, fair_b = remove_vig_two_way(prob_a, prob_b)
        total = prob_a + prob_b
        assert math.isclose(fair_a, prob_a / total)
        assert math.isclose(fair_b, prob_b / total)

    def test_output_sums_to_one(self) -> None:
        fair_a, fair_b = remove_vig_two_way(0.55, 0.52)
        assert math.isclose(fair_a + fair_b, 1.0)

    def test_no_vig_market_unchanged(self) -> None:
        fair_a, fair_b = remove_vig_two_way(0.6, 0.4)
        assert math.isclose(fair_a, 0.6)
        assert math.isclose(fair_b, 0.4)

    def test_preserves_ordering(self) -> None:
        fair_a, fair_b = remove_vig_two_way(0.65, 0.45)
        assert fair_a > fair_b

    @pytest.mark.parametrize(
        ("prob_a", "prob_b"),
        [
            (0.0, 0.5),
            (0.5, 0.0),
            (1.0, 0.5),
            (0.5, 1.0),
            (-0.1, 0.5),
            (0.5, 1.2),
        ],
    )
    def test_invalid_probabilities_raise(self, prob_a: float, prob_b: float) -> None:
        with pytest.raises(ValueError):
            remove_vig_two_way(prob_a, prob_b)
