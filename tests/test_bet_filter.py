"""Unit tests for src.betting.bet_filter."""

import math

import pytest

from src.betting.bet_filter import classify_bet_side, should_bet


class TestShouldBet:
    def test_qualifies_when_both_thresholds_met(self) -> None:
        assert should_bet(0.58, 0.50, ev=0.10) is True

    def test_rejects_when_edge_too_small(self) -> None:
        # Edge 0.02 < default min_edge 0.03, EV fine.
        assert should_bet(0.52, 0.50, ev=0.10) is False

    def test_rejects_when_ev_too_small(self) -> None:
        # Edge fine, EV 0.01 < default min_ev 0.02.
        assert should_bet(0.58, 0.50, ev=0.01) is False

    def test_thresholds_are_inclusive(self) -> None:
        assert should_bet(0.53, 0.50, ev=0.02, min_edge=0.03, min_ev=0.02) is True

    def test_custom_thresholds(self) -> None:
        assert should_bet(0.52, 0.50, ev=0.01, min_edge=0.01, min_ev=0.01) is True
        assert should_bet(0.52, 0.50, ev=0.01, min_edge=0.05, min_ev=0.01) is False

    def test_negative_edge_never_qualifies(self) -> None:
        assert should_bet(0.45, 0.50, ev=0.10) is False

    @pytest.mark.parametrize("bad_prob", [-0.1, 1.1])
    def test_invalid_model_prob_raises(self, bad_prob: float) -> None:
        with pytest.raises(ValueError):
            should_bet(bad_prob, 0.5, ev=0.1)

    @pytest.mark.parametrize("bad_prob", [-0.1, 1.1])
    def test_invalid_fair_prob_raises(self, bad_prob: float) -> None:
        with pytest.raises(ValueError):
            should_bet(0.5, bad_prob, ev=0.1)

    def test_negative_min_edge_raises(self) -> None:
        with pytest.raises(ValueError):
            should_bet(0.55, 0.50, ev=0.1, min_edge=-0.01)

    def test_negative_min_ev_raises(self) -> None:
        with pytest.raises(ValueError):
            should_bet(0.55, 0.50, ev=0.1, min_ev=-0.01)


class TestClassifyBetSide:
    def test_home_side_qualifies(self) -> None:
        # Home: edge 0.05, EV = 0.60 * 0.9 - 0.40 = 0.14.
        result = classify_bet_side(
            model_home_prob=0.60,
            fair_home_prob=0.55,
            home_decimal_odds=1.9,
            away_decimal_odds=2.1,
        )
        assert result["bet_flag"] is True
        assert result["side"] == "home"
        assert math.isclose(result["model_prob"], 0.60)
        assert math.isclose(result["fair_market_prob"], 0.55)
        assert math.isclose(result["edge"], 0.05)
        assert math.isclose(result["expected_value"], 0.14)
        assert math.isclose(result["decimal_odds"], 1.9)

    def test_away_side_qualifies(self) -> None:
        # Away: model 0.60 vs fair 0.55 -> edge 0.05; EV = 0.60 * 1.2 - 0.40 = 0.32.
        result = classify_bet_side(
            model_home_prob=0.40,
            fair_home_prob=0.45,
            home_decimal_odds=2.4,
            away_decimal_odds=2.2,
        )
        assert result["bet_flag"] is True
        assert result["side"] == "away"
        assert math.isclose(result["model_prob"], 0.60)
        assert math.isclose(result["fair_market_prob"], 0.55)
        assert math.isclose(result["edge"], 0.05)
        assert math.isclose(result["expected_value"], 0.32)
        assert math.isclose(result["decimal_odds"], 2.2)

    def test_no_bet_when_model_agrees_with_market(self) -> None:
        result = classify_bet_side(
            model_home_prob=0.50,
            fair_home_prob=0.50,
            home_decimal_odds=1.9,
            away_decimal_odds=1.9,
        )
        assert result["bet_flag"] is False
        assert result["side"] == "no_bet"
        assert result["model_prob"] is None
        assert result["fair_market_prob"] is None
        assert result["edge"] is None
        assert result["expected_value"] is None
        assert result["decimal_odds"] is None

    def test_no_bet_when_edge_exists_but_ev_insufficient(self) -> None:
        # Home edge 0.05 but short odds make EV negative:
        # EV = 0.55 * 0.05 - 0.45 < 0.
        result = classify_bet_side(
            model_home_prob=0.55,
            fair_home_prob=0.50,
            home_decimal_odds=1.05,
            away_decimal_odds=2.0,
        )
        assert result["bet_flag"] is False
        assert result["side"] == "no_bet"

    def test_both_sides_qualifying_raises(self) -> None:
        # Zero thresholds + arbitrage-like odds make both sides "qualify",
        # which must be rejected as inconsistent input.
        with pytest.raises(ValueError):
            classify_bet_side(
                model_home_prob=0.50,
                fair_home_prob=0.50,
                home_decimal_odds=2.1,
                away_decimal_odds=2.1,
                min_edge=0.0,
                min_ev=0.0,
            )

    def test_expected_dict_keys(self) -> None:
        result = classify_bet_side(0.60, 0.55, 1.9, 2.1)
        assert set(result) == {
            "bet_flag",
            "side",
            "model_prob",
            "fair_market_prob",
            "edge",
            "expected_value",
            "decimal_odds",
        }

    @pytest.mark.parametrize("bad_prob", [-0.1, 1.1])
    def test_invalid_probabilities_raise(self, bad_prob: float) -> None:
        with pytest.raises(ValueError):
            classify_bet_side(bad_prob, 0.5, 1.9, 1.9)
        with pytest.raises(ValueError):
            classify_bet_side(0.5, bad_prob, 1.9, 1.9)

    @pytest.mark.parametrize("bad_odds", [1.0, 0.5, -2.0])
    def test_invalid_decimal_odds_raise(self, bad_odds: float) -> None:
        with pytest.raises(ValueError):
            classify_bet_side(0.6, 0.55, bad_odds, 2.0)
        with pytest.raises(ValueError):
            classify_bet_side(0.6, 0.55, 1.9, bad_odds)

    def test_negative_thresholds_raise(self) -> None:
        with pytest.raises(ValueError):
            classify_bet_side(0.6, 0.55, 1.9, 2.1, min_edge=-0.01)
        with pytest.raises(ValueError):
            classify_bet_side(0.6, 0.55, 1.9, 2.1, min_ev=-0.01)
