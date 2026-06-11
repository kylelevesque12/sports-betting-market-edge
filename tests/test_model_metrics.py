"""Unit tests for src.evaluation.model_metrics."""

import math

import numpy as np
import pytest

from src.evaluation.model_metrics import evaluate_probability_predictions


class TestMetricValues:
    def test_perfect_predictions_near_zero_losses(self) -> None:
        result = evaluate_probability_predictions([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0])
        assert result["log_loss"] < 1e-6
        assert result["brier_score"] < 1e-12
        assert result["accuracy_at_0_5"] == 1.0

    def test_known_brier_score(self) -> None:
        # ((0.8-1)^2 + (0.2-0)^2 + (0.6-1)^2) / 3 = (0.04 + 0.04 + 0.16) / 3
        result = evaluate_probability_predictions([1, 0, 1], [0.8, 0.2, 0.6])
        assert math.isclose(result["brier_score"], 0.08)

    def test_uninformative_coin_flip(self) -> None:
        result = evaluate_probability_predictions([0, 1], [0.5, 0.5])
        assert math.isclose(result["brier_score"], 0.25)
        assert math.isclose(result["log_loss"], math.log(2), rel_tol=1e-9)

    def test_single_class_outcomes_supported(self) -> None:
        # All wins: log_loss must remain defined (labels pinned to [0, 1]).
        result = evaluate_probability_predictions([1, 1, 1], [0.9, 0.8, 0.7])
        assert result["log_loss"] > 0
        assert result["accuracy_at_0_5"] == 1.0


class TestAccuracyAtHalf:
    def test_threshold_is_inclusive_at_0_5(self) -> None:
        # 0.5 classifies as 1.
        result = evaluate_probability_predictions([1, 0], [0.5, 0.5])
        assert math.isclose(result["accuracy_at_0_5"], 0.5)

    def test_mixed_correct_and_incorrect(self) -> None:
        # Predictions at 0.5 threshold: [1, 0, 1, 0] vs truth [1, 0, 0, 1] -> 2/4.
        result = evaluate_probability_predictions(
            [1, 0, 0, 1], [0.9, 0.1, 0.7, 0.3]
        )
        assert math.isclose(result["accuracy_at_0_5"], 0.5)

    def test_accepts_numpy_arrays(self) -> None:
        result = evaluate_probability_predictions(
            np.array([1, 0]), np.array([0.8, 0.2])
        )
        assert math.isclose(result["accuracy_at_0_5"], 1.0)


class TestValidation:
    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            evaluate_probability_predictions([0, 1], [0.5])

    @pytest.mark.parametrize("bad_true", [[0, 2], [0.5, 1], [-1, 0]])
    def test_non_binary_y_true_raises(self, bad_true: list) -> None:
        with pytest.raises(ValueError, match="only 0 and 1"):
            evaluate_probability_predictions(bad_true, [0.5, 0.5])

    @pytest.mark.parametrize("bad_prob", [[-0.1, 0.5], [0.5, 1.1]])
    def test_out_of_range_probabilities_raise(self, bad_prob: list) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            evaluate_probability_predictions([0, 1], bad_prob)

    def test_empty_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            evaluate_probability_predictions([], [])

    def test_null_in_y_true_raises(self) -> None:
        with pytest.raises(ValueError, match="y_true contains null"):
            evaluate_probability_predictions([0, None], [0.5, 0.5])

    def test_null_in_y_prob_raises(self) -> None:
        with pytest.raises(ValueError, match="y_prob contains null"):
            evaluate_probability_predictions([0, 1], [0.5, None])

    def test_nan_in_y_prob_raises(self) -> None:
        with pytest.raises(ValueError, match="y_prob contains null"):
            evaluate_probability_predictions([0, 1], [0.5, float("nan")])
