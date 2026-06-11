"""Probability-quality metrics for win-probability predictions.

Per CLAUDE.md, probability quality (log loss, Brier score, calibration)
comes before betting ROI. These metrics are the first gate for every model,
starting with the market baseline. ROI metrics live elsewhere and are
meaningful only alongside these.
"""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


def _validate_inputs(y_true: Sequence, y_prob: Sequence) -> tuple[np.ndarray, np.ndarray]:
    true_arr = np.asarray(y_true, dtype=object)
    prob_arr = np.asarray(y_prob, dtype=object)

    if true_arr.size == 0 or prob_arr.size == 0:
        raise ValueError("y_true and y_prob must be non-empty.")
    if true_arr.shape != prob_arr.shape:
        raise ValueError(
            f"y_true and y_prob must have the same length, "
            f"got {true_arr.size} and {prob_arr.size}."
        )

    for name, arr in (("y_true", true_arr), ("y_prob", prob_arr)):
        if any(v is None for v in arr.ravel()):
            raise ValueError(f"{name} contains null values.")

    true_f = true_arr.astype(float)
    prob_f = prob_arr.astype(float)

    if np.isnan(true_f).any():
        raise ValueError("y_true contains null values.")
    if np.isnan(prob_f).any():
        raise ValueError("y_prob contains null values.")

    if not np.isin(true_f, (0.0, 1.0)).all():
        bad = sorted(set(true_f[~np.isin(true_f, (0.0, 1.0))].tolist()))
        raise ValueError(f"y_true must contain only 0 and 1, found: {bad}")
    if ((prob_f < 0) | (prob_f > 1)).any():
        bad = sorted(set(prob_f[(prob_f < 0) | (prob_f > 1)].tolist()))
        raise ValueError(f"y_prob must be between 0 and 1, found: {bad}")

    return true_f.astype(int), prob_f


def evaluate_probability_predictions(
    y_true: Sequence, y_prob: Sequence
) -> dict[str, float]:
    """Score predicted win probabilities against binary outcomes.

    Args:
        y_true: Array-like of binary outcomes (0 or 1), one per team-game row.
        y_prob: Array-like of predicted win probabilities in [0, 1].

    Returns:
        Dict with ``log_loss``, ``brier_score``, and ``accuracy_at_0_5``
        (fraction correct when classifying ``y_prob >= 0.5`` as a win).

    Raises:
        ValueError: If lengths differ, inputs are empty or contain nulls,
            ``y_true`` is not strictly 0/1, or ``y_prob`` is outside [0, 1].
    """
    true_arr, prob_arr = _validate_inputs(y_true, y_prob)

    predictions_at_0_5 = (prob_arr >= 0.5).astype(int)

    return {
        # labels=[0, 1] keeps log_loss defined when y_true has a single class.
        "log_loss": float(log_loss(true_arr, prob_arr, labels=[0, 1])),
        "brier_score": float(brier_score_loss(true_arr, prob_arr)),
        "accuracy_at_0_5": float(np.mean(predictions_at_0_5 == true_arr)),
    }
