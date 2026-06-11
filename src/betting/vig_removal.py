"""Removal of sportsbook vig from implied probabilities."""


def remove_vig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Remove the vig from a two-way market by proportional normalization.

    The two implied probabilities of a two-way market (e.g. moneyline home/away)
    sum to more than 1 because of the sportsbook's margin. This rescales them
    so they sum to exactly 1.

    Args:
        prob_a: Implied probability of outcome A, strictly between 0 and 1.
        prob_b: Implied probability of outcome B, strictly between 0 and 1.

    Returns:
        Tuple of fair (no-vig) probabilities ``(fair_a, fair_b)`` summing to 1.

    Raises:
        ValueError: If either probability is not strictly between 0 and 1.
    """
    for name, prob in (("prob_a", prob_a), ("prob_b", prob_b)):
        if not 0 < prob < 1:
            raise ValueError(f"{name} must be strictly between 0 and 1, got {prob}.")
    total = prob_a + prob_b
    return prob_a / total, prob_b / total
