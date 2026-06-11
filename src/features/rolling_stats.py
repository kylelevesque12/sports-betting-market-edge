"""Leakage-safe rolling team features from team-game rows.

Pre-game safety: every rolling feature is computed with ``shift(1)`` applied
*before* the rolling mean, within each team's chronologically sorted games.
The current game's result is therefore never included in its own features —
each row sees only games strictly before it. A team's first game has null
rolling features by construction; models must handle or drop those rows.
"""

from collections.abc import Sequence

import polars as pl

REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "team",
    "game_date",
    "points_for",
    "points_against",
    "point_diff",
    "team_win",
)

# (input column, output feature stem)
_ROLLING_STATS: tuple[tuple[str, str], ...] = (
    ("team_win", "rolling_win_pct"),
    ("points_for", "rolling_points_for"),
    ("points_against", "rolling_points_against"),
    ("point_diff", "rolling_point_diff"),
)


def add_rolling_team_features(
    team_games: pl.DataFrame,
    windows: Sequence[int] = (3, 5),
) -> pl.DataFrame:
    """Append pre-game rolling means of team results for each window.

    For each team (sorted by ``game_date``), each feature is the mean of that
    stat over the team's previous ``window`` games, **excluding the current
    game** (``shift(1)`` before the rolling mean). Fewer than ``window`` prior
    games is allowed — the mean uses what exists; zero prior games yields null.

    Args:
        team_games: One row per team-game with the required input columns.
        windows: Rolling window sizes in games, each >= 1. Defaults to (3, 5),
            producing e.g. ``rolling_win_pct_3`` and ``rolling_win_pct_5``.

    Returns:
        The input DataFrame plus ``rolling_win_pct_{w}``,
        ``rolling_points_for_{w}``, ``rolling_points_against_{w}``, and
        ``rolling_point_diff_{w}`` for each window ``w``, sorted by
        ``game_date`` then ``team``.

    Raises:
        ValueError: If required input columns are missing, ``windows`` is
            empty, or any window is less than 1.
    """
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in team_games.columns]
    if missing:
        raise ValueError(f"team_games is missing required columns: {missing}")

    if len(windows) == 0:
        raise ValueError("windows must contain at least one window size.")
    bad_windows = [w for w in windows if w < 1]
    if bad_windows:
        raise ValueError(f"All windows must be >= 1, got: {bad_windows}")

    # Chronological order within each team is what makes shift(1) mean
    # "previous game" — sort before computing.
    sorted_games = team_games.sort(["team", "game_date"])

    feature_exprs = [
        pl.col(source)
        .shift(1)  # exclude the current game: leakage guard
        .rolling_mean(window_size=window, min_samples=1)
        .over("team")
        .alias(f"{stem}_{window}")
        for window in windows
        for source, stem in _ROLLING_STATS
    ]

    return sorted_games.with_columns(feature_exprs).sort(["game_date", "team"])
