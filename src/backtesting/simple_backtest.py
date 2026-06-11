"""Flat-stake backtest: mechanical P&L over already-decided historical bets.

This module settles bets; it does not select them. Bet selection (edge/EV
thresholds) happens upstream in ``src.betting.bet_filter``, and predictions
come from the modeling pipeline. Flat 1-unit staking only (v1 scope).

Results are historical research output on whatever sample was provided.
They are not betting advice and support no claims about future profit.
"""

import polars as pl

REQUIRED_COLUMNS: tuple[str, ...] = ("team_win", "bet_flag", "decimal_odds")


def run_flat_stake_backtest(
    predictions: pl.DataFrame,
    stake: float = 1.0,
) -> dict:
    """Settle flat-stake bets and summarize historical P&L.

    Only rows with ``bet_flag == True`` are settled. A winning bet
    (``team_win == 1``) returns ``stake * (decimal_odds - 1)``; a losing bet
    loses the stake.

    Args:
        predictions: One row per potential bet with ``team_win`` (0/1),
            ``bet_flag`` (boolean), and ``decimal_odds``.
        stake: Units risked per bet. Must be positive. Defaults to 1.0.

    Returns:
        Dict with ``total_bets``, ``total_staked``, ``total_profit``,
        ``roi``, ``win_rate``, and ``average_odds``. With zero bets, the
        numeric metrics are 0 and ``average_odds`` is None.

    Raises:
        ValueError: If required columns are missing or contain nulls,
            ``stake`` is not positive, ``bet_flag`` is not boolean,
            ``team_win`` has values other than 0/1, or any placed bet has
            ``decimal_odds <= 1``.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in predictions.columns]
    if missing:
        raise ValueError(f"predictions is missing required columns: {missing}")

    if stake <= 0:
        raise ValueError(f"stake must be positive, got {stake}.")

    if predictions.schema["bet_flag"] != pl.Boolean:
        raise ValueError(
            f"bet_flag must be a boolean column, got dtype "
            f"{predictions.schema['bet_flag']}."
        )

    for col in REQUIRED_COLUMNS:
        null_count = predictions.get_column(col).null_count()
        if null_count > 0:
            raise ValueError(f"column {col!r} contains {null_count} null value(s).")

    bad_wins = predictions.filter(~pl.col("team_win").is_in([0, 1]))
    if bad_wins.height > 0:
        bad = bad_wins.get_column("team_win").unique().to_list()
        raise ValueError(f"team_win must contain only 0 and 1, found: {bad}")

    bets = predictions.filter(pl.col("bet_flag"))

    bad_odds = bets.filter(pl.col("decimal_odds") <= 1)
    if bad_odds.height > 0:
        bad = bad_odds.get_column("decimal_odds").unique().to_list()
        raise ValueError(
            f"decimal_odds must be greater than 1 for placed bets, found: {bad}"
        )

    if bets.height == 0:
        return {
            "total_bets": 0,
            "total_staked": 0.0,
            "total_profit": 0.0,
            "roi": 0.0,
            "win_rate": 0.0,
            "average_odds": None,
        }

    settled = bets.with_columns(
        pl.when(pl.col("team_win") == 1)
        .then(stake * (pl.col("decimal_odds") - 1.0))
        .otherwise(-stake)
        .alias("profit")
    )

    total_bets = settled.height
    total_staked = float(total_bets * stake)
    total_profit = float(settled.get_column("profit").sum())
    wins = settled.filter(pl.col("team_win") == 1).height

    return {
        "total_bets": total_bets,
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": total_profit / total_staked,
        "win_rate": wins / total_bets,
        "average_odds": float(settled.get_column("decimal_odds").mean()),
    }
