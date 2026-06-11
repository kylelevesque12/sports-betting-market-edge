"""Toy model comparison: market baseline vs. logistic regression.

Composition script only — wires together existing modules on synthetic toy
data to validate that the full modeling loop (features -> time split ->
train -> predict -> evaluate) runs end to end.

THE NUMBERS PRINTED HERE ARE NOT MEANINGFUL. The data is synthetic. This
output validates pipeline structure only and supports no claims about model
quality or profitability.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from src.data.merge_datasets import load_games, load_odds, merge_games_and_odds
from src.evaluation.model_metrics import evaluate_probability_predictions
from src.features.market_features import add_moneyline_market_features
from src.features.rolling_stats import add_rolling_team_features
from src.features.team_game_rows import create_team_game_rows
from src.models.baseline import predict_market_baseline
from src.models.logistic_model import predict_logistic_model, train_logistic_model
from src.models.time_split import time_based_split

GAMES_CSV = REPO_ROOT / "data" / "external" / "sample_games.csv"
ODDS_CSV = REPO_ROOT / "data" / "external" / "sample_odds.csv"

ROLLING_FEATURES = [
    "rolling_win_pct_3",
    "rolling_points_for_3",
    "rolling_points_against_3",
    "rolling_point_diff_3",
]
FEATURE_COLS = [*ROLLING_FEATURES, "market_prob", "is_home"]


def choose_split_dates(model_ready: pl.DataFrame) -> tuple[str, str, str]:
    """Pick train/test boundary dates dynamically from the toy data.

    Splits the unique sorted dates roughly 60/40 so both sides are non-empty
    even after null rolling rows were dropped.

    Raises:
        ValueError: If there are too few distinct dates to split.
    """
    dates = sorted(model_ready.get_column("game_date").unique().to_list())
    if len(dates) < 2:
        raise ValueError(
            "Toy data is too small for a time-based split: it has "
            f"{len(dates)} distinct game date(s) after dropping null rolling "
            "features, but a chronological train/test split needs at least 2."
        )
    split_idx = max(len(dates) * 3 // 5, 1)
    return dates[split_idx - 1], dates[split_idx], dates[-1]


def main() -> None:
    """Run the toy comparison and print counts plus metrics for both models."""
    # 1-3. Load toy data; merge as a structural integrity check.
    games = load_games(str(GAMES_CSV))
    odds = load_odds(str(ODDS_CSV))
    merged = merge_games_and_odds(games, odds)
    assert merged.height == odds.height

    # 4-5. Closing lines only, with market-derived features.
    closing = add_moneyline_market_features(
        odds.filter(pl.col("is_closing_line") == 1)
    )

    # 6-7. Team-game rows with leakage-safe rolling features.
    team_games = add_rolling_team_features(create_team_game_rows(games))

    # 8. Join game-level closing-line market probabilities by game_id.
    #    With multiple sportsbooks, each team-game row joins to each book's
    #    closing line, so prediction rows can exceed team-game rows.
    market_probs = closing.select(
        "game_id", "sportsbook", "home_fair_market_prob", "away_fair_market_prob"
    )
    joined = team_games.join(market_probs, on="game_id", how="inner")

    # 9. Side-specific market probability for the logistic model.
    joined = joined.with_columns(
        pl.when(pl.col("is_home") == 1)
        .then(pl.col("home_fair_market_prob"))
        .otherwise(pl.col("away_fair_market_prob"))
        .alias("market_prob")
    )

    # 10. Market baseline predictions (home rows get home fair prob, etc.).
    joined = predict_market_baseline(joined)

    # 11. Drop rows with null rolling features: each team's first game has no
    #     prior history, and the logistic model correctly rejects nulls.
    model_ready = joined.drop_nulls(subset=ROLLING_FEATURES)

    # 12. Time-based split with boundaries chosen from the toy data itself.
    train_end, test_start, test_end = choose_split_dates(model_ready)
    train_df, test_df = time_based_split(
        model_ready,
        date_col="game_date",
        train_end_date=train_end,
        test_start_date=test_start,
        test_end_date=test_end,
    )

    # 13-14. Train logistic regression; predict on the held-out window.
    model = train_logistic_model(train_df, FEATURE_COLS)
    logistic_probs = predict_logistic_model(model, test_df, FEATURE_COLS)

    # 15. Evaluate both models on the same test rows.
    y_true = test_df.get_column("team_win").to_list()
    baseline_metrics = evaluate_probability_predictions(
        y_true, test_df.get_column("predicted_win_prob").to_list()
    )
    logistic_metrics = evaluate_probability_predictions(
        y_true, logistic_probs.tolist()
    )

    # 16-17. Summary.
    print("Toy model comparison: market baseline vs logistic regression")
    print("=" * 66)
    print("NOTE: synthetic toy data — metrics validate pipeline structure")
    print("only. No model-quality or profitability conclusions can be drawn.")
    print("-" * 66)
    print(f"games:                          {games.height}")
    print(f"raw odds rows:                  {odds.height}")
    print(f"closing odds rows:              {closing.height}")
    print(f"team-game rows:                 {team_games.height}")
    print(f"model-ready rows (rolling ok):  {model_ready.height}")
    print(f"train rows (through {train_end}): {train_df.height}")
    print(f"test rows ({test_start} to {test_end}): {test_df.height}")
    print("(prediction rows can exceed team-game rows: one row per")
    print(" team-game per sportsbook closing line)")
    print(f"feature columns: {FEATURE_COLS}")
    print("-" * 66)
    print(f"{'metric':<20} {'market baseline':>16} {'logistic':>12}")
    for key in ("log_loss", "brier_score", "accuracy_at_0_5"):
        print(
            f"{key:<20} {baseline_metrics[key]:>16.4f} {logistic_metrics[key]:>12.4f}"
        )


if __name__ == "__main__":
    main()
