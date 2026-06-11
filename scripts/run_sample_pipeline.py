"""Toy end-to-end pipeline: raw toy CSVs -> market baseline evaluation.

Structural validation only — proves the modules compose, on synthetic data.
This is not betting analysis and says nothing about real-market performance.

Pipeline: load games + odds -> merge -> moneyline market features ->
team-game rows -> join market probabilities -> market baseline predictions ->
probability-quality metrics.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from src.data.merge_datasets import load_games, load_odds, merge_games_and_odds
from src.evaluation.model_metrics import evaluate_probability_predictions
from src.features.market_features import add_moneyline_market_features
from src.features.team_game_rows import create_team_game_rows
from src.models.baseline import predict_market_baseline

GAMES_CSV = REPO_ROOT / "data" / "external" / "sample_games.csv"
ODDS_CSV = REPO_ROOT / "data" / "external" / "sample_odds.csv"


def main() -> None:
    """Run the toy pipeline and print a summary of counts and metrics."""
    # 1-2. Load toy data (schema-validated on read).
    games = load_games(str(GAMES_CSV))
    odds = load_odds(str(ODDS_CSV))

    # 3. Structural merge check: every odds row must match a game.
    merged = merge_games_and_odds(games, odds)
    assert merged.height == odds.height

    # 4. Market features on game-level odds. Closing lines only: one
    #    snapshot per game per sportsbook, the market's final pre-game view.
    closing_odds = odds.filter(pl.col("is_closing_line") == 1)
    market = add_moneyline_market_features(closing_odds)

    # 5. Team-game rows (two per game, the project's unit of prediction).
    team_games = create_team_game_rows(games)

    # 6. Join game-level market probabilities onto team-game rows by game_id.
    #    Each team-game row pairs with each sportsbook's closing line.
    market_probs = market.select(
        "game_id", "sportsbook", "home_fair_market_prob", "away_fair_market_prob"
    )
    team_games_with_market = team_games.join(market_probs, on="game_id", how="inner")

    # 7. Market baseline: home rows get the home fair prob, away rows the away.
    predictions = predict_market_baseline(team_games_with_market)

    # 8. Probability-quality metrics (lists only at the evaluation boundary).
    metrics = evaluate_probability_predictions(
        y_true=predictions.get_column("team_win").to_list(),
        y_prob=predictions.get_column("predicted_win_prob").to_list(),
    )

    # 9. Summary.
    print("Toy market baseline pipeline (synthetic data — structural check only)")
    print("-" * 68)
    print(f"games:               {games.height}")
    print(f"odds rows:           {odds.height}")
    print(f"team-game rows:      {team_games.height}")
    print(f"prediction rows:     {predictions.height}")
    print(f"log loss:            {metrics['log_loss']:.4f}")
    print(f"brier score:         {metrics['brier_score']:.4f}")
    print(f"accuracy at 0.5:     {metrics['accuracy_at_0_5']:.4f}")


if __name__ == "__main__":
    main()
