"""Real-data-shape dry run: provider-shaped local samples through the
ingestion path (M6 dry run).

Exercises the full real-data composition — raw nba_api-shaped games CSV,
The Odds API-shaped JSON snapshots, normalization, opening/closing marking,
event matching, validation, market features, and the market baseline — with
LOCAL SAMPLE FILES ONLY. No live API calls, no API keys, no real data.

THE OUTPUT IS NOT A REAL BETTING RESULT. It validates that the pipeline
composes; the numbers mean nothing.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from src.data.collect_nba_games import normalize_nba_games
from src.data.collect_odds import (
    mark_opening_and_closing_lines,
    normalize_the_odds_api_h2h_snapshot,
)
from src.data.match_events import match_odds_to_games, validate_matched_odds
from src.evaluation.model_metrics import evaluate_probability_predictions
from src.features.market_features import add_moneyline_market_features
from src.features.team_game_rows import create_team_game_rows
from src.models.baseline import predict_market_baseline

GAMES_CSV = REPO_ROOT / "data" / "external" / "sample_real_shape_games.csv"
SNAPSHOT_DIR = REPO_ROOT / "data" / "external" / "sample_odds_snapshots"
SNAPSHOT_FILES = ["sample_opening_snapshot.json", "sample_closing_snapshot.json"]


def load_and_normalize_snapshots() -> pl.DataFrame:
    """Normalize each local snapshot into the clean odds schema.

    The clean schema carries ``commence_time`` (M4/M5 integration), so the
    output is directly matchable to games — no raw-JSON re-joins needed.
    """
    frames = [
        normalize_the_odds_api_h2h_snapshot(raw, raw["timestamp"])
        for raw in (
            json.loads((SNAPSHOT_DIR / filename).read_text())
            for filename in SNAPSHOT_FILES
        )
    ]
    return pl.concat(frames)


def main() -> None:
    """Run the dry run and print pipeline counts plus baseline metrics."""
    # 1-2. Raw-shape games CSV -> clean games schema.
    raw_games = pl.read_csv(GAMES_CSV, schema_overrides={"GAME_ID": pl.String})
    games = normalize_nba_games(raw_games)

    # 3-6. Local provider-shaped snapshots -> normalized odds rows.
    odds = load_and_normalize_snapshots()

    # 7. Earliest snapshot per event/book/market = opening; latest = closing.
    odds = mark_opening_and_closing_lines(odds)

    # 8-9. Attach internal game_id (exact date + canonical teams) and verify
    # readiness for feature creation.
    matched = match_odds_to_games(odds, games)
    validate_matched_odds(matched)

    # 10. Market features (implied probs, no-vig fair probs, vig, decimal).
    featured = add_moneyline_market_features(matched)

    # 11. Team-game rows: the project's unit of prediction.
    team_games = create_team_game_rows(games)

    # 12. Closing-line fair probabilities joined onto team-game rows
    # (closing line = market benchmark, per docs/research_plan.md).
    closing = featured.filter(pl.col("is_closing_line"))
    closing_probs = closing.select(
        "game_id", "sportsbook", "home_fair_market_prob", "away_fair_market_prob"
    )
    prediction_frame = team_games.join(closing_probs, on="game_id", how="inner")

    # 13. Market baseline: each team-game row gets its side's fair prob.
    predictions = predict_market_baseline(prediction_frame)

    # 14. Probability-quality metrics at the evaluation boundary.
    metrics = evaluate_probability_predictions(
        y_true=predictions.get_column("team_win").to_list(),
        y_prob=predictions.get_column("predicted_win_prob").to_list(),
    )

    # 15. Summary.
    opening_rows = matched.filter(pl.col("is_opening_line")).height
    closing_rows = matched.filter(pl.col("is_closing_line")).height
    print("Real-data-shape dry run (LOCAL SAMPLE FILES ONLY)")
    print("=" * 64)
    print("DISCLAIMER: local sample data for pipeline validation only.")
    print("This is NOT a real betting result and supports no conclusions")
    print("about models, markets, or profitability.")
    print("-" * 64)
    print(f"games loaded:              {games.height}")
    print(f"normalized odds rows:      {odds.height}")
    print(f"matched odds rows:         {matched.height}")
    print(f"opening line rows:         {opening_rows}")
    print(f"closing line rows:         {closing_rows}")
    print(f"team-game rows:            {team_games.height}")
    print(f"prediction rows:           {predictions.height}")
    print(f"market baseline log loss:  {metrics['log_loss']:.4f}")
    print(f"market baseline Brier:     {metrics['brier_score']:.4f}")
    print(f"market baseline accuracy:  {metrics['accuracy_at_0_5']:.4f}")


if __name__ == "__main__":
    main()
