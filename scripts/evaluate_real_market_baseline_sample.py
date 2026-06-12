"""First real-data market baseline evaluation, on the tiny matched sample.

Evaluates the sportsbook closing-line market baseline (vig-removed fair
probabilities) against real outcomes for the matched odds sample. This is
the project's first metric computed on real games and real odds — but on a
TINY sample. It validates the evaluation path; the numbers carry no
statistical weight, and no profitability or model-quality claims follow
from them.

No training, no betting strategy, no threshold logic.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from src.evaluation.model_metrics import evaluate_probability_predictions
from src.features.market_features import add_moneyline_market_features
from src.features.team_game_rows import create_team_game_rows
from src.models.baseline import predict_market_baseline

GAMES_PATH = REPO_ROOT / "data" / "processed" / "nba_games.parquet"
MATCHED_ODDS_PATH = REPO_ROOT / "data" / "processed" / "matched_odds_sample.parquet"


def load_inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load both processed inputs, failing clearly if either is missing."""
    if not GAMES_PATH.exists():
        raise SystemExit(
            f"ERROR: {GAMES_PATH.relative_to(REPO_ROOT)} not found. Run "
            "scripts/collect_real_nba_games.py first."
        )
    if not MATCHED_ODDS_PATH.exists():
        raise SystemExit(
            f"ERROR: {MATCHED_ODDS_PATH.relative_to(REPO_ROOT)} not found. "
            "Run scripts/match_real_odds_sample.py first."
        )
    return pl.read_parquet(GAMES_PATH), pl.read_parquet(MATCHED_ODDS_PATH)


def main() -> None:
    """Evaluate the closing-line baseline on the matched real sample."""
    games, matched_odds = load_inputs()

    closing = matched_odds.filter(pl.col("is_closing_line"))
    if closing.height == 0:
        raise SystemExit(
            "ERROR: matched odds sample contains no closing lines; cannot "
            "evaluate the closing-line baseline."
        )

    # Market features on closing lines (vig-removed fair probabilities).
    featured = add_moneyline_market_features(closing)

    # Team-game rows for the games the sample covers.
    team_games = create_team_game_rows(games)

    # One row per team-game per sportsbook closing line.
    closing_probs = featured.select(
        "game_id", "sportsbook", "home_fair_market_prob", "away_fair_market_prob"
    )
    prediction_frame = team_games.join(closing_probs, on="game_id", how="inner")
    if prediction_frame.height == 0:
        raise SystemExit(
            "ERROR: no prediction rows produced — the matched odds game_ids "
            "do not appear in the games table."
        )

    predictions = predict_market_baseline(prediction_frame)

    # Hard checks before metrics.
    for col in ("predicted_win_prob", "team_win"):
        if predictions.get_column(col).null_count() > 0:
            raise SystemExit(f"ERROR: null values in {col!r} prediction rows.")
    out_of_range = predictions.filter(
        (pl.col("predicted_win_prob") < 0) | (pl.col("predicted_win_prob") > 1)
    )
    if out_of_range.height > 0:
        raise SystemExit("ERROR: predicted_win_prob outside [0, 1].")

    metrics = evaluate_probability_predictions(
        y_true=predictions.get_column("team_win").to_list(),
        y_prob=predictions.get_column("predicted_win_prob").to_list(),
    )

    books = sorted(closing.get_column("sportsbook").unique().to_list())
    print("Real market baseline evaluation — TINY SAMPLE")
    print("=" * 66)
    print("DISCLAIMER: tiny real-data sample; pipeline validation only.")
    print("These numbers carry no statistical weight. No profitability or")
    print("model-quality claims can be made from this output.")
    print("-" * 66)
    print(f"games loaded:             {games.height}")
    print(f"matched odds rows:        {matched_odds.height}")
    print(f"closing odds rows used:   {closing.height}")
    print(f"sportsbooks found:        {', '.join(books)}")
    print(f"team-game rows:           {team_games.height}")
    print(f"prediction rows:          {predictions.height}")
    print(f"unique games evaluated:   {predictions.get_column('game_id').n_unique()}")
    print(f"log loss:                 {metrics['log_loss']:.4f}")
    print(f"Brier score:              {metrics['brier_score']:.4f}")
    print(f"accuracy at 0.5:          {metrics['accuracy_at_0_5']:.4f}")


if __name__ == "__main__":
    main()
