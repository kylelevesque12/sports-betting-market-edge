# Research Plan — Real-Data Phase (Pre-Registered)

Pre-registered research design for the real-data phase of
sports-betting-market-edge, written before any real data ingestion. Agents
read this before any research-phase task. Decisions recorded here are binding
until explicitly revised; revisions are committed with reasoning so the
decision history stays auditable.

This is an educational research project. Nothing in this plan or its outputs
is betting advice, and no result will be framed as a claim of profitability.
A backtest showing large ROI is, by default, treated as evidence of leakage
or overfitting until it survives every check in Sections 10–11.

## 1. Project Status

Built and tested so far (214 passing tests):

- **Betting math** (`src/betting/`): odds conversion, vig removal, expected
  value, edge, and threshold-based bet selection — all unit-tested.
- **Data** (`src/data/`): toy CSV loaders and a validated games/odds merge.
- **Features** (`src/features/`): team-game row transformation, market
  features (no-vig probabilities, vig, decimal odds), and leakage-safe
  shifted rolling features.
- **Models** (`src/models/`): no-training market baseline, time-based
  train/test splitting, and a logistic regression with feature-order
  guarding.
- **Evaluation** (`src/evaluation/`): log loss, Brier score, accuracy at 0.5.
- **Backtesting** (`src/backtesting/`): flat-stake settlement (P&L, ROI,
  win rate, average odds).
- **Scripts** (`scripts/`): two composition scripts proving the full loop
  runs end to end on toy data.

All metrics produced to date are from synthetic toy data (10 fake games,
2 fake sportsbooks). **They are structurally informative and numerically
meaningless.** No conclusion about models or markets can be drawn from them.

## 2. V1 Research Question

> Can model-estimated probabilities improve on market-implied probabilities
> for NBA pre-game moneyline outcomes in a leakage-safe historical framework?

Scope is limited to the NBA pre-game moneyline. No other sports, markets,
bet types, or staking schemes. The question is deliberately not "can we
predict winners" — the closing moneyline is among the most efficient prices
in sports, and the project treats it as the information-rich benchmark to
measure against, not an opponent to out-predict. A rigorous negative result
("the market prices everything we measured, except possibly X") is a
successful outcome of this phase.

## 3. Unit Definitions

- **Modeling unit: one team-game row.** Win probability is a property of a
  team in a game; each game contributes a home row and an away row with
  `team_win` as the target.
- **Odds unit: one game-book-timestamp-market row.** A price is a fact about
  one sportsbook's market at one moment. Books disagree and lines move, so
  odds cannot be stored at game grain without destroying exactly the
  information (movement, cross-book disagreement) the research question
  needs.
- **Betting/backtest unit: one team-game-book-timestamp row.** A simulated
  bet is a team-game prediction joined to one specific available price at
  one specific time. The same model opinion can be a bet at one book's price
  and a pass at another's.

These differ because they answer different questions: the model estimates a
probability (team-game), the market quotes prices (game-book-timestamp), and
a bet is the join of the two. Collapsing any of these grains loses
information the evaluation ladder requires.

## 4. Market Timing Rules

These rules are binding and violations invalidate results.

1. **Closing line = market benchmark.** All probability-quality comparisons
   (Section 10) measure the model against no-vig closing probabilities.
2. **Opening line or fixed pre-game snapshot = simulated bet price.** Bets
   are simulated at prices that actually existed at the simulated bet time.
   You cannot bet a price that has already closed.
3. **Closing odds are never model input when simulating bets placed before
   close.** Closing lines appear only as the benchmark and in (future)
   closing-line-value computation.
4. **No odds captured after the simulated bet time may be used as a
   feature.** Enforcement lives in training/simulation scripts:
   `market_features` computes features for whatever odds snapshot it is
   given, so the caller owns snapshot selection and must filter by timestamp
   before featurization.

## 5. Data Requirements

- **Historical NBA game results:** game_id, season, game_date, home/away
  teams, final scores, home_win. Target coverage: 8–15 seasons
  (~10–18k games).
- **Historical team-level performance data:** derivable from results at
  first (rolling win pct, point differential via existing modules);
  box-score-based ratings (offensive/defensive/net) are a later ingestion
  pass.
- **Historical sportsbook moneyline odds:** one row per
  game-book-timestamp-market, American odds for both sides.
- **Opening lines:** required — they are the simulated bet price.
- **Closing lines:** required — they are the benchmark.
- **Optional fixed pre-game snapshots** (e.g. 24h/12h/4h/1h before tip):
  adopted if and when a snapshot-capable source is justified.
- **Team mapping table:** canonical team IDs mapping every name variant,
  abbreviation, and relocation/rename (e.g. NJN→BKN, SEA→OKC, CHA/CHO
  history) across games and odds sources, with effective season ranges.
- **Date and timestamp parsing rules:** `game_date` parsed to `pl.Date`;
  odds timestamps parsed to timezone-aware `pl.Datetime` (UTC internally);
  source-local timezones converted at ingestion and documented per source.

## 6. Data Source Candidates

- **NBA games/stats:** `nba_api` (Python client for the NBA.com endpoints),
  free, full historical coverage. FiveThirtyEight's public historical Elo
  file as a cross-check on results data.
- **Historical odds, in adoption order:**
  1. Sportsbook Reviews Online archives (free, ~2007+, opening and closing).
  2. Kaggle historical NBA odds datasets (free; quality varies — only
     adopted where cross-validated against SBR).
  3. aussportsbetting.com closing-line workbook (free; cross-check).
  4. The Odds API (paid; point-in-time snapshots from mid-2020, multiple
     books) — adopted only if free-data results justify snapshot-grain work.
- **Fallback plan if historical odds API access is unavailable:** begin with
  a small saved historical odds sample committed under `data/external/`
  (same schema as the toy data), while keeping the ingestion interface
  API-ready — loaders take a source-agnostic schema so swapping a saved
  sample for an API client changes no downstream code.

## 7. Data Validation Rules

Every ingestion module implements these, with tests:

- Parse `game_date` as a date; parse odds timestamps as datetimes.
- Reject null dates and null timestamps.
- Reject null odds.
- Reject invalid American odds: |odds| < 100 is impossible and rejected;
  |odds| > 2000 is quarantined for manual review (defense against the
  single-outlier failure mode documented in published betting-research
  corrections).
- Validate team IDs/names against the team mapping table; unmapped names
  block ingestion rather than passing through.
- Validate every odds row maps to exactly one game (anti-join check, as in
  `merge_datasets`).
- Validate no duplicate game/book/timestamp/market rows.
- Validate opening and closing lines are clearly and mutually exclusively
  labeled, with exactly one closing line per game-book-market.
- Validate model features are known before simulated bet time (timestamp
  comparison, not trust).
- Cross-source agreement where sources overlap: closing lines must agree
  within tolerance; systematic disagreement blocks adoption of the worse
  source.
- Raw downloads saved untouched to `data/raw/` before any cleaning
  (CLAUDE.md immutability rule).

## 8. Leakage Rules

- No current-game stats as features (scores, box stats of the game being
  predicted).
- Rolling features must be shifted: the current game is excluded from its
  own windows (`shift(1)` pattern already enforced in `rolling_stats.py`).
- Future games cannot affect current features — no centered windows, no
  whole-season aggregates applied retroactively.
- Random train/test splits are not allowed for any final evaluation;
  time-based splits only (`time_split.py`).
- Closing lines cannot be used as features for opening-line betting
  simulations (Section 4, rule 3).
- Bet-filter thresholds (`min_edge`, `min_ev`) are fixed before test-set
  evaluation and recorded in this document's revision history. Tuning
  thresholds on test ROI is prohibited.

## 9. Modeling Plan

In order, each step gated by the previous:

1. **Market baseline** from no-vig closing probabilities
   (`predict_market_baseline`) — the bar all models must beat.
2. **Logistic regression** (`logistic_model.py`) on leakage-safe features,
   including the pre-bet-time market probability as a feature. Models are
   studied through their residuals: if non-market coefficients shrink to
   zero, the market already prices those features — a finding, not a
   failure.
3. **Regularized logistic regression** (explicit L1/L2 tuning on
   training-period data only) later.
4. **Calibration** (CalibratedClassifierCV, reliability curves) later, after
   real data exists — calibration on toy samples is theater.
5. **MLflow** adopted at the first real-data model training script, per
   docs/tech_stack.md, logging model name, feature set version, split dates,
   parameters, log loss, Brier, calibration metrics, and artifacts.

Capacity discipline: ~6–15 features; every feature requires a one-sentence
causal story in its docstring before entering a model; walk-forward
coefficient stability is checked, and sign-flipping features are removed as
noise.

## 10. Evaluation Plan

Probability quality strictly precedes betting metrics (CLAUDE.md).

- **Predictive metrics:** log loss, Brier score, accuracy at 0.5
  (`evaluate_probability_predictions`); calibration curves later, once real
  data exists.
- **Market comparison:** model vs. the no-vig closing-line baseline on
  identical test rows, walk-forward by season. A model that does not at
  least match the baseline's log loss does not proceed to betting
  evaluation.
- **Betting metrics** (only after the above): total bets, total profit, ROI,
  win rate, average odds (`run_flat_stake_backtest`); maximum drawdown
  later; closing line value (bet price vs. closing price per flagged bet)
  later — CLV is the primary skill signal once implemented, because it
  detects real edge at far smaller samples than ROI significance requires.

All ROI results are reported with uncertainty (bootstrap confidence
intervals) and broken out by season, sportsbook, and odds bucket
(favorite/underdog ranges) where sample sizes allow. Results
indistinguishable from zero are reported as such.

## 11. Backtesting Rules

- Flat 1-unit staking only. No Kelly or variable staking in this phase.
- Bet-filter thresholds fixed before test evaluation (Section 8).
- Bets priced at opening/snapshot lines only (Section 4).
- Report results by season if enough data exists; a single pooled number is
  never the headline.
- Do not claim profitability from small samples. At flat stakes, ROI noise
  bands are wide; hundreds of bets distinguish little. Conclusions are
  worded against the confidence interval, not the point estimate.

## 12. Implementation Milestones

In order. Each is one small, testable task; none begins until the previous
is reviewed.

**M1 — Team/date schema hardening**
Agent: feature-engineering (with code-review).
Files: `src/data/merge_datasets.py` tests, possibly a new
`src/data/schema.py`; touched modules' tests.
Done when: `game_date` is `pl.Date` end to end, null dates/odds rejected
loudly, and standing dtype advisories from past reviews are closed.

**M2 — Team mapping module**
Agent: data.
Files: `src/data/team_mapping.py`, `tests/test_team_mapping.py`,
`data/external/team_mapping.csv`.
Done when: canonical IDs cover all 30 franchises with historical
relocations/renames and effective season ranges; unmapped names raise;
round-trip tested against both a games source and an odds source name list.

**M3 — Real NBA game ingestion**
Agent: data.
Files: `src/data/ingest_games.py`, `tests/test_ingest_games.py`.
Done when: 8+ seasons of game results saved raw to `data/raw/`, cleaned to
Parquet in `data/processed/` via Polars, passing all Section 7 validations;
row counts reconciled against known games-per-season.

**M4 — Real odds ingestion or saved odds sample loader**
Agent: data.
Files: `src/data/ingest_odds.py`, `tests/test_ingest_odds.py`.
Done when: historical odds (SBR-derived or saved sample per the Section 6
fallback) stored raw, cleaned to one game-book-timestamp-market row,
opening/closing labeled, all Section 7 validations passing; interface is
API-ready (schema-stable regardless of source).

**M5 — Real data merge validation**
Agent: data (with code-review).
Files: extensions to `src/data/merge_datasets.py` tests or a
`scripts/validate_real_merge.py`.
Done when: every odds row maps to exactly one game across the full sample;
orphan/duplicate/timestamp violations are zero or quarantined with counts
reported.

**M6 — Real market baseline evaluation**
Agent: modeling + evaluation.
Files: `scripts/run_market_baseline_real.py`.
Done when: no-vig closing baseline evaluated walk-forward by season on real
data (log loss, Brier, accuracy); this is the project's first real result
and the Layer-1/Layer-2 reference numbers.

**M7 — Logistic model on real data**
Agent: modeling.
Files: `scripts/train_logistic_real.py` (MLflow adopted here per
tech_stack.md).
Done when: walk-forward logistic results vs. baseline reported with
coefficient stability across folds; features limited to schedule/rolling +
pre-bet-time market probability.

**M8 — Flat-stake backtest on real data**
Agent: backtesting + evaluation.
Files: `scripts/backtest_real.py`.
Done when: bets simulated at opening prices with pre-registered thresholds,
settled via `run_flat_stake_backtest`, reported by season/book/odds bucket
with bootstrap intervals; no profitability claims beyond what the intervals
support.

**M9 — README/results update**
Agent: documentation.
Files: `README.md`, `reports/`.
Done when: methodology and results documented with the responsible-use
framing intact; toy metrics clearly separated from real results.

## 13. Deferred Features

Explicitly out of scope for this phase, revisited only after M1–M9:

- XGBoost / LightGBM (only after the logistic baseline sets a calibration
  bar worth beating)
- Optuna hyperparameter search
- MLflow integration (deferred until M7, not beyond)
- Calibration models (deferred until real data exists; then prioritized)
- Injury / player availability data (highest leakage risk in the project;
  added only with as-known-at-bet-time timestamp discipline, and only if M8
  residuals justify it)
- Player props and other markets
- Live / in-play betting
- Kelly or variable staking
- Docker / deployment / orchestration tooling
