# Technical Stack

This document defines the technology choices for sports-betting-market-edge and the
justification for each. Every package addition must be justified here before it enters
`requirements.txt`. The guiding principle: look like a serious quantitative research
pipeline without adding complexity before it earns its place.

## Core Data Pipeline — Polars

**Package:** `polars`

Polars is the preferred library for data ingestion, cleaning, joins, feature
engineering, and Parquet storage.

- Use `polars.LazyFrame` where useful for larger datasets (full NBA seasons of
  odds snapshots), so query plans are optimized and memory stays bounded.
- Use `pl.scan_parquet` for lazy Parquet reads rather than eager loads.
- Expression-based transformations make leakage-safe rolling/window features easier
  to express and audit (e.g. explicit `shift` before rolling aggregations).

**Justification:** fast, modern columnar engine with first-class Parquet support and
an API that makes time-ordered feature engineering explicit — directly relevant to the
no-leakage rule.

## pandas — compatibility boundary only

**Package:** `pandas`

pandas is used only when needed for compatibility: scikit-learn interop, matplotlib
plotting, or simple small examples in notebooks. New pipeline modules should not use
pandas as the primary engine. Convert at the model boundary via
`pl.DataFrame.to_pandas()` / `.to_numpy()`.

## Storage — Parquet first

- **Processed datasets:** Parquet (typed, compressed, columnar; lazy-scannable).
- **CSV:** acceptable only for tiny toy sample data and human-readable examples
  (e.g. `data/external/sample_games.csv`).
- **Raw data:** saved exactly as received, before any transformation, into
  `data/raw/` — and never modified afterward (see CLAUDE.md).

## Modeling — scikit-learn

**Package:** `scikit-learn`

v1 models, in order:

1. Market baseline (vig-removed implied probability as the prediction).
2. Logistic regression.
3. Regularized logistic regression.
4. Calibration (`CalibratedClassifierCV`, calibration curves).
5. Evaluation metrics (log loss, Brier score).

**Justification:** interpretable, well-tested baselines; calibration tooling built in.
Probability quality comes before ROI, so the calibration module matters more than
model sophistication at this stage.

## NBA Data — nba_api

**Package:** `nba_api`

Python client for the NBA.com stats endpoints; the games/results source named
in docs/research_plan.md (Section 6). Used only inside collection functions
(lazy import) so the rest of the pipeline has no hard dependency on it; unit
tests never call the live API.

**Justification:** free, complete historical schedule/results coverage, and
the de facto standard client. Returns pandas at the boundary, converted to
Polars immediately.

## Odds Fetching — requests

**Package:** `requests`

Used only inside ``fetch_the_odds_api_historical_snapshot`` (lazy import) to
call The Odds API historical endpoint. Never used in tests; normalization is
fully decoupled from fetching, so the pipeline runs on saved JSON without it.

**Justification:** the standard, minimal HTTP client; one call site.

## Numerics & Plotting

- **`numpy`** — array math underlying scikit-learn and metric computations.
- **`matplotlib`** — calibration curves, backtest equity curves, report figures in
  `reports/figures/`.

## Experiment Tracking — MLflow (planned, not yet added)

MLflow will be adopted once the first real model training script exists. Do not add
the dependency or any integration before then.

When added, every training run should log: model name, feature set version,
train/test date split, hyperparameters, log loss, Brier score, calibration metrics,
ROI/backtest metrics (later), and the model artifact.

**Justification for delay:** tracking infrastructure with zero experiments is
complexity without payoff.

## Testing & Code Quality

- **`pytest`** — all tests; mandatory for every mathematical betting function.
- **`ruff`** — fast linting (and import sorting).
- **`black`** — formatting; no style debates in a portfolio repo.
- **pre-commit** — considered after the core pipeline is stable; not yet added.

## Configuration & Secrets

- **`pyyaml`** — read `config.yaml` (see `config.example.yaml`).
- **`python-dotenv`** — load environment variables from an uncommitted `.env` file;
  API keys come only from environment variables.

## Deliberately Delayed Packages

These are future milestones, not current dependencies. Do not add them until the
baseline model and flat-staking backtest are complete and evaluated:

| Package | Why delayed |
|---|---|
| XGBoost / LightGBM | Gradient boosting only after the logistic baseline sets a calibration bar to beat. |
| Optuna | Hyperparameter search is meaningless before there is a validated model to tune. |
| MLflow | Added with the first training script (see above). |
| Airflow | Orchestration overkill for a batch research pipeline at this scale. |
| Docker | Containerization adds value at deployment/reproducibility milestones, not during early research. |
| Cloud deployment tools | Nothing to deploy yet. |

## Architecture Rules

1. Do not rewrite the already-tested betting math modules (`src/betting/`) unless
   explicitly asked. They are dependency-free, unit-tested, and stable.
2. Do not refactor everything to Polars at once. Migrate opportunistically: new data
   pipeline modules prefer Polars; existing code changes only when touched for
   another reason.
3. Model-facing functions may convert Polars DataFrames to pandas/numpy at the
   scikit-learn boundary. Keep the conversion at the boundary, not scattered
   through the pipeline.
4. Keep the toy sample data simple — small CSVs a human can verify by eye.
5. Every package addition must be justified in this document first.
