# CLAUDE.md — sports-betting-market-edge

Project rules for all work in this repository. Read before writing any code.

## Project Goal

Build a modular Python research pipeline for identifying **historical** expected-value opportunities in sports betting markets.

This project is for educational, statistical modeling, and portfolio purposes only. It is **not** real-money betting advice, and no output should be framed as such.

## Initial Scope

- **Sport:** NBA
- **Market:** pre-game moneyline
- **Target:** team wins game (binary)
- **Unit of prediction:** one team-game row
- **Core idea:** compare model-estimated win probability to sportsbook implied probability after removing vig
- **Backtesting:** flat 1-unit staking only (no Kelly or variable staking yet)

Do not expand scope (other sports, markets, staking schemes) without explicit direction.

## Hard Rules

### Build incrementally
- Do not build too much at once. Prefer small, testable modules.
- One module, one responsibility. Keep functions small.

### No data leakage
- Every feature must be knowable **before game start**.
- Never use post-game or same-game information in features.
- Use **time-based train/test splits** for final evaluation. Never random splits.

### Modeling priorities
- Probability quality (calibration, log loss, Brier score) comes **before** betting ROI.
- A model with good ROI but poor calibration is not trustworthy; treat it as suspect.

### Data integrity
- Raw data is immutable: once saved, never modify it. Derived/cleaned data goes in separate files or directories.

### Secrets
- Never hard-code API keys. All keys come from environment variables (e.g. `os.environ["..."]`).
- Never commit keys, `.env` files, or credentials.

### Testing
- Every mathematical betting function (odds conversion, vig removal, implied probability, EV, bankroll/staking math) requires unit tests.
- Use **pytest**.

### Claims
- Never make claims of guaranteed profitability — in code comments, docs, README, or output.

## Technical Stack

Full rationale lives in `docs/tech_stack.md`. Every new package must be justified there before being added to `requirements.txt`.

- **Data pipeline:** prefer **Polars** for ingestion, cleaning, joins, feature engineering, and Parquet storage. Use `LazyFrame`/`scan_parquet` where useful. pandas only for scikit-learn/plotting compatibility or tiny examples.
- **Storage:** Parquet for processed datasets. CSV only for tiny toy/sample data. Raw data saved before any transformation.
- **Modeling:** scikit-learn for v1 (market baseline, logistic regression, regularized logistic regression, calibration, metrics).
- **Experiment tracking:** MLflow planned, but not added until the first real model training script exists.
- **Quality:** pytest, ruff, black. pre-commit considered once the core pipeline is stable.
- **Delayed:** no XGBoost, LightGBM, Optuna, Airflow, Docker, or cloud tooling yet — future milestones after baseline model + backtest.

Architecture rules:
- Do not rewrite the tested betting math modules unless explicitly asked.
- Do not refactor everything to Polars at once — new data modules prefer Polars; convert to pandas/numpy only at the scikit-learn boundary.
- Keep toy sample data simple.

## Coding Style

- Python 3.11+
- Type hints where reasonable.
- Clear docstrings for public functions.
- Keep functions small.
- Notebooks are for exploration and explanation only — core logic lives in importable modules, never in notebooks.
