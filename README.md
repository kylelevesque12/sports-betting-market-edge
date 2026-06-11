# sports-betting-market-edge

A modular Python research pipeline for identifying **historical** expected-value opportunities in sports betting markets. The core idea: estimate a team's win probability with a statistical model, compare it to the sportsbook's vig-removed implied probability, and backtest where the model and market historically disagreed.

## Responsible Use

This project exists for educational, statistical modeling, and portfolio purposes only. It is **not** betting advice, makes **no claims of profitability**, and should not be used to place real-money wagers. Historical results do not predict future outcomes.

## Current Scope (v1)

- **Sport:** NBA
- **Market:** pre-game moneyline
- **Target:** team wins game (binary)
- **Unit of prediction:** one team-game row
- **Backtesting:** flat 1-unit staking only
