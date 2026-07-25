# Changelog — JJ Simon NQ Fair-Price Strategy

All notable changes to this strategy implementation are documented in this file.

## [0.1.0] — 2026-07-25

### Added
- Initial production-ready Python backtest engine in `../jj_quant_backtest/jj_quant_backtest.py`.
- Full JJ Simon Fair-Price state machine:
  - 08:29 scheduled-news anchor.
  - 09:29 NY-open anchor.
  - 01:59 PM-session anchor.
  - Displacement-candle and Break-of-Structure triggers.
  - Unexpected-news anchor reset with 3-candle consolidation zone.
- Two prop-firm risk profiles (50k and 150k) with profile-specific starting balances.
- Dynamic candle rule: switch to 50 pt SL / 75 pt TP and halve size when trigger candle > threshold.
- Full institutional quant suite per config:
  - 20,000-run Monte Carlo (standard and ratchet compounding).
  - 20,000-sample bootstrap Sharpe confidence interval.
  - Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR).
  - Bayesian win-rate posterior.
  - First-order Markov transition analysis.
  - Linear / quadratic / exponential / logarithmic equity-curve regressions.
  - 5-fold chronological walk-forward analysis.
  - Start-of-day to trough drawdown.
- GitHub Actions workflow for distributed chunked execution across NQ, ES, and YM.
- Project silo: `README.md`, `CHANGELOG.md`, `config/profiles.json`, `src/jj_simon_fair_price.py`, `scripts/run_single_backtest.py`, and Pine Script v5 indicator.

### Fixed
- Data path resolution now points to canonical `fvg_execution_engine/backtests/data/<symbol>/M1.csv.gz` instead of missing `gdrive_raw/*_1min.csv.gz` files.
- Timestamp parsing handles both `ts` and `timestamp` columns and converts timezone-aware data to `America/New_York`.
- Profile-specific starting balances ($50k / $150k) to keep R-multiples realistic and prevent Monte-Carlo overflow.

### Notes
- Initial NQ 10-year backtest shows the strategy is currently unprofitable on the tested parameter grid; optimization and walk-forward refinement are next.
