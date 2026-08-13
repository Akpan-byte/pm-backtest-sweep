# Polymarket Trading Bot Clean Rebuild

## Objective
Clean rebuild of the Polymarket live trading bot from existing source files (`shadow_paper_bot.py`, `FINAL_GOLDEN_BOT.py`, `ULTIMATE_BOT_AUDITED.py`).

## Architecture
- `bot/signals.py` — 17 signal generators
- `bot/lifecycle.py` — Trade lifecycle management
- `bot/market_data.py` — Market data fetching
- `bot/config.py` — Configuration
- `bot/main.py` — Main wiring

## 17 Signals
1. SNIPE
2. BREAKOUT_PCT_0.04
3. BREAKOUT_PCT_0.08
4. BREAKOUT_Z_1.6
5. KINETIC_VELOCITY_BREAKOUT
6. L2_ABSORPTION_SPREAD_COLLAPSE
7. LIQUIDATION_SPOT_GAP_FADE
8. MR_GAMMA_EXPIRY_PIN
9. MR_L2_OFI_DELTA_FADE
10. MEAN_REVERSION
11. MEAN_REVERSION_Z_1.5
12. MEAN_REVERSION_OPPOSITE_EXIT
13. BREAKOUT_PCT_0.03
14. BREAKOUT_Z_1.5
15. BREAKOUT_Z_3.0
16. BREAKOUT_PCT_0.06
17. BREAKOUT_Z_2.0

## Stacks
- 5-MIN STACK ($100): 12 strategies
- 15-MIN STACK ($100): 6 strategies
- COMBINED ($200 shared): All 17

## Data Sources
- `/config/projects/trading/data/poly-data/poly_data/btc_polymarket_ticks.csv`
- `/config/projects/trading/data/poly-data/poly_data_elite_7-17/*/btc_polymarket_ticks.csv`
- `/config/projects/trading/price-pipeline/price_pipeline/prices.db`

## Validation
- QuantSuite: Sharpe, PSR, DSR, Markov, Monte Carlo, Brownian
- Look-ahead bias check
- Walk-forward analysis
- Regime analysis
