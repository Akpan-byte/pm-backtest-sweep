# Flexing Joe ORB Strategy — Backtest Specification

## Strategy source
`/config/_remote_edits/strategy_from_drive.txt` (PDF extracted from Google Drive).

## Core rules (exact from PDF)
1. **ORB window**: 09:30–10:00 AM ET. Mark high/low of the first 30-min candle (including wicks).
2. **Confirmation**: wait for a 10-min candle to **close** outside the ORB range.
3. **Entry**: drop to 2-min timeframe, add 20 EMA. Enter on pullback to 20 EMA that respects it and turns back in the breakout direction.
   - Long: price reclaims 20 EMA and closes back above → enter next 1-2 candles.
   - Short: price taps EMA and closes back under → enter next 1-2 candles.
4. **Stop**: below/above the most recent extreme (orb low/high or recent swing).
5. **Target**: HOD/LOD or 2× ORB range.

## Pre-market bias filters (from PDF)
- External: major macro event today → no trade; mental prep check (skip for backtest).
- VIX environment (optional — no local VIX data; implement as optional input).
- Gap vs prior-day 16:00 ET close.
- Position relative to prior-day high/low (PDH/PDL).
- London ORB range (03:00–03:30 AM ET).
- ES/NQ/VIX alignment (optional — cross-instrument data).
- Prior-day candle type (inside day/doji vs trending/wide-range).

## Backtest variants
1. `one_trade_per_day`: first valid signal only; no more entries that session.
2. `reentries`: allow new signals after a stop/target, up to `max_entries_per_day`.
3. `one_per_direction`: allow at most one long and one short signal per session.

## Instruments / data
- NQ, ES, YM 1-minute OHLCV CSVs in `/config/projects/trading/v5_orb_nq_backtest/market_data/`.
- Optional VIX / ES / NQ data paths for cross-instrument filters.

## Metrics to compute (Gemini Spark engine parity)
- Win rate, total trades, net PnL, avg trade PnL, profit factor
- Annualized Sharpe & Sortino (per-trade, scaled by trades/day)
- Max drawdown ($ and %)
- Deflated Sharpe Ratio (DSR)
- White's Reality Check p-value (WRC) via stationary block bootstrap
- Benjamini-Hochberg FDR q-value
- Monte Carlo: 50k runs with noise injection → p5/p50/mean total PnL
- Bootstrapped 95% CI: 50k resamples of per-trade PnL → CI for mean/total PnL

## Prop-firm payout modeling
- Accounts: 50k and 100k.
- Paths: standard (5-day cycle) and consistency (3-day cycle).
- Payout caps: 4k on 50k, 12k on 100k.
- Consistency rule: no single winning day > 50% of window profit (standard/consistency both enforce, consistency stricter window).
- Daily loss limit: 900 (50k), 1500–3000 (100k).
- Max/trailing loss limit: 2000 (50k), 3000–4500 (100k).
- 20k Monte Carlo on daily PnL and 20k bootstrapped CI on payout frequency/amount.

## Output structure
Each backtest run produces JSON with:
- `parameters`
- `daily_pnl` map
- `trades` list
- `metrics`
- `monte_carlo_50k`
- `bootstrap_ci_50k`
- `prop_firm_payouts`
- `prop_monte_carlo_20k`
- `prop_bootstrap_20k`

## GitHub Actions
20 chunks per instrument (60 jobs), aggregated into final report.
