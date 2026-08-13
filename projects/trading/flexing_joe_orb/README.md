# Flexing Joe ORB Futures Backtest

Implementation of the **ORB Pre-Market Bias Checklist / Opening Range Breakout** strategy from the Flexing Joe Trades PDF, with full 10-year backtesting across NQ, ES, and YM futures.

## Strategy rules (from PDF)
1. **Opening range**: high/low of the 09:30–10:00 ET 30-minute candle.
2. **Confirmation**: wait for a 10-minute candle to **close** outside the ORB range.
3. **Entry**: drop to the 2-minute chart with a 20 EMA; enter on a pullback that reclaims/closes back in the breakout direction.
4. **Stop**: below/above the ORB low/high.
5. **Target**: HOD/LOD or `target_multiple` × ORB range.

## Pre-market bias filters
- Gap vs prior-day 16:00 ET close.
- Position relative to prior-day high/low.
- London ORB range (03:00–03:30 ET).
- Prior-day candle type (inside day / doji proxy).
- Optional VIX and ES/NQ alignment when data paths are supplied.

## Variants
- `one_trade_per_day`: first valid signal only.
- `one_per_direction`: at most one long and one short per day.
- `reentries`: up to `max_entries_per_day` signals per day.

## Files
| File | Purpose |
|------|---------|
| `data.py` | CSV loading, bar resampling, EMA, session masks |
| `signals.py` | Daily bias computation and ORB signal generation |
| `execution.py` | Slippage, commission, daily loss halt, trailing drawdown, EOD close |
| `backtest.py` | End-to-end runner + CLI |
| `metrics.py` | Sharpe, Sortino, max drawdown, DSR, WRC, FDR |
| `mc_bootstrap.py` | 50k Monte Carlo and 50k bootstrapped CI |
| `prop_firm.py` | Topstep-style payout modeling for 50k/100k standard/consistency |
| `run_chunk.py` | GitHub Actions chunk runner |
| `aggregate_results.py` | Combines chunks and recomputes MC/bootstrap/prop-firm stats |
| `run_full_local.py` | Local parallel runner for all instruments/chunks |
| `.github/workflows/flexing_joe_orb.yml` | 20-chunk GitHub Actions workflow |

## Quick CLI test
```bash
cd /config/projects/trading
PYTHONPATH=/config/projects/trading python3 -m flexing_joe_orb.backtest \
  --symbol NQ \
  --data-path /config/projects/trading/v5_orb_nq_backtest/market_data/NQ_1min.csv \
  --start-date 2016-06-01 --end-date 2016-08-31 \
  --variant reentries \
  --output /tmp/fjo_test.json
```

## Local full 10-year run
```bash
cd /config/projects/trading
PYTHONPATH=/config/projects/trading python3 flexing_joe_orb/run_full_local.py \
  --instruments NQ,ES,YM --total-chunks 20 --workers 8 \
  --output-dir /tmp/fjo_full_results \
  --mc-runs 50000 --bootstrap-samples 50000
```

## GitHub Actions
Trigger `.github/workflows/flexing_joe_orb.yml` manually or push changes under `projects/trading/flexing_joe_orb/`. It runs 20 chunks per instrument (60 jobs) and aggregates into a final report artifact.

## Notes / caveats
- VIX data is not included locally; VIX filters are skipped unless `vix_path` is provided.
- ES/NQ alignment filters are skipped unless `es_path`/`nq_path` are provided.
- The strategy PDF does not specify exact numeric stops/targets; the backtest parameterizes them and defaults to 2× ORB range.
