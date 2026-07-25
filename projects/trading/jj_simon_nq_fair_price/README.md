# JJ Simon NQ Fair-Price Strategy

Production-ready implementation of JJ Simon's "Fair Pricing Theory" continuation / mean-reversion model for NASDAQ Futures (`NQ`) on the 1-minute chart.

## Strategy Goal

Trade the NY session around three fixed Fair-Price Anchors:

| Anchor | Candle (EST) | Used In |
|--------|--------------|---------|
| Scheduled News | 08:29 AM | 08:30–09:00 news drift / reversion |
| NY Open | 09:29 AM | 09:30–09:40 continuation, 09:40–11:00 mean reversion |
| PM Session | 01:59 PM | 02:00–03:00 PM continuation / reversion (optional) |

## Core Definitions

### Displacement Candle

```
candle_body = |close - open|
upper_wick  = high - max(open, close)
lower_wick  = min(open, close) - low

isDisplacement =
    candle_body > previous_candle_body
    AND upper_wick < candle_body * 0.3
    AND lower_wick < candle_body * 0.3
```

### Break of Structure (BOS)

```
Bullish BOS = close > highest high of last N candles
Bearish BOS = close < lowest low of last N candles
```

### Fair-Price Anchor

Body range (`min(open,close)` to `max(open,close)`) of the anchor candle, fixed for the session unless an unexpected news spike resets it.

## Session State Machine (EST)

| Window | Time | Anchor | Direction | Entry Trigger |
|--------|------|--------|-----------|---------------|
| Scheduled News | 08:30–09:00 | 08:29 | Reversion toward anchor | BOS / Displacement pointing back |
| Opening Continuation | 09:30–09:40 | 09:29 | Follow 09:30 candle | First BOS / Displacement in open direction |
| Mean Reversion | 09:40–11:00 | 09:29 | Back to anchor | BOS / Displacement pointing back, ≥ 38 pts away |
| PM Continuation | 02:00–02:10 | 01:59 | Follow 02:00 candle | First BOS / Displacement |
| PM Reversion | 02:10–03:00 | 01:59 | Back to anchor | BOS / Displacement pointing back, ≥ 38 pts away |

Hard cutoff: **no new entries after 11:00 AM EST** unless the optional PM session is enabled.

## Unexpected News Reset

If a 1-minute candle range > `news_spike_threshold` (default 60 NQ points) outside of scheduled news/open windows:

1. Invalidate the current anchor.
2. Build a new anchor from the consolidation zone of the next 3 candles.
3. Switch to **News Drift Mode**: trade continuations in the spike direction with 50 pt SL / 75 pt TP.

## Risk Profiles

Two prop-firm presets are included:

| Profile | Stop Loss | Take Profit | R:R | Starting Balance |
|---------|-----------|-------------|-----|------------------|
| 50k | 25 pts | 38 pts | 1 : 1.5 | $50,000 |
| 150k | 50 pts | 75 pts | 1 : 1.5 | $150,000 |

### Dynamic Candle Rule

If the trigger candle body > `dynamic_candle_trigger` (default 25 pts):

- Reduce size 50%.
- Auto-switch to 50 pt SL / 75 pt TP.

### Trade Limits

- Max 3 trades per morning session.
- Lockout after 2 consecutive losses.

## Repository Layout

```
jj_simon_nq_fair_price/
├── README.md                    # this file
├── CHANGELOG.md                 # version history
├── config/
│   └── profiles.json            # 50k / 150k prop profiles
├── src/
│   └── jj_simon_fair_price.py   # clean modular strategy engine
├── scripts/
│   └── run_single_backtest.py   # run one parameter set locally
├── pinescript/
│   └── JJ_Simon_NQ_FairPrice.pine   # TradingView Pine Script v5
└── docs/
    └── STRATEGY_SPEC.md         # full written spec from JJ Simon
```

## Quick Start

### 1. Run a single backtest

```bash
cd /config/projects/trading/jj_simon_nq_fair_price
python3 scripts/run_single_backtest.py --profile 50k
```

### 2. Run the full quant suite sweep

The full sweep is managed from `../jj_quant_backtest/`:

```bash
cd /config/projects/trading/jj_quant_backtest

# Run all 20 chunks locally on 4 workers (quick mode: 2k MC/bootstrap)
python3 run_full_sweep.py --instrument NQ --workers 4 --quick

# Or run one chunk manually
python3 jj_quant_backtest.py --instrument NQ --chunk_id 0 --total_chunks 20 --quick
```

Aggregate after all chunks finish:

```bash
python3 aggregate_results.py --results_dir results/
```

## Data

The backtest consumes 1-minute NQ/ES/YM futures data from:

```
/config/fvg_execution_engine/backtests/data/<symbol>/M1.csv.gz
```

Columns: `ts,open,high,low,close,volume` (timezone-aware, converted to America/New_York internally).

## Dependencies

- Python 3.12+
- numpy
- pandas

## License

Private — for prop-firm evaluation only.
