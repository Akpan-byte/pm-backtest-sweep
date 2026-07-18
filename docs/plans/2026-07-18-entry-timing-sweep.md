# Entry Timing Sweep Implementation Plan

## Goal
Test whether delaying entry by N seconds after market open improves fill price and PnL for the top 3 trend-family strategies.

## Strategy
- Sweep 10 opening window durations (0s, 5s, 10s, 15s, 20s, 30s, 45s, 60s, 90s, 120s)
- Test all3 strategies: tf_dema_lb20_dev002_emax85_alp0001, tf_vwap_ticks_lb50_dev002_emax80, tf_holt_lb50_dev002_emax85_alp0002_hol0005
- Phase1: 5x subsample sweep (~15 min, ~99% accurate)
- Phase2: Full resolution re-run of top2 configs (~10 min, 100% accurate)

## Implementation

### Task 1: Add opening_window_sec parameter

**File:** `driver.py` (~line 986)

Add one check after signal triggers:

```python
# After: if sig and sig.get("triggered"):
opening_window = float(reg_entry.get("opening_window_sec", 0))
if elapsed_sec < opening_window:
    continue  # skip entry, wait for market to settle
```

### Task 2: Create batch sweep script

**File:** `sweep_entry_timing.py` (new)

```
Usage: python3 sweep_entry_timing.py --strategy tf_dema_lb20_dev002_emax85_alp0001 --workers 4 --subsample 5

For each strategy:
1. Load 2k sample data once
2. For each window (0,5,10,15,20,30,45,60,90,120):
   - Set opening_window_sec on registry entry
   - Run all markets (subsampled)
   - Collect: PnL, win_rate, max_dd, n_trades, avg_entry_price
3. Output results as JSON
```

### Task 3: Create GHA workflow

**File:** `.github/workflows/entry_timing_sweep.yml`

```yaml
matrix:
  strategy:
    - tf_dema_lb20_dev002_emax85_alp0001
    - tf_vwap_ticks_lb50_dev002_emax80
    - tf_holt_lb50_dev002_emax85_alp0002_hol0005
  subsample: [5]
```

3 workers, ~5 min each, ~15 min total.

### Task 4: Aggregate results

**File:** `analyze_sweep.py` (new)

Read all3 result files, produce comparison table:

| Strategy | Window | PnL | Win Rate | Max DD | Trades | Avg Entry |
|----------|--------|-----|----------|--------|--------|-----------|
| dema | 0s | ... | ... | ... | ... | ... |
| dema | 5s | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |

### Task5: Full resolution re-run

Run top2 configs at full resolution (no subsample) to confirm exact numbers.

## Estimated Time
- Phase1 (5x sub): ~15 min
- Phase2 (full res): ~10 min
- **Total: ~25 min**

## Verification
- Compare baseline (0s window) against existing drawdown study results to confirm consistency
- Check that win rate and PnL are within 1% of baseline at 5x subsample
