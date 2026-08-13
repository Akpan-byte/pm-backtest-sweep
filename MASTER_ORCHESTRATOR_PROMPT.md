# MASTER ORCHESTRATOR PROMPT
# Polymarket Trading Bot Clean Rebuild
# Orchestrator delegates to gemini --yolo subagents
# REALISTIC TIMESTAMPS (2026-06-16T00:26:50 start)

## TIME ESTIMATES (REALISTIC & UNBIASED)

| Step | Duration | End Time (if start 00:26) |
|------|----------|---------------------------|
| Resource check + compression | 1-2 min | 00:27-00:28 |
| Backup to Google Drive | 2-5 min | 00:29-00:31 |
| Signal extraction (17 signals, batches of 4) | 15-25 min | 00:31-00:51 |
| Lifecycle.py creation | 5-8 min | 00:53 |
| Market_data.py creation | 3-5 min | 00:56 |
| Config.py creation | 1-2 min | 00:57 |
| Main.py wiring | 6-10 min | 01:03 |
| Unit testing/signals parity | 8-12 min | 01:13 |
| **Full backtest (tick data)** | 45-90 min | 02:33 |
| 48-hour paper trading | 48 hours | 02:33+ |

**Total build + validation: 2.5-3.5 hours**
**Total with paper trading: 48 hours 2.5-3.5 hours**

## RESOURCE CHECK & OPTIMIZATION

```bash
# Time: 00:26-00:28 (1-2 min)
df -h /config && du -sh /config/* 2>/dev/null | sort -rh | head -20
# If disk >80% used:
find /config -type f -mtime +90 -size +10M 2>/dev/null | head -50 > /tmp/old_files.txt
tar -czf /tmp/old_archives_$(date +%s).tar.gz -T /tmp/old_files.txt 2>/dev/null && \
echo "Compressed at $(date)" >> /config/.hermes/compression_log.txt
```

## PARALLEL DELEGATION STRATEGY

```bash
# MAX 4 concurrent gemini calls (prevent VM overload)
# DELAY: 250-500ms between calls

# Batch 1-5: 17 signals extraction (4 each)
# Batch 6: lifecycle, market_data, config, main (sequential)
# Batch 7: testing, backtesting (sequential)
```

## ORCHESTRATOR ENTRY POINT

```bash
gemini --yolo -i -p 'TIMESTAMP 00:26:50 START. You are the ORCHESTRATOR for Polymarket bot rebuild. DELEGATE to gemini --yolo subagents. MAXIMIZE EFFICIENCY: Run UP TO 4 parallel, then sequential. Read /config/.hermes/plans/polymarket_clean_rebuild.md.

STEP 1 (00:26-00:28): Resource check + compress old files >90 days if disk >80%
STEP 2 (00:28-00:31): Delegate backup_to_drive extension and Google Drive backup
STEP 3 (00:31-00:51): Delegate 17 signal extractions in 4-parallel batches
STEP 4 (00:51-00:58): Delegate lifecycle.py, market_data.py, config.py, main.py
STEP 5 (00:58-01:10): Delegate unit testing and signals parity validation
STEP 6 (01:10-02:33): Delegate FULL BACKTEST against tick data
STEP 7 (02:33+): Delegate 48-hour paper trading

QUALITY CONTROL FOR EACH SUBAGENT TASK:
- Subagent MUST run tests/simulations before marking task complete
- Orchestrator MUST verify outputs: run test script, check for look-ahead bias
- Compare ALL metrics to original build: Sharpe, PSR, DSR, Markov, Drawdown, PnL

BACKTEST DATA SOURCES (use ALL):
- /config/projects/trading/data/poly-data/poly_data/btc_polymarket_ticks.csv
- /config/projects/trading/data/poly-data/poly_data_elite_7-17/.../btc_polymarket_ticks.csv
- /config/projects/trading/price-pipeline/price_pipeline/prices.db

QUANT SUITE ENGINE:
- /config/projects/trading/quant-suite/engine/quant_suite.py (PSR, DSR, Sharpe, Sortino, Markov, Monte Carlo, Brownian)

IF BLOCKED: Document error + timestamp, gh search OR websearch, apply solution, retry.'
```

## CRON SELF-CHECK (Every 15 minutes)

```bash
# Add to crontab: crontab -e
*/15 * * * * cd /config && python3 -c "
import os, subprocess
from datetime import datetime

status = '.hermes/orchestrator_status.log'
bot = 'projects/trading/polymarket/bot'

checks = {
    'Signals': os.path.exists(f'{bot}/signals.py'),
    'Lifecycle': os.path.exists(f'{bot}/lifecycle.py'),
    'Main': os.path.exists(f'{bot}/main.py'),
    'Legacy': os.path.exists('legacy/shadow_paper_bot_LEGACY.py'),
    'Metrics': os.path.exists('.hermes/backtest_metrics.json'),
    'Tests Passed': os.path.exists('.hermes/test_passed.json'),
}

with open(status, 'a') as f:
    f.write(f'{datetime.now().isoformat()}: ASSESSMENT: ' + ' | '.join(f'{k}={v}' for k,v in checks.items()) + '\n')

if not os.path.exists(f'{bot}/main.py'):
    subprocess.run(['gemini', '--yolo', '-i', '-p', 'ORCHESTRATOR WAKE. Assess rebuild. Report: Progress[X/7] Last:[task] Current:[work] Blockers:[issues] Problems:[errors] Next:[actions]. QUALITY CHECK: Verify subagent tests passed and no look-ahead bias.'])
"
```

## SELF-ASSESSMENT RESPONSE FORMAT

```
== ORCHESTRATOR STATUS REPORT ==
Timestamp: [exact time]
Progress: [X/7 steps complete]
Last Completed: [step description + timestamp]
Current Work: [step + timestamp started]
Blockers: [none OR exact error]
Problems Found: [none OR exact error]
Solutions Attempted: [none OR action taken]
Next Steps: [immediate action]
Resource State: [disk% / CPU load / RAM usage]
Time Elapsed: [minutes since start]
Est. Completion: [ETA based on remaining work]

QUALITY METRICS TO REPORT:
- Signals parity accuracy: [vs legacy]
- Lookahead bias check: [PASS/FAIL]
- Sharpe ratio: [value]
- PSR: [value]
- DSR: [value]
- Markov transitions: [W->W, W->L, L->W, L->L]
- Max drawdown: [value]
- Monte Carlo (1M sims) P99: [value]
- Brownian motion (1M sims) P95: [value]

IF BLOCKED:
1. Document exact error + timestamp
2. Run: gh search --repo=kilocode/kilocode 'error_message' --tool-name
3. Or: websearch query='error_message python solution'
4. Apply solution to subagent prompt
5. Retry delegation
```

## SUBAGENT TESTING DIRECTIVE

```bash
# Before subagent marks task complete, MUST run:
python3 -c "
# 1. Compile check
import py_compile; py_compile.compile('filepath', doraise=True)

# 2. Import check  
import filepath; print('Import OK')

# 3. Unit test
# Run tests, assert no exceptions

# 4. Lookahead bias check
# Verify no future data in signal logic
"

# If fails, subagent MUST NOT mark complete - fix and retry
```

## BACKTEST VALIDATION (FINAL STEP)

### Time Estimates for Backtest Matrix:
| Stack | Balance | Duration | End Time |
|-------|---------|----------|----------|
| 5-min stack alone | $100 | 15-20 min | 01:55 |
| 15-min stack alone | $100 | 25-35 min | 02:20 |
| Combined stack ($200 shared) | $200 | 20-30 min | 02:33 |
| Regime analysis | - | 15-20 min | 02:53 |
| Walk-forward analysis | - | 10-15 min | 03:08 |
| Overfitting check | - | 5-10 min | 03:18 |
| Final report | - | 2 min | 03:20 |

**Total backtest time: ~2h 44m**

### Regime Analysis to Include:
- High volatility (ATR > 1.5% in 1h window)
- Low volatility (ATR < 0.5% in 1h window)
- Trend up (BTC > 20 SMA)
- Trend down (BTC < 20 SMA)
- Mean reversion zones (price near Bollinger midline)
- Opening hours (09:30-16:00 UTC)
- Closing hours (16:00-23:00 UTC)

### Stack Strategy Allocations:

**5-MIN STACK ($100):** SNIPE, BREAKOUT_PCT_0.04, BREAKOUT_PCT_0.08, BREAKOUT_Z_1.6, KINETIC_VELOCITY_BREAKOUT, L2_ABSORPTION_SPREAD_COLLAPSE, LIQUIDATION_SPOT_GAP_FADE, MR_GAMMA_EXPIRY_PIN, MR_L2_OFI_DELTA_FADE, MEAN_REVERSION, MEAN_REVERSION_Z_1.5, MEAN_REVERSION_OPPOSITE_EXIT

**15-MIN STACK ($100):** MEAN_REVERSION, MEAN_REVERSION_Z_1.5, MEAN_REVERSION_OPPOSITE_EXIT, BREAKOUT_PCT_0.03-0.08, BREAKOUT_Z_1.5-3.0, L2_ABSORPTION_SPREAD_COLLAPSE

**COMBINED ($200 shared):** All 12+ strategies use $200 shared pool

```bash
gemini --yolo -i -p '
BACKTEST MATRIX VALIDATION: Run 3 stack configurations:
(1) 5-MIN STACK ALONE ($100): SNIPE, BREAKOUT_PCT_0.04, BREAKOUT_PCT_0.08, BREAKOUT_Z_1.6, KINETIC_VELOCITY_BREAKOUT, L2_ABSORPTION_SPREAD_COLLAPSE, LIQUIDATION_SPOT_GAP_FADE, MR_GAMMA_EXPIRY_PIN, MR_L2_OFI_DELTA_FADE, MEAN_REVERSION, MEAN_REVERSION_Z_1.5, MEAN_REVERSION_OPPOSITE_EXIT
(2) 15-MIN STACK ALONE ($100): MEAN_REVERSION, MEAN_REVERSION_Z_1.5, MEAN_REVERSION_OPPOSITE_EXIT, BREAKOUT_PCT_0.03-0.08, BREAKOUT_Z_1.5-3.0
(3) COMBINED STACK ($200 shared): All 12+ strategies

For ALL stacks:
- Run QuantSuite with 1M Monte Carlo + 1M Brownian simulations
- Calculate: Sharpe, PSR, DSR, Markov transitions, max_drawdown, drawdown_from_peak, drawdown_from_sod, PnL per strategy
- Day-by-day analysis required
- Regime analysis required (volatility, trend, hours)
- Walk-forward analysis (train/test split 80/20)
- Overfitting check (walk-forward correlation > 0.1 = OK)
- Monitor CPU/storage every 5 minutes, log to .hermes/resource_monitor.log
- Look-ahead bias check REQUIRED (no future data in signal)

Output: /config/.hermes/backtest_matrix_results.md with performance comparison vs legacy
'
```

## SUCCESS VERIFICATION

```bash
python3 -c "
import os, json
checks = [
    ('signals.py', os.path.exists('projects/trading/polymarket/bot/signals.py')),
    ('lifecycle.py', os.path.exists('projects/trading/polymarket/bot/lifecycle.py')),
    ('main.py', os.path.exists('projects/trading/polymarket/bot/main.py')),
    ('tests_passed', os.path.exists('.hermes/test_passed.json')),
    ('backtest_metrics', os.path.exists('.hermes/backtest_metrics.json')),
    ('legacy', os.path.exists('legacy/shadow_paper_bot_LEGACY.py')),
    ('order_daemon.js', os.path.exists('order_daemon.js')),
]
all_ok = all(v for _,v in checks)
print('REBUILD STATUS:', 'SUCCESS' if all_ok else 'INCOMPLETE')
for n,v in checks: print(f'  {n}:', 'OK' if v else 'MISSING')
# Load and compare metrics
try:
    m = json.load(open('.hermes/backtest_metrics.json'))
    print('SHARPE:', m.get('sharpe', 'N/A'))
    print('PSR:', m.get('psr', 'N/A'))
    print('P99 MC DRAWDOWN:', m.get('mc_p99_dd', 'N/A'))
except: pass
"
```