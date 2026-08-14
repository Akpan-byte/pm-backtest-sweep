# Blueprint 4 — Post-8AM BPR Magnet (Orderflow Micro-Scalp)

> Module: `post_8am_bpr_magnet.py`
> Source: Post-8AM BPR Magnet / Orderflow Micro-Scalp extraction.

## Master Blueprint (extracted rules)

### Core Logic
Imbalances formed aggressively at the New York open act as algorithmic vacuums.
Wait for the structural gap to finalize, then scalp the rebalance.

### Execution Rules
- **Time Filter Lock**: The system ignores all data until 08:00 EST. Any FVG or BPR
  detected prior to 08:00 EST is classified as "stale" and scrubbed from targeting memory.
- **Detection Array (1m / 5m)**:
  - Listen for the creation of a BPR or DIRTY_BPR pushing in the direction of the trend.
  - Verify orderflow state: If the algorithm has already traded into and respected an FVG
    earlier in the session, "orderflow is bearish/bullish."
- **Execution & Micro-Risk Matrix**:
  - Entry: Market order in the direction of the BPR as localized price action creates
    internal liquidity (minor sweeps) pulling toward it.
  - Take Profit (TP): Hardcoded to 2 Pips.
  - Stop Loss (SL): Hardcoded to 5 Pips (0.4 RR).
  - Dynamic Risk Tiering: Standard risk is 1.0%. If the BPR contradicts the 15m
    structural trend (flagged as a counter-trend or "A- setup"), dynamically override
    and halve risk to 0.5%.

## CHANGELOG
- 2026-08-14 kilo — Created `post_8am_bpr_magnet.py`. Time lock via
  `u.is_after_ny_open_filter_time()` (08:00 EST). Detects BPR/DIRTY_BPR on 1m/5m
  (`u.detect_bpr`, `u.detect_dirty_bpr`), keeps only those aligned with the 15m
  structural trend (`_fifteen_min_trend`). Requires orderflow FVG confirmation.
  Hardcoded TP=2 pips / SL=5 pips. Risk halved to 0.5% when BPR contradicts 15m trend.
  State keyed by (asset, date, "post_8am_bpr_magnet", max_reentries); disk persistence + prune.

## Implementation Mapping
| Blueprint element | Code location |
|-------------------|---------------|
| 08:00 EST lock | `u.is_after_ny_open_filter_time()` |
| BPR / DIRTY_BPR | `u.detect_bpr()`, `u.detect_dirty_bpr()` |
| 15m trend | `_fifteen_min_trend()` |
| Orderflow confirm | `u.detect_fvg()` + `state["orderflow_state"]` |
| Hardcoded TP/SL | `TP_PIPS=2.0`, `SL_PIPS=5.0` |
| Dynamic risk | `STD_RISK_PCT=0.01`, `COUNTER_RISK_PCT=0.005` |
