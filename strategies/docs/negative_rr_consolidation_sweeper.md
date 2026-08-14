# Blueprint 2 — Negative RR Consolidation Sweeper

> Module: `negative_rr_consolidation_sweeper.py`
> Source: Consolidation Trap / Negative RR into Equal Highs/Lows extraction.

## Master Blueprint (extracted rules)

### Core Logic ("Price Over Time" variant)
Ranging markets trap retail traders. Trade directly into EQH/EQL liquidity with a
massive stop-loss and a micro-target to mathematically guarantee a ~95% win rate.

### Execution Rules
- **Phase Filter**: IF `ESTABLISHED_MOVEMENT == TRUE`, disable strategy. Market phase
  must flag as "Consolidation." "We like choppiness... we don't like when the market
  is just trending."
- **ABORT** on NFP days, Bank Holidays, and the first 3 days of any month.
- **Target Acquisition**: Scan for clear RELATIVE_EQUAL_HIGHS or RELATIVE_EQUAL_LOWS.
- **Negative Risk Matrix**:
  - Entry: Market order heading toward the EQH/EQL pool. "You need to be outcome
    independent... you don't care if it takes minutes or hours."
  - Take Profit (TP): Placed precisely at the EQH/EQL boundary (targeting 0.2 RR).
  - Stop Loss (SL): Exceptionally wide (5× the TP distance). "You have to give the
    trade room to breathe."
- **Recovery Loop Protocol**: Standard risk fixed at 1% per trade. IF TradeResult ==
  LOSS (historically ~5 in 100 trades): Enter Recovery State. Recovery State: the next
  two consecutive trade executions adjust TP logic to capture 0.5 RR. Once both recovery
  trades flag as WIN, revert to baseline 0.2 RR.

## CHANGELOG
- 2026-08-14 kilo — Created `negative_rr_consolidation_sweeper.py`. Phase filter gates
  on `u.is_consolidation_phase()` (opposite of ESTABLISHED_MOVEMENT). Aborts on
  NFP/first-Friday/bank-holiday/early-month via `_is_nfp_bank_holiday_early_month()`.
  Detects EQH/EQL via `u.detect_eqh_eql()`. TP at EQ level (0.2 RR baseline, 0.5 RR in
  recovery), SL = 5× distance. Recovery loop tracks last trade P/L. State keyed by
  (asset, date, "neg_rr_consolidation", max_reentries); disk persistence + prune.

## Implementation Mapping
| Blueprint element | Code location |
|-------------------|---------------|
| Phase filter | `u.is_established_movement()` inverted in signal |
| Abort windows | `_is_nfp_bank_holiday_early_month()` |
| EQH/EQL | `u.detect_eqh_eql(highs, lows, 0.001)` |
| TP baseline / recovery | `BASE_TP_RR=0.2`, `RECOVERY_TP_RR=0.5` |
| Wide SL | `SL_MULTIPLE_OF_TP=5.0` |
| Recovery loop | `state["recovery_state"]` (2 wins to exit) |
| Risk | 1% per trade (standard) |
