# Blueprint 3 — 00:00 UTC "MOS Session" Daily Draw

> Module: `mos_session_daily_draw.py`
> Source: "This Session is Made For Negative RR Traders" + MOS session / 11PM UTC extraction.

## Master Blueprint (extracted rules)

### Core Logic
Execute a negative RR trade exactly at the daily rollover, capitalizing on the
algorithmic pull toward the Previous Day High (PDH) or Previous Day Low (PDL),
utilizing a 4H structural wick as a protective wall.

### Execution Rules
- **State Check & Validation**:
  - Query the Daily Chart. IF `50_50_CANDLE == TRUE`, abort the day completely.
  - There must be `ESTABLISHED_MOVEMENT` (2+ candles trending).
  - IF the target (PDH/PDL) is swept before session open, abort the day.
- **Magnet Stacking**:
  - If Trend is UP → Target is PDH. If Trend is DOWN → Target is PDL.
  - If RELATIVE_EQUAL_HIGHS sit exactly at the PDH, flag as an "A+ Setup"
    ("Three magnets are better than one").
- **Execution & Risk Matrix**:
  - Entry: Market Order at exactly 00:00 UTC (the open of the MOS session).
  - Stop Loss (SL): Scan the 4H/1H state for the nearest `PROTECTIVE_SWING`
    opposing the trend. Place SL directly behind it.
  - Take Profit (TP): Aim for the PDH/PDL. Hard Constraint: Cap the TP at a strict
    maximum of 10 Pips. "Overshooting the previous day high... is not even needed."
  - Sizing: Since the win rate is mathematically skewed, risk sizing can be increased
    to 3% or 4% equity per trigger.

## CHANGELOG
- 2026-08-14 kilo — Created `mos_session_daily_draw.py`. Entry gated to first seconds
  of 00:00 UTC via `u.is_mos_session_time()`. Aborts on doji prior day
  (`u.is_doji_candle`), missing ESTABLISHED_MOVEMENT, or premature PDH/PDL sweep.
  PDH/PDL taken from last completed daily bar. SL behind 4H/1H PROTECTIVE_SWING
  (`u.find_protective_swing`). TP raw = PDH/PDL but hard-capped at 10 pips. Risk
  raised to 4%. State keyed by (asset, date, "mos_session_daily_draw", max_reentries=1);
  disk persistence + prune.

## Implementation Mapping
| Blueprint element | Code location |
|-------------------|---------------|
| 50_50 candle abort | `u.is_doji_candle(prev)` |
| Established movement | `u.is_established_movement(daily_bars, 2)` |
| Premature sweep | `spot >= pdh` (UP) / `spot <= pdl` (DOWN) |
| Entry time | `u.is_mos_session_time()` |
| Protective swing | `u.find_protective_swing()` on 4H then 1H |
| TP pip cap | `TP_PIP_CAP = 10.0` |
| Risk sizing | `HIGH_RISK_PCT = 0.04` |
