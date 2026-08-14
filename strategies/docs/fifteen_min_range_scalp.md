# Blueprint 1 — 15-Minute Range & 1-Minute Movement Candle Scalp

> Module: `fifteen_min_range_scalp.py`
> Source: `Steal This Scalping Strategy to Get Funded FAST` + Day 6/8/9/10/11 live trading.

## Master Blueprint (extracted rules)

### The 4-Phase Execution System
1. **HTF Bias** → Daily (PDH/PDL) & 4H (POI / FVG / BPR interaction)
2. **15M Range** → Mark extreme High & Low; center current price
3. **1M Raid** → Wait for 1M Swing High/Low sweep strictly past 8:30 AM ET
4. **Movement Candle** → Displacement candle (>=70% body, <30% wicks, wait to close)

### Core Parameters & Market Conditions
- Applicable Assets: Futures (NQ, ES, YM), Forex, Crypto, Indices, Commodities.
- Timezone: UTC-4 (New York local).
- Execution Window: 8:30 AM – 2:00 PM EST.
- Strict Timing Filter: No analysis/entry before 8:30 AM EST.
- Hard Session Exit: Close any running trade at 2:00 PM EST at market.
- Typical Trade Duration: 2–10 min (max 1 hour).

### Step-by-Step Checklist
- **Step 1 HTF Bias**: Daily → draw to PDH/PDL or fill daily gap. 4H → confirm POI
  (FVG/BPR) interaction. Wick through POI = respected; body close outside = disrespected.
  Simplified mode: infer from 15m LL/LH (bearish) or HH/HL (bullish).
- **Step 2 15M Range**: identify most extreme recent 15m High/Low (binary 50/50 state).
  Centering rule: price must be near center; extend range if near a boundary.
- **Step 3 1M Sweep (post 8:30 ET)**: Shorts → sweep 1m swing high (buy-side raid).
  Longs → sweep 1m swing low (sell-side raid). News spikes accelerate, don't invalidate.
- **Step 4 Movement Candle**: strong displacement in bias direction; significantly larger
  than recent candles; >=70% body, <30% wicks; wait for close. 3 consecutive medium 1m
  candles = valid MC on 2m/3m.

### Order Placement & Risk
- Entry: at MC close, or limit slightly higher (short)/lower (long) during next bar wick.
- SL Conservative: above raid swing high (short) / below sweep swing low (long).
- SL Aggressive: above MC high / above MC body high.
- TP Standard: opposing 15m range boundary.
- TP Internal: nearest 1m/5m BPR or FVG.
- TP Extended: PDH/PDL or EQH/EQL for 1:2–1:3 RR.

## CHANGELOG
- 2026-08-14 kilo — Created `fifteen_min_range_scalp.py`. Implements 4-phase system
  with HTF bias (Daily+4H), centered 15m range framing, post-8:30 ET 1m liquidity
  sweep, and Movement Candle confirmation (>=0.70 body, >2.5×ATR). State keyed by
  (asset, date, "15m_range_scalp", max_reentries); disk persistence + 2-day prune;
  hard 2PM ET exit; cooldown guard. Verified triggering on crafted bars.

## Implementation Mapping
| Blueprint element | Code location |
|-------------------|---------------|
| HTF bias | `_frame_htf_bias()` → `u.is_established_movement()` + 4H FVG respect |
| 15m range | `_frame_15m_range()` (center within 25% of range) |
| 1m sweep | `_detect_liquidity_sweep()` (after `u.ny_open_et()`) |
| Movement Candle | `_detect_movement_candle()` (`STRONG_BODY_RATIO`, `MOVEMENT_CANDLE_ATR_MULT`) |
| Time gates | `u.ny_open_et()`, `u.hard_session_exit_et()`, `u.is_market_hours_et()` |
| Signal return | `sl` at sweep level, `tp` at opposing range boundary |
