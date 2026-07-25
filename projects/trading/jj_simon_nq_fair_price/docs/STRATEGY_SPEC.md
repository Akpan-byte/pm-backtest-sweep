# JJ Simon NQ Fair-Price Strategy — Full Specification

## Overview & Goal

Build a production-ready, fully automated trading strategy/indicator for NASDAQ Futures (NQ) on the 1-minute chart (with 5-minute fallback logic) that strictly implements JJ Simon's "Fair Pricing Theory" continuation and mean-reversion model.

---

## 1. Core Definitions & Anchors

### A. Fair Price Anchors (The Reference Zones)

1. **Standard NY Open Anchor:** The `open` to `close` range (body only) of the **09:29 AM EST** 1-minute candle.
2. **Scheduled News Anchor (8:30 AM):** The `open` to `close` range (body only) of the **08:29 AM EST** 1-minute candle.
3. **PM Session Anchor (2:00 PM):** The `open` to `close` range (body only) of the **01:59 PM EST** 1-minute candle.

> Note: The anchor line/zone must remain fixed for that session unless an unscheduled news event overrides it.

### B. Structural & Candle Triggers

1. **Displacement Candle (`isDisplacement`):**
   - `candle_body = abs(close - open)`
   - Condition 1: `candle_body > prev_candle_body`
   - Condition 2: `upper_wick < (candle_body * 0.3)`
   - Condition 3: `lower_wick < (candle_body * 0.3)`

2. **Break of Structure (`isBOS`):**
   - **Bullish BOS:** 1-minute candle `close` is strictly GREATER THAN the highest high of the last 5 candles.
   - **Bearish BOS:** 1-minute candle `close` is strictly LESS THAN the lowest low of the last 5 candles.

---

## 2. Timing Windows & State Machine

All times are in **US Eastern Time (EST/EDT)**:

### Window 1: Scheduled News Phase (08:30 - 09:00 AM)

- Target Anchor: 08:29 AM Candle Body.
- If 8:30 news releases as expected (actual ≈ forecast), treat 08:30 volatility spike as "unfair." Look for Reversion entries back to the 08:29 AM Anchor.
- If position is open entering 09:30 AM open, automatically move Stop Loss to Break-Even to avoid open-wick stop-outs.

### Window 2: Opening Continuation Phase (09:30 - 09:40 AM)

- Target Anchor: 09:29 AM Candle Body.
- Direction: Follow the color/direction of the 09:30 AM opening candle.
- Entry Trigger: First `isDisplacement` or `isBOS` in the direction of the open.

### Window 3: Mean Reversion Phase (09:40 - 11:00 AM)

- Target Anchor: 09:29 AM Candle Body.
- Direction: ALWAYS pointing *back* toward the 09:29 AM Fair Price Anchor.
- Distance Filter: Only enter if current price is at least **38 points** away from the Fair Price Anchor (ensures R:R expectancy).
- Entry Trigger: `isBOS` or `isDisplacement` pointing back toward the 09:29 AM Anchor.

### Window 4: Afternoon Session Phase (02:00 - 03:00 PM) [Optional]

- Target Anchor: 01:59 PM Candle Body.
- Same Continuation (0-10m) / Reversion (10-60m) rules apply.

### Hard Cutoff (11:00 AM)

- DO NOT enter any new positions after 11:00 AM EST.
- Active positions may run to Take Profit or Stop Loss.

---

## 3. Unexpected News & Fair Price Reset

- **Detection:** If a single 1-minute candle exceeds **60 NQ points** outside of 8:30 AM or 9:30 AM (e.g., random tweet/headline), trigger an "Unexpected News Event."
- **Action:**
  1. Invalidate the previous Fair Price Anchor.
  2. Define the new Fair Price Anchor as the consolidation zone formed in the 3 candles immediately following the spike.
  3. Switch to **News Drift Mode**: Trade continuations in the direction of the spike with wider stops (50 pts SL / 75 pts TP).

---

## 4. Risk Management & Execution Matrix

Default parameters must support two preset profiles:

### Profile A: 50k Prop Account (Standard)

- **Stop Loss:** 25 NQ Points (100 Ticks)
- **Take Profit:** 38 NQ Points (152 Ticks)
- **R:R Ratio:** 1 : 1.5

### Profile B: 150k Prop Account (Large)

- **Stop Loss:** 50 NQ Points (200 Ticks)
- **Take Profit:** 75 NQ Points (300 Ticks)
- **R:R Ratio:** 1 : 1.5

### Dynamic Candle Size Rule

- If the trigger candle range > **25 points**:
  - Cut Position Size by **50%**.
  - Auto-switch to **50 pt SL / 75 pt TP** to avoid getting wicked out by increased volatility.

### Trade Limits

- **Max Trades Per Morning Session:** 3 trades maximum.
- **Consecutive Loss Lockout:** Stop trading for the morning session after 2 consecutive losses.

---

## 5. Technical Deliverable Requirements

1. Write clean, modular, and fully commented code (specify language: Pine Script v5 or Python/NinjaScript).
2. Draw visual rectangles on the chart for the Fair Price Anchors (08:29 AM, 09:29 AM, 01:59 PM).
3. Highlight `isDisplacement` candles with distinct colors (e.g., bright green for bullish, bright magenta for bearish).
4. Implement strict session-based time filtering so execution strictly adheres to the EST time windows defined above.
