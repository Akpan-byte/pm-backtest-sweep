# Master Blueprint — StarTrading Intraday Strategy Suite

This document is the canonical source for the four coded intraday strategies
derived from the StarTrading / StarBeyondTheSky video extractions. Each
strategy has its own documentation file under `docs/` that contains the exact
master blueprint, a changelog, and the implementation mapping.

| # | Strategy | Code Module | Doc |
|---|----------|-------------|-----|
| 1 | 15m Range & 1m Movement Candle Scalp | `fifteen_min_range_scalp.py` | `docs/fifteen_min_range_scalp.md` |
| 2 | Negative RR Consolidation Sweeper | `negative_rr_consolidation_sweeper.py` | `docs/negative_rr_consolidation_sweeper.md` |
| 3 | 00:00 UTC MOS Session Daily Draw | `mos_session_daily_draw.py` | `docs/mos_session_daily_draw.md` |
| 4 | Post-8AM BPR Magnet (Orderflow Micro-Scalp) | `post_8am_bpr_magnet.py` | `docs/post_8am_bpr_magnet.md` |

## Universal State Variables & Lexicon

These definitions use the educator's exact words and visual heuristics,
translated into programmatic logic in `strategies/core/` (time_utils, candle_utils, detectors, state_store):

- **ESTABLISHED_MOVEMENT**: "Clear willingness to go higher or lower." Requires
  >= 2 consecutive Daily candles closing heavily in the same direction with
  large bodies and minimal opposing wicks.
- **50_50_CANDLE**: A Daily candle where the "body is basically so close to the
  wick" (a Doji). If the previous daily candle was a 50/50 candle, the state is
  flagged as indecisive.
- **RELATIVE_EQUAL_HIGHS_LOWS (EQH/EQL)**: Two or more swing highs/lows that
  are "relatively equal close to each other." Constraint: "If I give you 3
  seconds to identify [them] and you're not able to tell... they are not."
  They must be visually obvious micro-threshold matches, acting as algorithmic
  "magnets" holding "a bunch of liquidity."
- **PROTECTIVE_SWING**: A 1H or 4H swing extreme that has traded into a Fair
  Value Gap (FVG) and immediately rejected, closing outside of it. This
  establishes a zone that "will most likely not be taken out."
- **MOVEMENT_CANDLE**: A 1-minute candle that is "significantly bigger than all
  of the previous candles." Constraint: Must have a large body and a small wick.
  (Body Size > 2.5 * ATR(20)).
- **BPR (Balanced Price Range)**: Overlapping FVGs. "The more BPRs you're going
  to stack on top of each other, the more likely it becomes price is going to
  fall back into that."
- **DIRTY_BPR**: A zone where an FVG has been partially pierced by a wick, but a
  full candle body has not yet closed through it.

## Shared Risk / Account Rules (apply to all)

| Rule / Component | Execution Protocol |
|------------------|-------------------|
| Max Daily Trade Limit | 1 trade per day by default. Re-entries prohibited unless a completely new, independent setup forms (new liquidity sweep + new valid Movement Candle). |
| Fixed Risk Sizing | Fixed 1% risk per trade on funded accounts (or fixed micro contracts). Never scale into losing positions. |
| Breakeven Trigger | Move stop loss to breakeven once price trades cleanly through the initial 1m FVG/BPR or completes a 1:1 expansion. |
| Dynamic Early Exit | If price stalls in prolonged consolidation or an opposing 1m BPR forms with body closure against the position, exit immediately at market to preserve capital. |
| Capital Growth Over Risk | Reinvest challenge payout rewards into larger account sizes rather than increasing risk %. |
| Pre- & Post-Market Journaling | Log every trade before entry (thesis + checklist) and after exit (emotions, errors, overshot R-multiples). |

## Coded Quantified Thresholds (quick reference)

| Parameter | Value |
|-----------|-------|
| Strong body ratio (Movement Candle) | >= 0.70 |
| Doji body ratio (50_50 candle) | <= 0.30 |
| Movement candle vs ATR(20) | > 2.5 * ATR(20) |
| NY open liquidity-sweep gate | 08:30 ET |
| Hard session exit | 14:00 ET |
| Blueprint 2 base TP RR | 0.2 |
| Blueprint 2 recovery TP RR | 0.5 |
| Blueprint 2 SL multiple of TP | 5.0 |
| Blueprint 3 TP pip cap | 10 pips |
| Blueprint 3 risk sizing | 3-4% |
| Blueprint 4 TP / SL | 2 pips / 5 pips (0.4 RR) |
| Blueprint 4 std / counter risk | 1.0% / 0.5% |

## Source Videos Parsed

- Steal This Scalping Strategy to Get Funded FAST (1000 Trades Backtested)
- Day 6 / 7 / 8 / 9 / 10 / 11 Live Trading
- FundingPips Success Story Interview: $6,719 Rewards
- 3 Reasons Your Stop Loss Gets Ignored Every Single Time
- High Win Rate Day Trading Strategy = $$$ (zOofwXbil94)
- If My Negative RR Strategy Won't Make You Profitable
- The Only Trading Blueprint You Need... (3CV04RQ5MCc)
- This Session is Made For Negative RR Traders
- The Scalping Strategy I Use Every Day...
- The ONLY Day Trading Strategy you'll EVER Need (Sept 3 2025)
- https://www.youtube.com/watch?v=2qJMaF_nhUc
- https://www.youtube.com/watch?v=zOofwXbil94
- https://www.youtube.com/watch?v=3CV04RQ5MCc
