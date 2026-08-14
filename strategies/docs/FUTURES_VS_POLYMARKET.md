# CHANGE_SUMMARY
# 2026-08-14  coder
#   - Created docs/FUTURES_VS_POLYMARKET.md documenting the split between the
#     futures-native signals (strategies/signals/) and the archived Polymarket
#     binary-contract edition (strategies/signals_polymarket/).
# WHY: The four blueprints are futures intraday strategies; the binary
#      signature was a legacy adaptation now isolated so each can be tested
#      against the market it actually trades.
# WHY: Prevent future agents from re-introducing binary semantics into the
#      futures backtest and vice-versa.

# Futures vs Polymarket: Signal Differentiation

## TL;DR

| Aspect | `strategies/signals/` (ACTIVE) | `strategies/signals_polymarket/` (ARCHIVED) |
|---|---|---|
| Purpose | Futures / normal trading backtests | Polymarket binary contracts (reference) |
| Direction vocabulary | `LONG` / `SHORT` | `YES` / `NO` |
| Entry price | Market price (`spot_price` / MC close) | Binary ask (`yes_ask` / `no_ask`), fallback `yp` / `np_val` |
| Price cap | None | `max_entry_price=0.85` (binary prices are 0..1) |
| Time gate | Session windows via `time_utils` (ET/UTC) | `rem_sec` countdown + `time_gate_seconds` |
| Binary-only kwargs | Removed | `yp`, `np_val`, `yes_ask`, `no_ask`, `rem_sec`, `tf_hint`, `market_id`, `max_entry_price`, `time_gate_seconds` |
| `pip_value` | Maps pips → index points (default 1.0 = 1 point) | Same concept (0.1 for BTC etc.) |
| State store keys | `entry_count`/`cooldown` keyed `LONG`/`SHORT` | Keyed `YES`/`NO` |
| Signal logic | Identical to archived (filters, phases, exits) | Identical to active |

## Why this split exists

The four StarTrading blueprints (15-min range scalp, negative-RR consolidation
sweeper, 00:00 UTC MOS session draw, post-8AM BPR magnet) were **designed as
futures intraday strategies** — they reference daily/4h/15m/5m/1m structure,
sessions (8:30 ET, 14:00 ET, 08:00 ET, 00:00 UTC), index levels (PDH/PDL,
swing highs/lows), and point/pip-based risk.

They were originally written against a **Polymarket binary-contract
signature**: direction as `YES`/`NO`, entry at a binary `yes_ask`/`no_ask`,
prices bounded 0..1 with a `0.85` cap, and a `rem_sec` countdown gate. That
adaptation was convenient for reuse but is semantically wrong for futures
(index prices are thousands of points, no expiry countdown, no yes/no side).

## What changed in the futures edition (2026-08-14)

1. **Directions are `LONG`/`SHORT`.**  All internal state (`entry_count`,
   `cooldown`) and the returned `direction` use the futures vocabulary.  The
   backtest engine maps `LONG→+1`, `SHORT→-1` in the trades CSV.
2. **Entry is at the market price.**  Binary asks are gone; entry uses
   `spot_price` (the signal bar's close) — Blueprint 1 still uses the movement
   candle close per its blueprint.  No 0..1 range, no cap.
3. **No `rem_sec` / `time_gate_seconds`.**  Session gating is purely from
   `time_utils` (ET/UTC), which the backtest engine monkeypatches to replay
   historical bar time.  Futures don't expire.
4. **Dropped binary kwargs.**  `yp`, `np_val`, `yes_ask`, `no_ask`, `tf_hint`,
   `market_id`, `max_entry_price`, `time_gate_seconds` removed from signatures.
5. **`pip_value` means index points.**  Default 1.0 = 1 point (NQ/ES/YM), so
   Blueprint 3's 10-pip TP cap and Blueprint 4's 2/5-pip TP/SL are expressed in
   raw points.  Point *dollar* values (NQ=20, ES=50, YM=5) are handled by the
   metrics layer, not the signals.
6. **Shared store is direction-agnostic.**  `StateStore.tick_cooldowns`
   iterates whatever keys exist, so both editions work.

## What stayed identical

All market-structure logic is byte-for-byte the same: HTF bias
(`is_established_movement`), FVG/BPR/DIRTY_BPR detection, EQH/EQL, protective
swing, range framing, liquidity-sweep and movement-candle filters, orderflow
FVG confirmation, recovery loop, NFP/early-month abort, doji aborts, PDH/PDL
sweep checks, and the pip/point SL-TP math.

## How to use

- **Futures backtests (GHA + laptop):** import from `strategies.signals`.  The
  engine (`strategies/backtest/engine.py`) and CLI (`run_backtest.py`) already
  target these.  See `docs/BLUEPRINTS.md` for strategy design.
- **Polymarket (reference only):** import from `strategies.signals_polymarket`.
  Do not use these in the futures backtest — they need binary prices and a
  `rem_sec` countdown that the harness does not supply.

## File map

- `strategies/signals/{common,fifteen_min_range_scalp,negative_rr_consolidation_sweeper,mos_session_daily_draw,post_8am_bpr_magnet}.py` — ACTIVE futures editions.
- `strategies/signals_polymarket/` — ARCHIVED binary-contract editions (verbatim originals + this doc's split rationale in `__init__.py`).
- `strategies/core/{time_utils,candle_utils,detectors,state_store}.py` — shared primitives used by both.
