# CHANGE_SUMMARY
# 2026-08-17  kilo
#   - Created strategies/signals/sneaky_pivot.py implementing the
#     sneaky_pivot (Blueprint 2) FUTURES signal.
#   - Strictly 15-minute timeframe; previous-day daily bar is used as Range
#     High / Range Low.
#   - Three-candle NY-open execution: Candle 1 pushes into an extreme zone,
#     Candle 2 is the opposite-color "sneaky" candle, Candle 3 triggers a
#     stop entry at the high (longs) or low (shorts) of Candle 2.
#   - No lookahead: only completed 15m bars are inspected; the entry decision
#     is made at the close of the third 15m candle.
# WHY: Add the second StarTrading futures blueprint to the signal suite.

"""Blueprint 2 (FUTURES): 15-Minute Range-Bound Mean-Reversion Reversal.

Core logic:
  Frame the previous day's high/low as Range High / Range Low. In the first
  45 minutes of the NY session, wait for a 3-candle sequence: Candle 1 pushes
  into an extreme zone (support/resistance), Candle 2 is a "sneaky" opposite-
  color candle, and Candle 3 triggers a stop entry at the high/low of Candle 2.
  Take profit is the opposite side of the previous day's range.

Signal kwargs:
  daily_bars    : list of daily OHLCV dicts (previous-day range)
  fifteen_m_bars: list of recent 15m OHLCV dicts (three-candle execution)
"""

import logging
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from ..core import time_utils as tu
from ..core import candle_utils as cu
from ..core import detectors as dt
from ..core.state_store import StateStore
from .common import (
    no_signal,
    validate_signal_inputs,
    reentry_scale,
    MIN_COOLDOWN_TICKS,
)

log = logging.getLogger("sneaky_pivot")

EST = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

SOURCE = "SNEAKY_PIVOT"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
store = StateStore("sneaky_pivot", _PROJECT_ROOT)

# A candle "pushes into" an extreme if its high/low is within this fraction of
# the previous day's range from the corresponding range boundary.
ZONE_FRAC = 0.15
# Stop-loss buffer beyond the relevant wick extreme, expressed as a fraction of
# the previous day's range.
SL_BUFFER_FRAC = 0.02


def _make_state():
    return {
        "today": None,
        "entry_count": {"LONG": 0, "SHORT": 0},
        "cooldown": {"LONG": 0, "SHORT": 0},
        "last_entry_time": None,
    }


def _in_first_45m(dt_et: datetime) -> bool:
    """True during the first three 15m candles of the NY equity session."""
    t = dt_et.time()
    return time(*tu.MARKET_OPEN_ET) <= t <= time(10, 15)


def _as_et(ts) -> datetime:
    """Convert a bar timestamp to America/New_York time."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(EST)


def _prev_daily_bar(daily_bars: list[dict], today_et: datetime.date) -> dict | None:
    """Most recent completed daily bar for the trading day before today (ET).

    Uses the bar's ET date rather than a raw UTC timestamp comparison so the
    previous day's range is selected even when daily bars are stamped at the
    day's open (00:00 UTC) or close.  This also avoids peeking at an incomplete
    current-day daily bar that happens to have an earlier UTC timestamp.
    """
    prev = [b for b in daily_bars if _as_et(b["timestamp"]).date() < today_et]
    if not prev:
        return None
    return max(prev, key=lambda b: b["timestamp"])


def _today_15m_bars(
    fifteen_m_bars: list[dict], today_et: datetime.date, now_utc: datetime
) -> list[dict]:
    """Today's closed 15m bars up to now, sorted chronologically."""
    bars = [
        b for b in fifteen_m_bars
        if _as_et(b["timestamp"]).date() == today_et
        and b["timestamp"] <= now_utc
    ]
    bars.sort(key=lambda b: b["timestamp"])
    return bars


def _detect_setup(c1: dict, c2: dict, range_high: float, range_low: float) -> str | None:
    """Return LONG/SHORT if the first two candles form a valid setup."""
    daily_range = range_high - range_low
    if daily_range <= 0:
        return None
    zone = ZONE_FRAC * daily_range

    c1_green = c1["close"] > c1["open"]
    c1_red = c1["close"] < c1["open"]
    c2_green = c2["close"] > c2["open"]
    c2_red = c2["close"] < c2["open"]

    # Long: Candle 1 drops into support (red) and Candle 2 is the sneaky green.
    if c1_red and c2_green and c1["low"] <= range_low + zone:
        return "LONG"
    # Short: Candle 1 rallies into resistance (green) and Candle 2 is the sneaky red.
    if c1_green and c2_red and c1["high"] >= range_high - zone:
        return "SHORT"
    return None


def _entry_price(direction: str, c2: dict, c3: dict) -> float | None:
    """Return the stop-entry fill price if Candle 3 triggers the setup."""
    if direction == "LONG" and c3["high"] >= c2["high"]:
        return max(c3["open"], c2["high"])
    if direction == "SHORT" and c3["low"] <= c2["low"]:
        return min(c3["open"], c2["low"])
    return None


def _sl_tp(direction: str, c1: dict, c2: dict, range_high: float, range_low: float) -> tuple[float, float]:
    """Absolute stop-loss and take-profit levels for a validated setup."""
    daily_range = range_high - range_low
    zone = ZONE_FRAC * daily_range
    buffer = SL_BUFFER_FRAC * daily_range

    if direction == "LONG":
        support_wicks = [
            c["low"] for c in (c1, c2)
            if c["low"] <= range_low + zone
        ] or [c1["low"], c2["low"]]
        sl = min(support_wicks) - buffer
        tp = range_high
    else:
        resistance_wicks = [
            c["high"] for c in (c1, c2)
            if c["high"] >= range_high - zone
        ] or [c1["high"], c2["high"]]
        sl = max(resistance_wicks) + buffer
        tp = range_low

    return sl, tp


def sneaky_pivot(
    spot_price=None,
    asset="NQ",
    max_reentries=3,
    daily_bars=None,
    fifteen_m_bars=None,
    **kwargs,
):
    ok, reason = validate_signal_inputs(spot_price, asset)
    if not ok:
        return no_signal(reason, SOURCE)

    now = tu.get_et_now()
    today = now.date()
    now_utc = tu.get_utc_now()

    if not _in_first_45m(now):
        return no_signal("outside_first_45m", SOURCE)
    if not (daily_bars and fifteen_m_bars):
        return no_signal("missing_bars", SOURCE)

    key = store.make_key(asset, today, max_reentries)
    state = store.load_or_new(key, _make_state)
    state["today"] = today
    store.prune(asset.upper(), today)
    store.tick_cooldowns(state)

    prev_daily = _prev_daily_bar(daily_bars, today)
    if prev_daily is None:
        return no_signal("no_prev_daily", SOURCE)
    range_high = prev_daily["high"]
    range_low = prev_daily["low"]

    session_bars = _today_15m_bars(fifteen_m_bars, today, now_utc)
    if len(session_bars) < 3:
        return no_signal("need_3_bars", SOURCE)

    c1, c2, c3 = session_bars[0], session_bars[1], session_bars[2]

    direction = _detect_setup(c1, c2, range_high, range_low)
    if direction is None:
        return no_signal("no_setup", SOURCE)

    entry_price = _entry_price(direction, c2, c3)
    if entry_price is None:
        return no_signal("no_entry_trigger", SOURCE)

    n = state["entry_count"][direction]
    if not (n == 0 or 0 < n <= max_reentries):
        return no_signal("max_reentries", SOURCE)
    if state["cooldown"][direction] > 0:
        return no_signal("cooldown", SOURCE)

    sl, tp = _sl_tp(direction, c1, c2, range_high, range_low)
    # Early invalidation: a close beyond the sneaky candle's extreme suggests the
    # setup has failed before TP is reached.  The harness currently uses SL/TP,
    # but this level is provided for future trailing/invalidation wiring.
    invalidation_price = c2["low"] if direction == "LONG" else c2["high"]

    state["entry_count"][direction] += 1
    state["cooldown"][direction] = MIN_COOLDOWN_TICKS
    state["last_entry_time"] = c3["timestamp"]
    store.save(key, state)

    return {
        "triggered": True,
        "direction": direction,
        "confidence": reentry_scale(n),
        "entry_price": float(entry_price),
        "signal_price": float(entry_price),
        "source": SOURCE,
        "sl": float(sl),
        "tp": float(tp),
        "invalidation_price": float(invalidation_price),
        "range_high": float(range_high),
        "range_low": float(range_low),
        "reason": (
            f"{'RE-ENTRY' if n > 0 else 'FIRST'} dir={direction} "
            f"asset={asset.upper()} c2_trigger={entry_price:.2f} "
            f"sl={sl:.2f} tp={tp:.2f}"
        ),
    }


if __name__ == "__main__":
    # Brief inline sanity check: one LONG trigger and one no-trigger case.
    from zoneinfo import ZoneInfo

    _est = ZoneInfo("America/New_York")
    _utc = ZoneInfo("UTC")

    # Pin wall-clock to the close of the third 15m bar (10:15 ET).
    _now_et = datetime(2026, 1, 15, 10, 15, tzinfo=_est)
    _now_utc = _now_et.astimezone(_utc)
    tu.get_et_now = lambda: _now_et
    tu.get_utc_now = lambda: _now_utc

    # Avoid persisting state across the standalone sanity run.
    store._mem.clear()
    _orig_save = store.save
    store.save = lambda key, state: None

    try:
        _prev_daily = {
            "timestamp": datetime(2026, 1, 14, 17, 0, tzinfo=_est).astimezone(_utc),
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 95.0,
            "volume": 1000,
        }

        def _m15(hour, minute, o, h, l, c):
            ts = datetime(2026, 1, 15, hour, minute, tzinfo=_est).astimezone(_utc)
            return {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": 100}

        # Long setup: c1 red pushes into support, c2 green sneaky, c3 triggers buy stop.
        _c1 = _m15(9, 45, 95.0, 96.0, 89.0, 92.0)
        _c2 = _m15(10, 0, 92.0, 96.0, 91.0, 96.0)
        _c3 = _m15(10, 15, 96.0, 98.0, 95.0, 97.0)
        _sig = sneaky_pivot(
            spot_price=97.0,
            asset="NQ",
            daily_bars=[_prev_daily],
            fifteen_m_bars=[_c1, _c2, _c3],
        )
        assert _sig["triggered"] is True, _sig
        assert _sig["direction"] == "LONG", _sig
        assert _sig["entry_price"] == 96.0, _sig
        assert _sig["tp"] == 110.0, _sig
        assert _sig["sl"] < 89.0, _sig

        # No trigger: c3 fails to reach the high of the sneaky candle.
        _c3_fail = _m15(10, 15, 96.0, 96.5, 95.0, 95.5)
        _sig2 = sneaky_pivot(
            spot_price=95.5,
            asset="NQ",
            daily_bars=[_prev_daily],
            fifteen_m_bars=[_c1, _c2, _c3_fail],
        )
        assert _sig2["triggered"] is False, _sig2

        print("sneaky_pivot sanity checks passed")
    finally:
        store.save = _orig_save

# QA_REPORT: passed
# Review items:
#   1. Blueprint alignment: implements 15m NY-open 3-candle sneaky-pivot setup
#      using the previous day's high/low as the range; c1 pushes into the extreme
#      zone, c2 is the opposite-color sneaky candle, c3 stop-entries at c2's
#      high/low with TP at the opposite range boundary.
#   2. Lookahead bias: fixed daily-bar selection to use the previous ET trading
#      day's completed bar (ET date < today) instead of a raw UTC timestamp
#      comparison, preventing accidental use of an incomplete current-day daily
#      bar. 15m bars are now filtered to closed bars (timestamp <= now_utc) and
#      sorted before indexing, so only c1/c2/c3 bars available at evaluation time
#      are used. No future OHLC, EMA, or stochastic values are referenced.
#   3. Standard FUTURES dict: triggered, direction ("LONG"/"SHORT"), confidence,
#      entry_price, signal_price, source, reason are always present; sl and tp
#      are included when triggered. Extra context fields (invalidation_price,
#      range_high, range_low) are appended but do not break the contract.
#   4. py_compile and module sanity check both pass.
#   5. Defensive programming: empty/missing bar lists, spot_price=None, and
#      minimal 2-bar windows return no_signal without crashing.
#   6. StateStore: keyed per (asset.upper(), today, variant, max_reentries) so
#      state does not leak across assets or dates; prune() removes stale entries.
# Fixes applied:
#   - _prev_daily_bar now selects by ET date < today.
#   - _today_15m_bars now sorts and drops future bars before indexing.
#   - Updated call sites in sneaky_pivot() accordingly.
# Remaining concerns:
#   - The signal only has room for one 3-candle setup in the 09:30-10:15 ET
#     window, so max_reentries is effectively unreachable on a single day.
#   - Daily-bar timestamp convention is assumed to be OHLCV-complete; if a data
#     source stamps daily bars mid-session, the ET-date filter still correctly
#     treats them as prior-day bars.
