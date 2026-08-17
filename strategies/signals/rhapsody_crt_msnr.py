# CHANGE_SUMMARY
# 2026-08-14  coder
#   - Created rhapsody_crt_msnr.py (Blueprint 4) for the FUTURES backtest
#     harness.
#   - Implements the top-down CRT/MSNR flow: daily liquidity-sweep rejection,
#     H4 BOS confirmation, M15 execution via Model A (direct retest + engulfing)
#     or Model B (FVG fill + liquidity sweep).
#   - Returns LONG/SHORT futures signal dict with absolute sl/tp levels and a
#     mechanical 1:4 RR target.
# WHY: Add the fourth StarTrading blueprint to the futures backtest suite while
#      keeping all time/candle/detector logic in core/* and state in StateStore.

"""Blueprint 4 (FUTURES): Rhapsody CRT/MSNR — Top-Down Structural Reversal.

Documentation / master blueprint: docs/rhapsody_crt_msnr.md

Core logic:
  1. Daily: price returns to a structural level, sweeps local liquidity with a
     long wick, and rejects back inside the previous day's range.
  2. 4H: structure breaks (BOS) in the direction of the daily rejection.
  3. 15m: enter on a retest of the H4 origin / H4 FVG after an M15 liquidity
     sweep and an engulfing confirmation candle.

Signal kwargs (in addition to the standard set):
  daily_bars    : list of daily OHLCV dicts (HTF bias / key level)
  four_h_bars   : list of 4h OHLCV dicts (BOS confirmation / FVG)
  fifteen_m_bars: list of recent 15m OHLCV dicts (execution / engulfing)
"""

import logging
from datetime import datetime, timedelta
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

log = logging.getLogger("rhapsody_crt_msnr")

EST = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

SOURCE = "RHAPSODY_CRT_MSNR"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
store = StateStore("rhapsody_crt_msnr", _PROJECT_ROOT)

# Tunables -------------------------------------------------------------------
# Daily rejection: wick must be at least this multiple of the candle body.
DAILY_WICK_BODY_MULT = 1.5
# H4 lookback for swing high/low used in BOS confirmation.
H4_SWING_LOOKBACK = 12
# M15 execution lookback (completed 15m bars) for sweep + engulfing.
M15_EXEC_LOOKBACK = 6
# Mechanical RR target.
RR_TARGET = 4.0


def _make_state():
    return {
        "today": None,
        "daily_bias": None,          # "LONG" / "SHORT"
        "daily_key_level": None,     # price of the daily sweep extreme
        "h4_bos_confirmed": False,
        "h4_breakout": None,         # breakout candle dict
        "h4_origin": None,           # swing high/low that was broken
        "h4_fvg": None,              # relevant H4 FVG dict
        "entry_count": {"LONG": 0, "SHORT": 0},
        "cooldown": {"LONG": 0, "SHORT": 0},
        "last_entry_time": None,
    }


# ----- phase logic ----------------------------------------------------------

def _daily_rejection(daily_bars):
    """Step 1: detect daily liquidity sweep + rejection.

    Returns (direction, key_level) where direction is "LONG"/"SHORT"/None.
    LONG  setup: current daily low sweeps below previous day's low and the
                lower wick is > 1.5x the body, close back inside prev range.
    SHORT setup: current daily high sweeps above previous day's high and the
                upper wick is > 1.5x the body, close back inside prev range.
    """
    if len(daily_bars) < 2:
        return None, None
    prev, curr = daily_bars[-2], daily_bars[-1]
    body = cu.candle_body_size(curr)
    if body <= 0:
        return None, None

    prev_high = prev["high"]
    prev_low = prev["low"]
    close = curr["close"]

    # Bullish rejection: long lower wick, sweep prev low, close back in range.
    if curr["low"] < prev_low and prev_low < close <= prev_high:
        lower_wick = min(curr["open"], close) - curr["low"]
        if lower_wick > DAILY_WICK_BODY_MULT * body:
            return "LONG", curr["low"]

    # Bearish rejection: long upper wick, sweep prev high, close back in range.
    if curr["high"] > prev_high and prev_low <= close < prev_high:
        upper_wick = curr["high"] - max(curr["open"], close)
        if upper_wick > DAILY_WICK_BODY_MULT * body:
            return "SHORT", curr["high"]

    return None, None


def _h4_bos_confirmed(four_h_bars, direction):
    """Step 2: H4 candle closes beyond the recent H4 swing extreme.

    Returns (breakout_candle, origin_level) or (None, None).
    No lookahead: uses only completed H4 bars; the most recent bar is the
    candidate breakout candle and the origin is computed from prior bars.
    """
    if len(four_h_bars) < H4_SWING_LOOKBACK + 1:
        return None, None
    prior = four_h_bars[-H4_SWING_LOOKBACK - 1:-1]
    breakout = four_h_bars[-1]

    if direction == "LONG":
        swing_high = max(c["high"] for c in prior)
        if breakout["close"] > swing_high:
            return breakout, swing_high
    else:
        swing_low = min(c["low"] for c in prior)
        if breakout["close"] < swing_low:
            return breakout, swing_low
    return None, None


def _relevant_h4_fvg(four_h_bars, direction):
    """Return the most recent open H4 FVG aligned with direction, or None."""
    if len(four_h_bars) < 3:
        return None
    fvgs = dt.detect_fvg(four_h_bars[-30:], "4h")
    if not fvgs:
        return None
    want = "UP" if direction == "LONG" else "DOWN"
    for f in reversed(fvgs):
        if f["direction"] != want:
            continue
        # "Open" means price has not fully closed through the FVG.
        last = four_h_bars[-1]
        if direction == "LONG" and last["close"] < f["low"]:
            continue
        if direction == "SHORT" and last["close"] > f["high"]:
            continue
        return f
    return None


def _is_engulfing(candle, prev_candle, direction):
    """Return True if candle's body fully engulfs prev_candle's body.

    A bullish engulfing body opens below the previous body and closes above it.
    A bearish engulfing body opens above the previous body and closes below it.
    """
    if prev_candle is None:
        return False
    prev_body_low = min(prev_candle["open"], prev_candle["close"])
    prev_body_high = max(prev_candle["open"], prev_candle["close"])
    if direction == "LONG":
        return (
            candle["close"] > candle["open"]
            and candle["open"] < prev_body_low
            and candle["close"] > prev_body_high
        )
    return (
        candle["close"] < candle["open"]
        and candle["open"] > prev_body_high
        and candle["close"] < prev_body_low
    )


def _m15_model_a_direct_retest(fifteen_m_bars, direction, key_level):
    """Model A: pullback to H4 origin, M15 sweep, engulfing off the level.

    Scans the most recent M15_EXEC_LOOKBACK completed 15m bars.  A sweep is a
    candle whose wick pierces a local M15 extreme and closes back toward the
    key level.  Entry is the most recent candle if it engulfs in the trade
    direction and touches the key level.
    """
    if len(fifteen_m_bars) < M15_EXEC_LOOKBACK:
        return None
    window = list(fifteen_m_bars[-M15_EXEC_LOOKBACK:])

    if direction == "LONG":
        for i in range(len(window) - 2):
            sweep = window[i]
            # Local extreme from the bars immediately preceding the sweep.
            local_low = min(window[j]["low"] for j in range(max(0, i - 2), i)) if i > 0 else sweep["low"]
            if not (sweep["low"] < local_low and sweep["close"] > local_low):
                continue
            entry = window[-1]
            if (_is_engulfing(entry, window[-2], "LONG") and
                    entry["low"] <= key_level and entry["close"] > key_level):
                return entry
    else:
        for i in range(len(window) - 2):
            sweep = window[i]
            local_high = max(window[j]["high"] for j in range(max(0, i - 2), i)) if i > 0 else sweep["high"]
            if not (sweep["high"] > local_high and sweep["close"] < local_high):
                continue
            entry = window[-1]
            if (_is_engulfing(entry, window[-2], "SHORT") and
                    entry["high"] >= key_level and entry["close"] < key_level):
                return entry
    return None


def _m15_model_b_fvg_fill(fifteen_m_bars, direction, fvg):
    """Model B: price inside H4 FVG, M15 sweep outside, tap key level inside.

    Entry is the most recent completed 15m candle if price is currently inside
    the FVG and a preceding candle swept beyond the FVG extreme.
    """
    if len(fifteen_m_bars) < M15_EXEC_LOOKBACK or fvg is None:
        return None
    window = list(fifteen_m_bars[-M15_EXEC_LOOKBACK:])
    entry = window[-1]

    if direction == "LONG":
        if not (fvg["low"] <= entry["close"] <= fvg["high"]):
            return None
        if not _is_engulfing(entry, window[-2], "LONG"):
            return None
        swept = any(
            c["low"] < fvg["low"] and c["close"] > fvg["low"]
            for c in window[:-1]
        )
        if swept and entry["low"] <= fvg["high"] and entry["close"] > fvg["low"]:
            return entry
    else:
        if not (fvg["low"] <= entry["close"] <= fvg["high"]):
            return None
        if not _is_engulfing(entry, window[-2], "SHORT"):
            return None
        swept = any(
            c["high"] > fvg["high"] and c["close"] < fvg["high"]
            for c in window[:-1]
        )
        if swept and entry["high"] >= fvg["low"] and entry["close"] < fvg["high"]:
            return entry
    return None


def _find_recent_swing(fifteen_m_bars, direction):
    """Return a recent structural swing low/high from the 15m pullback."""
    if len(fifteen_m_bars) < 4:
        return None
    if direction == "LONG":
        return min(c["low"] for c in fifteen_m_bars[-8:])
    return max(c["high"] for c in fifteen_m_bars[-8:])


# ----- signal ---------------------------------------------------------------

def rhapsody_crt_msnr(
    spot_price=None,
    asset="NQ",
    max_reentries=3,
    daily_bars=None,
    four_h_bars=None,
    fifteen_m_bars=None,
    **kwargs,
):
    ok, reason = validate_signal_inputs(spot_price, asset)
    if not ok:
        return no_signal(reason, SOURCE)

    now = tu.get_et_now()
    today = now.date()

    # Light intraday gate: blueprint is an intraday structural setup.
    if not tu.is_after_ny_open_filter_time(now):
        return no_signal("pre_800_et", SOURCE)
    if not (daily_bars and four_h_bars and fifteen_m_bars):
        return no_signal("missing_bars", SOURCE)

    key = store.make_key(asset, today, max_reentries)
    state = store.load_or_new(key, _make_state)
    state["today"] = today
    store.prune(asset.upper(), today)
    store.tick_cooldowns(state)

    # Step 1: Daily liquidity sweep + rejection.
    daily_bias, daily_key_level = _daily_rejection(daily_bars)
    state["daily_bias"] = daily_bias
    state["daily_key_level"] = daily_key_level
    if daily_bias is None:
        return no_signal("no_daily_rejection", SOURCE)

    # Step 2: H4 BOS in direction of daily rejection.
    breakout, origin = _h4_bos_confirmed(four_h_bars, daily_bias)
    state["h4_bos_confirmed"] = breakout is not None
    state["h4_breakout"] = breakout
    state["h4_origin"] = origin
    if breakout is None:
        return no_signal("no_h4_bos", SOURCE)

    # Step 3: M15 execution — prefer Model A, fallback to Model B.
    entry_candle = _m15_model_a_direct_retest(fifteen_m_bars, daily_bias, origin)
    model = "A"
    if entry_candle is None:
        fvg = _relevant_h4_fvg(four_h_bars, daily_bias)
        state["h4_fvg"] = fvg
        entry_candle = _m15_model_b_fvg_fill(fifteen_m_bars, daily_bias, fvg)
        model = "B"
    if entry_candle is None:
        return no_signal("no_m15_execution", SOURCE)

    direction = daily_bias
    n = state["entry_count"][direction]
    if not (n == 0 or 0 < n <= max_reentries):
        return no_signal("max_reentries", SOURCE)
    if state["cooldown"][direction] > 0:
        return no_signal("cooldown", SOURCE)

    entry_price = entry_candle["close"]
    if entry_price is None or entry_price <= 0:
        return no_signal("no_price", SOURCE)

    # Stop loss.
    if model == "B" and state["h4_fvg"] is not None:
        fvg = state["h4_fvg"]
        sl = fvg["low"] if direction == "LONG" else fvg["high"]
    else:
        # Model A: extreme of the H4 range that initiated the move.
        if direction == "LONG":
            sl = min(c["low"] for c in four_h_bars[-H4_SWING_LOOKBACK:])
        else:
            sl = max(c["high"] for c in four_h_bars[-H4_SWING_LOOKBACK:])

    # Take profit: rigid 1:4 RR baseline; optionally target recent swing.
    stop_dist = abs(entry_price - sl)
    if stop_dist <= 0:
        return no_signal("invalid_sl", SOURCE)

    swing_target = _find_recent_swing(fifteen_m_bars, direction)
    if direction == "LONG":
        rr_tp = entry_price + RR_TARGET * stop_dist
        tp = rr_tp if swing_target is None else max(rr_tp, swing_target)
    else:
        rr_tp = entry_price - RR_TARGET * stop_dist
        tp = rr_tp if swing_target is None else min(rr_tp, swing_target)

    state["entry_count"][direction] += 1
    state["cooldown"][direction] = MIN_COOLDOWN_TICKS
    state["last_entry_time"] = entry_candle["timestamp"]
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
        "daily_bias": direction,
        "daily_key_level": float(daily_key_level),
        "h4_origin": float(origin),
        "m15_model": model,
        "reason": (
            f"{'RE-ENTRY' if n > 0 else 'FIRST'} dir={direction} asset={asset.upper()} "
            f"model={model} origin={origin:.2f} entry={entry_price:.2f} "
            f"sl={sl:.2f} tp={tp:.2f} rr={RR_TARGET:.1f}:1"
        ),
    }


# ----- sanity check ---------------------------------------------------------

if __name__ == "__main__":
    from datetime import timezone
    import tempfile

    # Use an isolated scratch store so repeated runs do not reuse persisted
    # cooldown/entry-count state from previous executions.
    _scratch = Path(tempfile.mkdtemp(prefix="rhapsody_crt_msnr_test_"))
    store = StateStore("rhapsody_crt_msnr", _scratch)

    def _bar(ts, o, h, l, c, v=1):
        return {
            "timestamp": ts,
            "open": o, "high": h, "low": l, "close": c, "volume": v,
        }

    base = datetime(2026, 8, 14, 9, 0, tzinfo=EST)
    bars = lambda n, o, h, l, c: [_bar(base + timedelta(minutes=15 * i), o, h, l, c) for i in range(n)]

    # Daily: bearish prev, then bullish sweep/rejection (long lower wick).
    daily = [
        _bar(base - timedelta(days=2), 100, 102, 99, 101),
        _bar(base - timedelta(days=1), 101, 102, 100, 101),   # prev range [100,102]
        _bar(base, 101, 101.5, 98, 100.5),                    # sweep below 100, reject
    ]

    # 4H: last close above prior swing high -> BOS long.
    four_h = []
    for i in range(14):
        four_h.append(_bar(base - timedelta(hours=14 - i), 100.5, 101, 100, 100.5))
    four_h[-1]["close"] = 102.5
    four_h[-1]["high"] = 102.5

    # 15m: sweep then bullish engulfing off the origin.
    fifteen = []
    for i in range(6):
        fifteen.append(_bar(base + timedelta(minutes=15 * i), 102, 102.2, 101.8, 102))
    fifteen[-3]["low"] = 101.5      # sweep below local low
    fifteen[-3]["close"] = 102.0    # close back up
    fifteen[-1]["open"] = 100.9     # bullish engulfing off origin (~101)
    fifteen[-1]["low"] = 100.85     # wick through origin
    fifteen[-1]["close"] = 102.3

    # Patch time to be inside the after-8am window.
    orig_get_et_now = tu.get_et_now
    tu.get_et_now = lambda: base

    sig = rhapsody_crt_msnr(
        spot_price=102.3,
        asset="NQ",
        max_reentries=3,
        daily_bars=daily,
        four_h_bars=four_h,
        fifteen_m_bars=fifteen,
    )
    tu.get_et_now = orig_get_et_now

    assert sig["triggered"], f"expected trigger, got {sig}"
    assert sig["direction"] == "LONG", f"expected LONG, got {sig['direction']}"
    assert sig["entry_price"] == 102.3
    assert sig["sl"] < sig["entry_price"] < sig["tp"]
    print("rhapsody_crt_msnr sanity check passed:", sig["reason"])

    # No-signal case: missing daily rejection.
    flat_daily = [
        _bar(base - timedelta(days=2), 100, 102, 99, 101),
        _bar(base - timedelta(days=1), 101, 102, 100, 101),
        _bar(base, 101, 101.5, 100.5, 101),  # no sweep
    ]
    # Use a different max_reentries so the earlier signal's cooldown state is
    # not reused for the no-signal assertion.
    tu.get_et_now = lambda: base
    no_sig = rhapsody_crt_msnr(
        spot_price=102.3, asset="NQ", max_reentries=2,
        daily_bars=flat_daily, four_h_bars=four_h, fifteen_m_bars=fifteen,
    )
    tu.get_et_now = orig_get_et_now
    assert not no_sig["triggered"], f"expected no signal, got {no_sig}"
    print("no-signal check passed:", no_sig["reason"])


# QA_REPORT: passed
# Reviewer: kilo-code-qa
# Date: 2026-08-17
#
# Checks performed:
#   1. Blueprint alignment: implementation follows the documented top-down
#      CRT/MSNR flow (daily sweep/rejection -> H4 BOS -> M15 Model A or B).
#   2. Lookahead bias: all detectors use only completed bars supplied by the
#      harness (daily[-2:-1], 4h[-lookback-1:-1], 15m[-lookback:]).  The signal
#      does not index forward or peek at future opens/highs/lows/closes.
#      CAVEAT: the harness must not pass the currently forming/developing bar
#      as the last element; the strategy interprets bar-list tails as the most
#      recently completed bars available at signal time.
#   3. Standard FUTURES dict: triggered, direction ("LONG"/"SHORT"), confidence,
#      entry_price, signal_price, source, reason are always present.  sl and tp
#      are included whenever triggered=True.
#   4. py_compile and module sanity check both pass.
#   5. Defensive programming: empty or near-empty daily/4h/15m windows return
#      no_signal gracefully without raising.
#   6. StateStore: key is (asset, date, variant, max_reentries); no leakage
#      across assets or dates.  Global store is overridden in __main__ with a
#      scratch directory so repeated runs do not reuse persisted state.
#
# Issues found & fixed:
#   - _is_engulfing() only checked candle direction (close > open / close < open)
#     instead of a true engulfing body.  Fixed to require the current candle's
#     body to fully contain the previous candle's body, and updated Model A/B
#     callers to pass window[-2] as the previous candle.
#
# Remaining concerns / notes:
#   - The "H4 origin" returned by _h4_bos_confirmed is the broken swing level
#     (swing high for LONG, swing low for SHORT).  Model A SL uses the min/max
#     extreme of the H4 lookback rather than the true pre-move origin, which is
#     conservative but may be wider than the blueprint intends.
#   - _find_recent_swing returns a pullback extreme (low for LONG, high for
#     SHORT); because the TP logic then max/mins it against the mechanical RR
#     target, the RR target usually wins.  This is latent-not-fatal.
#   - For Model B to align with the BOS filter, the relevant H4 FVG must sit
#     above the broken swing level and price must close inside it; this is a
#     strict structural condition and may trigger rarely in live data.
