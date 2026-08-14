# CHANGE_SUMMARY
# 2026-08-14  coder
#   - REWRITTEN as the FUTURES/normal-trading version of Blueprint 1.
#   - Direction is LONG/SHORT (no more YES/NO); entry is the market price
#     (movement-candle close / spot), not a binary ask.
#   - Removed Polymarket-only params (yp, np_val, yes_ask, no_ask, rem_sec,
#     max_entry_price, time_gate_seconds). Session gate stays: post-8:30 ET
#     and before the 14:00 ET hard exit (via time_utils).
#   - State keys use LONG/SHORT.  SL/TP unchanged (sweep level + 15m range).
#   - The Polymarket-flavored original is archived verbatim in
#     strategies/signals_polymarket/fifteen_min_range_scalp.py.
# WHY: Futures-native contract for the 12-instrument backtest; see
#      docs/FUTURES_VS_POLYMARKET.md.
#
# 2026-08-14  kilo
#   - Created signals/fifteen_min_range_scalp.py (Blueprint 1). Refactored from
#     the flat module into the compartmentalized package: time/candle/detector
#     logic lives in core/*, persistence in core.state_store.StateStore, shared
#     contract in signals/common. This file is now a thin orchestrator.
#   - Implements the 4-phase system: HTF bias, 15m range framing, 1m liquidity
#     sweep post-8:30 ET, Movement Candle entry. State keyed per (asset,date).
# WHY: Compartmentalization; see docs/fifteen_min_range_scalp.md.

"""Blueprint 1 (FUTURES): 15-Minute Range & 1-Minute Movement Candle Scalp.

Documentation / master blueprint / changelog: docs/fifteen_min_range_scalp.md

Core logic:
  Frame an immediate localized 15m range, wait for retail stop runs (liquidity
  sweeps) on the 1m chart after the NY open, then enter dynamically on a
  displacement candle in the HTF bias direction.  Direction is LONG/SHORT and
  entry is at the market price.

Signal kwargs (in addition to the standard set):
  daily_bars    : list of daily OHLCV dicts (HTF bias / ESTABLISHED_MOVEMENT)
  four_h_bars   : list of 4h OHLCV dicts (POI / FVG respect check)
  fifteen_m_bars: list of recent 15m OHLCV dicts (range framing)
  one_m_bars    : list of recent 1m OHLCV dicts (sweep + movement candle)
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

log = logging.getLogger("fifteen_min_range_scalp")

EST = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

SOURCE = "15M_RANGE_SCALP"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
store = StateStore("fifteen_min_range_scalp", _PROJECT_ROOT)


def _make_state():
    return {
        "today": None,
        "htf_bias": None,            # "LONG" / "SHORT"
        "daily_bias_confirmed": False,
        "fifteen_min_range": {"high": None, "low": None, "timestamp": None, "is_valid": False},
        "liquidity_sweep_event": {"direction": None, "swept_level": None, "sweep_timestamp": None, "is_processed": False},
        "movement_candle_detected": False,
        "movement_candle_details": {},
        "entry_count": {"LONG": 0, "SHORT": 0},
        "cooldown": {"LONG": 0, "SHORT": 0},
        "last_entry_time": None,
    }


# ----- phase logic ----------------------------------------------------------

def _frame_htf_bias(daily_bars, four_h_bars):
    """Step 1: Daily + 4H directional bias. Returns ("LONG"/"SHORT", respected)."""
    confirmed, direction = dt.is_established_movement(daily_bars, 2)
    if not confirmed:
        return None, False
    bias = "LONG" if direction == "UP" else "SHORT"
    fvgs = dt.detect_fvg(four_h_bars[-30:], "4h")
    respected = True
    if fvgs:
        last = four_h_bars[-1]
        for f in fvgs[-3:]:
            if direction == "UP" and last["close"] < f["low"]:
                respected = False
            if direction == "DOWN" and last["close"] > f["high"]:
                respected = False
    return bias, respected


def _frame_15m_range(fifteen_m_bars, lookback=20):
    """Step 2: most extreme recent 15m high/low; valid if price near center."""
    recent = fifteen_m_bars[-lookback:] if len(fifteen_m_bars) >= lookback else fifteen_m_bars
    if len(recent) < 2:
        return {"high": None, "low": None, "timestamp": None, "is_valid": False}
    hi = max(c["high"] for c in recent)
    lo = min(c["low"] for c in recent)
    mid = (hi + lo) / 2.0
    rng = hi - lo
    last_close = fifteen_m_bars[-1]["close"]
    near_center = rng > 0 and abs(last_close - mid) <= 0.25 * rng
    return {"high": hi, "low": lo, "timestamp": recent[-1]["timestamp"], "is_valid": near_center}


def _detect_liquidity_sweep(bias, one_m_bars, after_et):
    """Step 3: post-8:30 ET sweep of an internal 1m swing.

    LONG bias -> sweep BELOW a prior 1m swing low (sell-side).
    SHORT bias -> sweep ABOVE a prior 1m swing high (buy-side).
    """
    if len(one_m_bars) < 5:
        return None, None, None
    for i in range(len(one_m_bars) - 1, 3, -1):
        c = one_m_bars[i]
        c_et = c["timestamp"].astimezone(EST) if hasattr(c["timestamp"], "astimezone") else c["timestamp"]
        if c_et < after_et:
            continue
        prev = one_m_bars[i - 3:i]
        if bias == "LONG":
            swing_low = min(p["low"] for p in prev)
            if c["low"] < swing_low:
                return "LONG", swing_low, c["timestamp"]
        else:
            swing_high = max(p["high"] for p in prev)
            if c["high"] > swing_high:
                return "SHORT", swing_high, c["timestamp"]
    return None, None, None


def _detect_movement_candle(bias, one_m_bars, after_ts):
    """Step 4: displacement candle >=70% body, <30% wicks, closes in bias dir."""
    if len(one_m_bars) < 2:
        return None
    window = [c for c in one_m_bars if c["timestamp"] >= after_ts]
    if not window:
        return None
    atr = cu.calculate_atr(one_m_bars[-25:], 20) or 0.0
    cand = window[-1]
    if bias == "LONG" and cand["close"] <= cand["open"]:
        return None
    if bias == "SHORT" and cand["close"] >= cand["open"]:
        return None
    if not cu.is_strong_body_candle(cand, cu.STRONG_BODY_RATIO):
        return None
    if atr > 0 and cu.candle_body_size(cand) < cu.MOVEMENT_CANDLE_ATR_MULT * atr:
        return None
    return cand


# ----- signal ---------------------------------------------------------------

def fifteen_min_range_scalp(
    spot_price=None,
    asset="NQ",
    max_reentries=3,
    daily_bars=None,
    four_h_bars=None,
    fifteen_m_bars=None,
    one_m_bars=None,
    **kwargs,
):
    ok, reason = validate_signal_inputs(spot_price, asset)
    if not ok:
        return no_signal(reason, SOURCE)

    now = tu.get_et_now()
    today = now.date()
    after_open = tu.ny_open_et()
    if now >= tu.hard_session_exit_et():
        return no_signal("hard_session_exit", SOURCE)
    if now < after_open:
        return no_signal("pre_830_et", SOURCE)
    if not (daily_bars and four_h_bars and fifteen_m_bars and one_m_bars):
        return no_signal("missing_bars", SOURCE)

    key = store.make_key(asset, today, max_reentries)
    state = store.load_or_new(key, _make_state)
    state["today"] = today
    store.prune(asset.upper(), today)
    store.tick_cooldowns(state)

    # Step 1: HTF bias.
    bias, bias_ok = _frame_htf_bias(daily_bars, four_h_bars)
    state["htf_bias"] = bias
    state["daily_bias_confirmed"] = bias_ok
    if bias is None or not bias_ok:
        return no_signal("no_htf_bias", SOURCE)

    # Step 2: 15m range.
    rng = _frame_15m_range(fifteen_m_bars)
    state["fifteen_min_range"] = rng
    if not rng["is_valid"]:
        return no_signal("range_not_centered", SOURCE)

    # Step 3: 1m liquidity sweep (post 8:30 ET).
    sweep_dir, swept_level, sweep_ts = _detect_liquidity_sweep(bias, one_m_bars, after_open)
    if sweep_dir is None:
        return no_signal("no_liquidity_sweep", SOURCE)
    state["liquidity_sweep_event"] = {
        "direction": sweep_dir, "swept_level": swept_level,
        "sweep_timestamp": sweep_ts, "is_processed": True,
    }

    # Step 4: Movement Candle after the sweep.
    mc = _detect_movement_candle(bias, one_m_bars, sweep_ts)
    if mc is None:
        return no_signal("no_movement_candle", SOURCE)
    state["movement_candle_detected"] = True
    state["movement_candle_details"] = {
        "timestamp": mc["timestamp"], "open": mc["open"], "high": mc["high"],
        "low": mc["low"], "close": mc["close"],
        "body_size": cu.candle_body_size(mc), "wick_ratio": cu.candle_wick_size(mc),
    }

    direction = bias  # LONG / SHORT
    n = state["entry_count"][direction]
    if not (n == 0 or 0 < n <= max_reentries):
        return no_signal("max_reentries", SOURCE)
    if state["cooldown"][direction] > 0:
        return no_signal("cooldown", SOURCE)

    # Entry at MC close per blueprint ("Entry: at MC close").  Futures market
    # entry: the MC close is the market price the displacement printed.
    entry_price = mc["close"]
    if entry_price is None or entry_price <= 0:
        return no_signal("no_price", SOURCE)

    if direction == "LONG":
        sl, tp = swept_level, rng["high"]
    else:
        sl, tp = swept_level, rng["low"]

    state["entry_count"][direction] += 1
    state["cooldown"][direction] = MIN_COOLDOWN_TICKS
    state["last_entry_time"] = mc["timestamp"]
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
        "htf_bias": bias,
        "swept_level": float(swept_level),
        "range_high": float(rng["high"]),
        "range_low": float(rng["low"]),
        "reason": (
            f"{'RE-ENTRY' if n > 0 else 'FIRST'} dir={direction} bias={bias} "
            f"asset={asset.upper()} sweep={swept_level:.2f} mc_close={mc['close']:.2f} "
            f"sl={sl:.2f} tp={tp:.2f} price={entry_price:.3f}"
        ),
    }
