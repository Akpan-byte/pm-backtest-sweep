# CHANGE_SUMMARY
# 2026-08-14  kilo
#   - Created signals/post_8am_bpr_magnet.py (Blueprint 4), refactored into the
#     compartmentalized package. 08:00 ET lock, BPR/DIRTY_BPR detection aligned
#     to 15m trend, orderflow FVG confirmation, hardcoded 2-pip/5-pip risk,
#     halved on counter-trend.
# WHY: Compartmentalization; see docs/post_8am_bpr_magnet.md.

"""Blueprint 4: Post-8AM BPR Magnet (Orderflow Micro-Scalp).

Documentation / master blueprint / changelog: docs/post_8am_bpr_magnet.md

Core logic:
  Imbalances formed aggressively at the NY open act as algorithmic vacuums.
  Wait for the structural gap to finalize, then scalp the rebalance.

Signal kwargs:
  one_m_bars    : list of 1m OHLCV dicts (BPR detection + internal sweeps)
  five_m_bars   : list of 5m OHLCV dicts (BPR detection)
  fifteen_m_bars: list of 15m OHLCV dicts (structural trend for risk tiering)
  pip_value     : price units per pip for the asset (default 1.0)
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

log = logging.getLogger("post_8am_bpr_magnet")

EST = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

SOURCE = "POST_8AM_BPR_MAGNET"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
store = StateStore("post_8am_bpr_magnet", _PROJECT_ROOT)

TP_PIPS = 2.0
SL_PIPS = 5.0
STD_RISK_PCT = 0.01
COUNTER_RISK_PCT = 0.005


def _make_state():
    return {
        "today": None,
        "is_after_filter_time": False,
        "bpr_formations": [],
        "orderflow_state": {"trend_direction": None, "confirmed_by_fvg": False},
        "fifteen_min_trend": {"direction": None, "timestamp": None},
        "entry_count": {"YES": 0, "NO": 0},
        "cooldown": {"YES": 0, "NO": 0},
    }


def _fifteen_min_trend(fifteen_m_bars):
    """Infer structural trend from 15m HH/HL vs LH/LL over recent bars."""
    if len(fifteen_m_bars) < 6:
        return None
    recent = fifteen_m_bars[-6:]
    highs = [c["high"] for c in recent]
    lows = [c["low"] for c in recent]
    if highs[-1] > highs[0] and lows[-1] > lows[0]:
        return "BULLISH"
    if highs[-1] < highs[0] and lows[-1] < lows[0]:
        return "BEARISH"
    return None


def post_8am_bpr_magnet(
    spot_price=None,
    asset="BTC",
    rem_sec=0,
    yp=None,
    np_val=None,
    yes_ask=None,
    no_ask=None,
    tf_hint="any",
    market_id=None,
    max_reentries=3,
    max_entry_price=0.85,
    time_gate_seconds=30,
    one_m_bars=None,
    five_m_bars=None,
    fifteen_m_bars=None,
    pip_value=1.0,
    **kwargs,
):
    ok, reason = validate_signal_inputs(spot_price, asset, rem_sec, yp, np_val, yes_ask, no_ask)
    if not ok:
        return no_signal(reason, SOURCE)

    now = tu.get_et_now()
    today = now.date()
    if rem_sec < time_gate_seconds:
        return no_signal("time_gate", SOURCE)
    if not tu.is_after_ny_open_filter_time(now):
        return no_signal("pre_0800_et", SOURCE)
    if not (one_m_bars and five_m_bars):
        return no_signal("missing_bars", SOURCE)

    key = store.make_key(asset, today, max_reentries)
    state = store.load_or_new(key, _make_state)
    state["today"] = today
    store.prune(asset.upper(), today)
    store.tick_cooldowns(state)
    state["is_after_filter_time"] = True

    tf_trend = _fifteen_min_trend(fifteen_m_bars or [])
    state["fifteen_min_trend"] = {"direction": tf_trend, "timestamp": now}
    if tf_trend is None:
        return no_signal("no_15m_trend", SOURCE)

    # Detection array: BPR / DIRTY_BPR on 1m and 5m, aligned to trend.
    trend_dir = "UP" if tf_trend == "BULLISH" else "DOWN"
    raw_bprs = (dt.detect_bpr(one_m_bars[-30:], "1m")
                + dt.detect_dirty_bpr(one_m_bars[-30:], "1m")
                + dt.detect_bpr(five_m_bars[-30:], "5m"))
    aligned = [f for f in raw_bprs if f["direction"] == trend_dir]
    state["bpr_formations"] = aligned[-5:]
    if not aligned:
        return no_signal("no_bpr_in_trend", SOURCE)

    # Orderflow confirmation: an FVG must have been *traded into and respected*
    # earlier in the session — i.e. price wicked into the FVG but closed back
    # outside it, proving the algorithm defended the level.  Plain existence of
    # an FVG is not enough.
    all_fvgs = dt.detect_fvg(one_m_bars[-30:], "1m") + dt.detect_fvg(five_m_bars[-30:], "5m")
    orderflow_bull = False
    orderflow_bear = False
    for f in all_fvgs:
        # Check subsequent bars for a wick-into-FVG-and-close-back-out pattern.
        for c in one_m_bars:
            if c["timestamp"] <= f["timestamp"]:
                continue
            if f["direction"] == "UP":
                # Bullish FVG: price dipped into it (wick below f["high"]) and
                # closed back above f["low"] — defended as support.
                if c["low"] <= f["high"] and c["close"] > f["low"]:
                    orderflow_bull = True
                    break
            elif f["direction"] == "DOWN":
                # Bearish FVG: price rallied into it (wick above f["low"]) and
                # closed back below f["high"] — defended as resistance.
                if c["high"] >= f["low"] and c["close"] < f["high"]:
                    orderflow_bear = True
                    break
        if orderflow_bull or orderflow_bear:
            break
    if orderflow_bull and not orderflow_bear:
        orderflow_dir = "BULLISH"
    elif orderflow_bear and not orderflow_bull:
        orderflow_dir = "BEARISH"
    else:
        orderflow_dir = None
    confirmed = orderflow_dir == ("BULLISH" if tf_trend == "BULLISH" else "BEARISH")
    state["orderflow_state"] = {"trend_direction": orderflow_dir, "confirmed_by_fvg": confirmed}
    if not confirmed:
        return no_signal("orderflow_unconfirmed", SOURCE)

    direction = "YES" if tf_trend == "BULLISH" else "NO"
    n = state["entry_count"][direction]
    if not (n == 0 or 0 < n <= max_reentries):
        return no_signal("max_reentries", SOURCE)
    if state["cooldown"][direction] > 0:
        return no_signal("cooldown", SOURCE)

    entry_price = yes_ask if direction == "YES" else no_ask
    if entry_price is None:
        entry_price = yp if direction == "YES" else np_val
    if entry_price is None or entry_price <= 0:
        return no_signal("no_price", SOURCE)
    if entry_price > max_entry_price:
        return no_signal("price_cap", SOURCE)

    if direction == "YES":
        tp, sl = entry_price + TP_PIPS * pip_value, entry_price - SL_PIPS * pip_value
    else:
        tp, sl = entry_price - TP_PIPS * pip_value, entry_price + SL_PIPS * pip_value

    # Counter-trend check: was the most recent *raw* BPR (before trend
    # filtering) in the opposite direction of the 15m trend?  If so, this is
    # an "A- setup" and risk is halved.
    counter = False
    if raw_bprs:
        last_raw_dir = raw_bprs[-1]["direction"]
        counter = last_raw_dir != trend_dir
    risk_pct = COUNTER_RISK_PCT if counter else STD_RISK_PCT

    state["entry_count"][direction] += 1
    state["cooldown"][direction] = MIN_COOLDOWN_TICKS
    store.save(key, state)

    return {
        "triggered": True,
        "direction": direction,
        "confidence": risk_pct,
        "entry_price": float(entry_price),
        "signal_price": float(entry_price),
        "source": SOURCE,
        "sl": float(sl),
        "tp": float(tp),
        "risk_pct": risk_pct,
        "tp_pips": TP_PIPS,
        "sl_pips": SL_PIPS,
        "counter_trend": counter,
        "reason": (
            f"{'RE-ENTRY' if n > 0 else 'FIRST'} dir={direction} trend={tf_trend} "
            f"bpr={aligned[-1]['type']} tp={TP_PIPS}p sl={SL_PIPS}p risk={risk_pct:.1%} "
            f"price={entry_price:.3f}"
        ),
    }
