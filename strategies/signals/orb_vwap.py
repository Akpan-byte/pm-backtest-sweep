# CHANGE_SUMMARY
# 2026-08-18  kilo
#   - Created strategies/signals/orb_vwap.py: 15-minute opening range breakout
#     with session VWAP filter. Uses an in-memory state dict keyed by
#     (asset, date) to incrementally track the 15-min range and session VWAP,
#     avoiding both per-bar disk I/O and full-window scans.
#     * Range is recorded from 09:30-09:44 ET (first 15 minutes).
#     * Entries are evaluated from 09:45 ET using the *previous* closed bar.
#     * VWAP is computed incrementally from the 09:30 open onward.
#     * Stop loss is fixed at the midpoint of the 15-minute range.
#     * Take profit is 2.0x the entry-to-stop distance.
#     * Flatten is enforced at 15:55 ET by the harness (registered hard-exit).
# WHY: Implement the ORB+VWAP strategy to the exact rules supplied by the user.

"""15-Minute Opening Range Breakout with VWAP filter (FUTURES edition)."""

import logging
from datetime import time
from zoneinfo import ZoneInfo

from ..core import time_utils as tu
from .common import no_signal, validate_signal_inputs, reentry_scale

log = logging.getLogger("orb_vwap")

EST = ZoneInfo("America/New_York")
SOURCE = "ORB_VWAP"

T0930 = time(9, 30)
T0944 = time(9, 44)
T0945 = time(9, 45)
T1545 = time(15, 45)

# In-memory per-(asset, date) state. No disk I/O.
_STATE: dict = {}
_LAST_DATE = None


def _maybe_clear_on_rewind(curr_date):
    global _LAST_DATE
    if _LAST_DATE is not None and curr_date < _LAST_DATE:
        _STATE.clear()
    _LAST_DATE = curr_date


def _et_time(bar):
    ts = bar["timestamp"]
    return ts.astimezone(EST).time() if hasattr(ts, "astimezone") else ts.time()


def _et_date(bar):
    ts = bar["timestamp"]
    return ts.astimezone(EST).date() if hasattr(ts, "astimezone") else ts.date()


def _make_state():
    return {
        "range_high": None,
        "range_low": None,
        "range_mid": None,
        "cum_tp_vol": 0.0,
        "cum_vol": 0.0,
        "entered_today": False,
    }


def orb_vwap(
    spot_price=None,
    asset="NQ",
    max_reentries=0,
    one_m_bars=None,
    **kwargs,
):
    ok, reason = validate_signal_inputs(spot_price, asset)
    if not ok:
        return no_signal(reason, SOURCE)

    now = tu.get_et_now()
    today = now.date()
    t = now.time()

    if not one_m_bars or len(one_m_bars) < 2:
        return no_signal("missing_bars", SOURCE)

    curr_bar = one_m_bars[-1]
    curr_date = _et_date(curr_bar)
    curr_time = _et_time(curr_bar)

    _maybe_clear_on_rewind(curr_date)

    key = (asset.upper(), curr_date)
    if key not in _STATE:
        _STATE[key] = _make_state()
    state = _STATE[key]

    # Update incremental VWAP from the RTH open onward.
    if curr_time >= T0930:
        tp = (curr_bar["high"] + curr_bar["low"] + curr_bar["close"]) / 3.0
        vol = float(curr_bar["volume"] or 0)
        state["cum_tp_vol"] += tp * vol
        state["cum_vol"] += vol

    # Update the 15-minute range from 09:30-09:44 ET.
    if T0930 <= curr_time <= T0944:
        if state["range_high"] is None:
            state["range_high"] = curr_bar["high"]
            state["range_low"] = curr_bar["low"]
        else:
            state["range_high"] = max(state["range_high"], curr_bar["high"])
            state["range_low"] = min(state["range_low"], curr_bar["low"])
        state["range_mid"] = (state["range_high"] + state["range_low"]) / 2.0
        return no_signal("building_range", SOURCE)

    # No entries before the range is formed or after the active window.
    if curr_time < T0945 or curr_time > T1545:
        return no_signal("outside_active_window", SOURCE)

    # Need a completed range to trade.
    if state["range_high"] is None or state["cum_vol"] <= 0:
        return no_signal("range_not_ready", SOURCE)

    # Only one entry per day.
    if state["entered_today"]:
        return no_signal("already_entered_today", SOURCE)

    vwap = state["cum_tp_vol"] / state["cum_vol"]
    high_15m = state["range_high"]
    low_15m = state["range_low"]
    mid_15m = state["range_mid"]

    # Entry decision uses the previous closed bar (t-1) to avoid lookahead.
    prev_bar = one_m_bars[-2]
    if _et_date(prev_bar) != curr_date:
        return no_signal("prev_bar_stale", SOURCE)
    prev_close = prev_bar["close"]

    rr_ratio = 2.0
    entry_price = float(spot_price)
    sl = mid_15m

    if prev_close > high_15m and prev_close > vwap:
        direction = "LONG"
        risk = entry_price - sl
        tp = entry_price + risk * rr_ratio
    elif prev_close < low_15m and prev_close < vwap:
        direction = "SHORT"
        risk = sl - entry_price
        tp = entry_price - risk * rr_ratio
    else:
        return no_signal("no_breakout", SOURCE)

    state["entered_today"] = True

    return {
        "triggered": True,
        "direction": direction,
        "confidence": reentry_scale(0),
        "entry_price": entry_price,
        "signal_price": entry_price,
        "source": SOURCE,
        "sl": float(sl),
        "tp": float(tp),
        "reason": (
            f"ORB_VWAP dir={direction} asset={asset.upper()} "
            f"range=[{low_15m:.2f},{high_15m:.2f}] mid={mid_15m:.2f} "
            f"vwap={vwap:.2f} prev_close={prev_close:.2f} "
            f"sl={sl:.2f} tp={tp:.2f}"
        ),
    }
