# CHANGE_SUMMARY
# 2026-08-18  kilo
#   - Created strategies/signals/vwap_sd_reversion.py: VWAP mean-reversion with
#     2.0 standard-deviation bands. Uses an in-memory state dict keyed by
#     (asset, date) to incrementally track session VWAP and volume-weighted
#     standard deviation, avoiding both per-bar disk I/O and full-window scans.
#     * VWAP and 2.0 SD bands are computed from the 09:30 ET open onward.
#     * Entries are evaluated from 10:00 to 15:30 ET using the previous closed bar.
#     * Stop loss is placed 15 points beyond the extreme wick of the rejection bar.
#     * Take profit is the current session VWAP.
#     * Flatten is enforced at 15:50 ET by the harness (registered hard-exit).
# WHY: Implement the VWAP SD band rejection strategy to the exact user rules.

"""VWAP mean-reversion / 2.0 SD band rejection (FUTURES edition)."""

import logging
from datetime import time
from zoneinfo import ZoneInfo

from ..core import time_utils as tu
from .common import no_signal, validate_signal_inputs, reentry_scale

log = logging.getLogger("vwap_sd_reversion")

EST = ZoneInfo("America/New_York")
SOURCE = "VWAP_SD_REVERSION"

T0930 = time(9, 30)
T1000 = time(10, 0)
T1530 = time(15, 30)

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
        "cum_vol": 0.0,
        "vwap_mean": 0.0,
        "vwap_m2": 0.0,
        "entered_today": False,
    }


def _update_vwap(state, bar):
    """Online weighted-mean / weighted-variance update for VWAP bands."""
    tp = (bar["high"] + bar["low"] + bar["close"]) / 3.0
    w = float(bar["volume"] or 0)
    if w <= 0:
        return
    old_w = state["cum_vol"]
    new_w = old_w + w
    delta = tp - state["vwap_mean"]
    delta2 = tp - (state["vwap_mean"] + delta * w / new_w)
    state["vwap_mean"] += delta * w / new_w
    state["vwap_m2"] += w * delta * delta2
    state["cum_vol"] = new_w


def vwap_sd_reversion(
    spot_price=None,
    asset="NQ",
    max_reentries=0,
    one_m_bars=None,
    sd_multiplier=2.0,
    fixed_stop_pts=15.0,
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

    # Update VWAP from the RTH open onward (bands need history before 10:00).
    if curr_time >= T0930:
        _update_vwap(state, curr_bar)

    # Active window: 10:00 - 15:30 ET.
    if curr_time < T1000 or curr_time > T1530:
        return no_signal("outside_active_window", SOURCE)

    if state["cum_vol"] <= 0:
        return no_signal("vwap_not_ready", SOURCE)

    vwap = state["vwap_mean"]
    variance = state["vwap_m2"] / state["cum_vol"]
    std = variance ** 0.5 if variance > 0 else 0.0
    upper = vwap + sd_multiplier * std
    lower = vwap - sd_multiplier * std

    # Only one entry per day.
    if state["entered_today"]:
        return no_signal("already_entered_today", SOURCE)

    prev_bar = one_m_bars[-2]
    if _et_date(prev_bar) != curr_date:
        return no_signal("prev_bar_stale", SOURCE)

    entry_price = float(spot_price)
    direction = None
    sl = None
    tp = vwap

    # Long: previous bar touched/breached the lower band and closed back inside.
    if prev_bar["low"] <= lower and prev_bar["close"] > lower:
        direction = "LONG"
        sl = prev_bar["low"] - fixed_stop_pts
    # Short: previous bar touched/breached the upper band and closed back inside.
    elif prev_bar["high"] >= upper and prev_bar["close"] < upper:
        direction = "SHORT"
        sl = prev_bar["high"] + fixed_stop_pts

    if direction is None:
        return no_signal("no_band_rejection", SOURCE)

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
            f"VWAP_SD_REVERSION dir={direction} asset={asset.upper()} "
            f"vwap={vwap:.2f} sd={std:.2f} bands=[{lower:.2f},{upper:.2f}] "
            f"prev_low={prev_bar['low']:.2f} prev_high={prev_bar['high']:.2f} "
            f"sl={sl:.2f} tp={tp:.2f}"
        ),
    }
