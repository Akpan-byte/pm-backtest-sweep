# CHANGE_SUMMARY
# 2026-08-14  kilo
#   - Created signals/mos_session_daily_draw.py (Blueprint 3), refactored into
#     the compartmentalized package. Entry gated to 00:00 UTC; aborts on doji
#     prior day, missing ESTABLISHED_MOVEMENT, or premature PDH/PDL sweep; TP
#     hard-capped at 10 pips; risk raised to 3-4%.
# WHY: Compartmentalization; see docs/mos_session_daily_draw.md.

"""Blueprint 3: 00:00 UTC MOS Session Daily Draw.

Documentation / master blueprint / changelog: docs/mos_session_daily_draw.md

Core logic:
  At the daily rollover the algorithm draws price toward the Previous Day
  High/Low.  With a wide protective swing stop and a capped 10-pip target we
  get a high win rate; risk sizing is raised to 3-4% to exploit it.

Signal kwargs:
  daily_bars : list of daily OHLCV dicts (prior day doji / movement / PDH-PDL)
  four_h_bars: list of 4h OHLCV dicts (protective swing search)
  one_h_bars : list of 1h OHLCV dicts (protective swing search)
  pip_value  : price units per pip for the asset (default 1.0)
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

log = logging.getLogger("mos_session_daily_draw")

EST = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

SOURCE = "MOS_SESSION_DRAW"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
store = StateStore("mos_session_daily_draw", _PROJECT_ROOT)

TP_PIP_CAP = 10.0           # hard TP cap in pips
HIGH_RISK_PCT = 0.04        # 3-4% equity per trigger


def _make_state():
    return {
        "today": None,
        "previous_day_candle": {},
        "established_movement": {"direction": None, "consecutive_candles": 0, "confirmed": False},
        "pdh_pdl": {"high": None, "low": None},
        "pdh_pdl_swept_premarket": False,
        "protective_swing": {},
        "entry_count": {"YES": 0, "NO": 0},
        "cooldown": {"YES": 0, "NO": 0},
        "has_entered_session": False,
    }


def mos_session_daily_draw(
    spot_price=None,
    asset="BTC",
    rem_sec=0,
    yp=None,
    np_val=None,
    yes_ask=None,
    no_ask=None,
    tf_hint="any",
    market_id=None,
    max_reentries=1,
    max_entry_price=0.85,
    time_gate_seconds=30,
    daily_bars=None,
    four_h_bars=None,
    one_h_bars=None,
    pip_value=1.0,
    **kwargs,
):
    ok, reason = validate_signal_inputs(spot_price, asset, rem_sec, yp, np_val, yes_ask, no_ask)
    if not ok:
        return no_signal(reason, SOURCE)

    now = tu.get_utc_now()
    today = now.date()
    if rem_sec < time_gate_seconds:
        return no_signal("time_gate", SOURCE)
    if not (daily_bars and (four_h_bars or one_h_bars)):
        return no_signal("missing_bars", SOURCE)

    # Entry window: only fire within the first seconds of 00:00 UTC.
    if not tu.is_mos_session_time(now):
        return no_signal("not_mos_session", SOURCE)

    key = store.make_key(asset, today, max_reentries)
    state = store.load_or_new(key, _make_state)
    state["today"] = today
    store.prune(asset.upper(), today)
    store.tick_cooldowns(state)

    # Validation 1: prior day must not be a doji (50_50_CANDLE).
    # Treat the final bar in daily_bars as the just-completed previous day.
    prev = daily_bars[-1]
    state["previous_day_candle"] = {
        "open": prev["open"], "high": prev["high"],
        "low": prev["low"], "close": prev["close"],
        "is_doji": cu.is_doji_candle(prev),
    }
    if cu.is_doji_candle(prev):
        return no_signal("50_50_candle_abort", SOURCE)

    # Validation 2: ESTABLISHED_MOVEMENT on daily.
    confirmed, direction = dt.is_established_movement(daily_bars, 2)
    state["established_movement"] = {
        "direction": direction, "confirmed": confirmed,
        "consecutive_candles": min(2, len(daily_bars)),
    }
    if not confirmed or direction is None:
        return no_signal("no_established_movement", SOURCE)

    # Validation 3: PDH/PDL not already swept before session open.
    pdh, pdl = prev["high"], prev["low"]
    state["pdh_pdl"] = {"high": pdh, "low": pdl}
    if direction == "UP" and spot_price >= pdh:
        return no_signal("pdh_swept_premarket", SOURCE)
    if direction == "DOWN" and spot_price <= pdl:
        return no_signal("pdl_swept_premarket", SOURCE)

    # Protective swing behind which we hide the stop (opposite of trend).
    fvgs_4h = dt.detect_fvg(four_h_bars[-30:], "4h") if four_h_bars else []
    fvgs_1h = dt.detect_fvg(one_h_bars[-30:], "1h") if one_h_bars else []
    swing = dt.find_protective_swing(four_h_bars[-30:], direction, fvgs_4h) if four_h_bars else None
    if swing is None and one_h_bars:
        swing = dt.find_protective_swing(one_h_bars[-30:], direction, fvgs_1h)
    state["protective_swing"] = swing or {}

    direction_signal = "YES" if direction == "UP" else "NO"
    n = state["entry_count"][direction_signal]
    if n >= max_reentries:
        return no_signal("max_reentries", SOURCE)
    if state["cooldown"][direction_signal] > 0:
        return no_signal("cooldown", SOURCE)
    if state["has_entered_session"]:
        return no_signal("already_entered_session", SOURCE)

    entry_price = yes_ask if direction_signal == "YES" else no_ask
    if entry_price is None:
        entry_price = yp if direction_signal == "YES" else np_val
    if entry_price is None or entry_price <= 0:
        return no_signal("no_price", SOURCE)
    if entry_price > max_entry_price:
        return no_signal("price_cap", SOURCE)

    if direction == "UP":
        tp_raw, sl = pdh, (swing["price_level"] if swing else pdl - 50 * pip_value)
    else:
        tp_raw, sl = pdl, (swing["price_level"] if swing else pdh + 50 * pip_value)
    tp_dist_pips = abs(tp_raw - entry_price) / max(pip_value, 1e-9)
    tp = tp_raw if tp_dist_pips <= TP_PIP_CAP else (
        entry_price + TP_PIP_CAP * pip_value if direction == "UP"
        else entry_price - TP_PIP_CAP * pip_value
    )

    state["entry_count"][direction_signal] += 1
    state["cooldown"][direction_signal] = MIN_COOLDOWN_TICKS
    state["has_entered_session"] = True
    store.save(key, state)

    return {
        "triggered": True,
        "direction": direction_signal,
        "confidence": HIGH_RISK_PCT,
        "entry_price": float(entry_price),
        "signal_price": float(entry_price),
        "source": SOURCE,
        "sl": float(sl),
        "tp": float(tp),
        "risk_pct": HIGH_RISK_PCT,
        "pdh": float(pdh),
        "pdl": float(pdl),
        "reason": (
            f"MOS dir={direction_signal} trend={direction} asset={asset.upper()} "
            f"pdh={pdh:.2f} pdl={pdl:.2f} tp_cap={TP_PIP_CAP}pips sl={sl:.2f} "
            f"price={entry_price:.3f} risk={HIGH_RISK_PCT:.0%}"
        ),
    }
