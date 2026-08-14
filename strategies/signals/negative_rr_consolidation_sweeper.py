# CHANGE_SUMMARY
# 2026-08-14  coder
#   - REWRITTEN as the FUTURES/normal-trading version of Blueprint 2.
#   - Direction is LONG/SHORT (no more YES/NO); entry is the market price
#     (spot_price), not a binary ask.
#   - Removed Polymarket-only params (yp, np_val, yes_ask, no_ask, rem_sec,
#     max_entry_price, time_gate_seconds). Phase filter (consolidation + NFP /
#     early-month abort) and the recovery loop are unchanged.
#   - EQH fade -> SHORT, EQL fade -> LONG. State keys use LONG/SHORT.
#   - The Polymarket-flavored original is archived verbatim in
#     strategies/signals_polymarket/negative_rr_consolidation_sweeper.py.
# WHY: Futures-native contract for the 12-instrument backtest; see
#      docs/FUTURES_VS_POLYMARKET.md.
#
# 2026-08-14  kilo
#   - Created signals/negative_rr_consolidation_sweeper.py (Blueprint 2),
#     refactored into the compartmentalized package. Phase filter, EQH/EQL
#     detection, wide-stop / micro-target risk, and the 2-trade recovery loop
#     are orchestrated here; primitives live in core/*.
# WHY: Compartmentalization; see docs/negative_rr_consolidation_sweeper.md.

"""Blueprint 2 (FUTURES): Negative RR Consolidation Sweeper.

Documentation / master blueprint / changelog: docs/negative_rr_consolidation_sweeper.md

Core logic:
  In choppy/consolidating markets, retail traders defend obvious Equal
  Highs/Lows.  Trade directly into that liquidity with a very wide stop and a
  micro target (0.2 RR) to mathematically favor a high win rate.

Signal kwargs:
  daily_bars  : list of daily OHLCV dicts (consolidation / movement check)
  swing_highs : list[float] of recent swing-high price levels
  swing_lows  : list[float] of recent swing-low price levels
  trade_history: list of recent closed trades (for the recovery loop)
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

log = logging.getLogger("negative_rr_consolidation_sweeper")

EST = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

SOURCE = "NEG_RR_CONSOLIDATION"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
store = StateStore("negative_rr_consolidation", _PROJECT_ROOT)

BASE_TP_RR = 0.2           # baseline target RR
RECOVERY_TP_RR = 0.5       # target RR during recovery
SL_MULTIPLE_OF_TP = 5.0   # exceptionally wide stop
RECOVERY_WIN_TARGET = 2     # consecutive wins to exit recovery


def _make_state():
    return {
        "today": None,
        "is_consolidation_phase": False,
        "eqh_eql_levels": {"highs": [], "lows": []},
        "recent_trades": [],
        "recovery_state": {"active": False, "wins_so_far": 0},
        "entry_count": {"LONG": 0, "SHORT": 0},
        "cooldown": {"LONG": 0, "SHORT": 0},
    }


def _is_nfp_bank_holiday_early_month(check_date):
    """Abort on NFP (first Friday), bank holidays, first 3 days of month."""
    if check_date.day <= 3:
        return True
    if check_date.day <= 7 and check_date.weekday() == 4:
        return True
    return False


def negative_rr_consolidation_sweeper(
    spot_price=None,
    asset="NQ",
    max_reentries=3,
    daily_bars=None,
    swing_highs=None,
    swing_lows=None,
    trade_history=None,
    **kwargs,
):
    ok, reason = validate_signal_inputs(spot_price, asset)
    if not ok:
        return no_signal(reason, SOURCE)

    now = tu.get_et_now()
    today = now.date()
    if not daily_bars:
        return no_signal("missing_daily_bars", SOURCE)

    # Phase filter: only in consolidation.
    if dt.is_established_movement(daily_bars, 2)[0]:
        return no_signal("established_movement_disabled", SOURCE)
    if _is_nfp_bank_holiday_early_month(today):
        return no_signal("nfp_or_early_month_abort", SOURCE)

    key = store.make_key(asset, today, max_reentries)
    state = store.load_or_new(key, _make_state)
    state["today"] = today
    store.prune(asset.upper(), today)
    store.tick_cooldowns(state)
    state["is_consolidation_phase"] = True

    # EQH / EQL detection.
    eq = dt.detect_eqh_eql(swing_highs or [], swing_lows or [], 0.001)
    state["eqh_eql_levels"] = eq
    if not eq["highs"] and not eq["lows"]:
        return no_signal("no_eqh_eql", SOURCE)

    candidates = [("SHORT", lvl) for lvl in eq["highs"]] + [("LONG", lvl) for lvl in eq["lows"]]
    if not candidates:
        return no_signal("no_eq_target", SOURCE)
    candidates.sort(key=lambda c: abs(c[1] - spot_price))
    direction, eq_level = candidates[0]

    # Recovery loop: after a loss, target higher RR for 2 CONSECUTIVE wins.
    # Each loss resets the consecutive-win counter; only back-to-back wins
    # count toward the RECOVERY_WIN_TARGET threshold.
    # Merge caller-provided trade_history into state so it persists across ticks.
    if trade_history:
        existing = {(t.get("id"), t.get("profit_loss")) for t in state["recent_trades"]}
        for t in trade_history:
            key_t = (t.get("id"), t.get("profit_loss"))
            if key_t not in existing:
                state["recent_trades"].append(t)
        state["recent_trades"] = state["recent_trades"][-20:]
    recent = state["recent_trades"][-10:]
    last_result = recent[-1].get("profit_loss", 0) if recent else 0
    if last_result < 0:
        # Loss — activate recovery (or re-activate if already active) and
        # reset the consecutive-win counter.
        state["recovery_state"]["active"] = True
        state["recovery_state"]["wins_so_far"] = 0
    elif last_result > 0 and state["recovery_state"]["active"]:
        # Win while in recovery — count consecutive wins.
        state["recovery_state"]["wins_so_far"] += 1
        if state["recovery_state"]["wins_so_far"] >= RECOVERY_WIN_TARGET:
            state["recovery_state"] = {"active": False, "wins_so_far": 0}
    tp_rr = RECOVERY_TP_RR if state["recovery_state"]["active"] else BASE_TP_RR

    n = state["entry_count"][direction]
    if not (n == 0 or 0 < n <= max_reentries):
        return no_signal("max_reentries", SOURCE)
    if state["cooldown"][direction] > 0:
        return no_signal("cooldown", SOURCE)

    entry_price = spot_price
    if entry_price is None or entry_price <= 0:
        return no_signal("no_price", SOURCE)

    dist = abs(eq_level - entry_price)
    tp = eq_level
    sl = entry_price - SL_MULTIPLE_OF_TP * dist if direction == "LONG" else entry_price + SL_MULTIPLE_OF_TP * dist

    state["entry_count"][direction] += 1
    state["cooldown"][direction] = MIN_COOLDOWN_TICKS
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
        "tp_rr": tp_rr,
        "recovery_active": state["recovery_state"]["active"],
        "eq_level": float(eq_level),
        "reason": (
            f"{'RE-ENTRY' if n > 0 else 'FIRST'} dir={direction} eq={eq_level:.2f} "
            f"tp_rr={tp_rr} sl={sl:.2f} tp={tp:.2f} price={entry_price:.3f}"
        ),
    }
