# CHANGE_SUMMARY
# 2026-08-14  coder
#   - REWRITTEN as the FUTURES/normal-trading signal contract.
#   - Direction vocabulary is LONG/SHORT (not YES/NO); entry is the market
#     price (spot_price), not a binary yes_ask/no_ask.
#   - Dropped the Polymarket-only params (yp, np_val, yes_ask, no_ask,
#     rem_sec, tf_hint, market_id, max_entry_price cap, time_gate_seconds).
#   - The original Polymarket-flavored contract is preserved verbatim in
#     strategies/signals_polymarket/ (see docs/FUTURES_VS_POLYMARKET.md).
# WHY: The four StarTrading blueprints are futures intraday strategies; the
#      binary-contract signature was a legacy adaptation, now split out.
#
# 2026-08-14  kilo
#   - Created signals/common.py with the shared signal contract: the standard
#     return dict, no_signal() builder, and validate_signal_inputs(). Moves the
#     boilerplate out of every strategy module into one navigable place.
# WHY: Compartmentalization; see docs/BLUEPRINTS.md.

"""Shared FUTURES signal contract for the four StarTrading strategies.

Every signal function returns a dict with at least:
  triggered, direction ("LONG"/"SHORT"/None), confidence, entry_price,
  signal_price, source, reason.  Optional fields: sl, tp, risk_pct, etc.

Entry is always at the current market price (spot_price).  SL/TP are expressed
in absolute index/point price terms; pip_value maps "pips" to points for the
strategies that hardcode pip-based targets (NQ/ES/YM: pip_value=1 point).
"""

import logging

log = logging.getLogger("strategies.signals.common")

# Standard per-trade cooldown in ticks before a re-entry is allowed.
MIN_COOLDOWN_TICKS = 3
# Size multiplier applied on successive re-entries (1st, 2nd, 3rd, 4th...).
REENTRY_SIZE_SCALE = [1.0, 0.75, 0.50, 0.33]


def no_signal(reason: str, source: str) -> dict:
    return {
        "triggered": False,
        "direction": None,
        "confidence": 0.0,
        "entry_price": 0.0,
        "signal_price": 0.0,
        "source": source,
        "reason": reason,
    }


def validate_signal_inputs(spot_price, asset, **kwargs):
    """Validate the FUTURES signal inputs.

    Only the market price and asset identity matter; the Polymarket binary
    fields (yp, np_val, yes_ask, no_ask, rem_sec, ...) are gone.  Session gating
    is handled by the strategies via time_utils, not by a binary rem_sec.
    """
    if spot_price is None or spot_price <= 0:
        return False, "no spot"
    if not asset:
        return False, "no asset"
    return True, ""


def reentry_scale(index: int) -> float:
    return REENTRY_SIZE_SCALE[min(index, len(REENTRY_SIZE_SCALE) - 1)]
