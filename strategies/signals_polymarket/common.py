# CHANGE_SUMMARY
# 2026-08-14  kilo
#   - Created signals/common.py with the shared signal contract: the standard
#     return dict, no_signal() builder, and validate_signal_inputs(). Moves the
#     boilerplate out of every strategy module into one navigable place.
# WHY: Compartmentalization; see docs/BLUEPRINTS.md.

"""Shared signal contract for the four StarTrading strategies.

Every signal function returns a dict with at least:
  triggered, direction ("YES"/"NO"/None), confidence, entry_price,
  signal_price, source, reason.  Optional fields: sl, tp, risk_pct, etc.
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


def validate_signal_inputs(spot_price, asset, rem_sec, yp, np_val, yes_ask, no_ask):
    if spot_price is None or spot_price <= 0:
        return False, "no spot"
    if rem_sec is None or rem_sec < 0:
        return False, "invalid rem_sec"
    return True, ""


def reentry_scale(index: int) -> float:
    return REENTRY_SIZE_SCALE[min(index, len(REENTRY_SIZE_SCALE) - 1)]
