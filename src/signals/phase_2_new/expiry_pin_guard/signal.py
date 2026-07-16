# CHANGE_SUMMARY
# 2026-07-16  (subagent)
#   - Implemented expiry_pin_guard_signal per STRATEGY_SPECS.md.
#   - Negative filter for BTC 5m UP/DOWN Polymarket markets inside the last 90s.
#   - Blocks entry when spot is within 0.0003 (relative) of the strike (pin-risk guard).
#   - Standalone signal never triggers; intended as a composite gate.
# 2026-07-16  kilo
#   - Switched entry_price to the ask side for taker fills:
#     YES direction uses yes_ask (fallback to yp), NO direction uses no_ask
#     (fallback to np_val) when ask is missing or invalid.
#   - The [0.05, 0.85] entry-price guard is unchanged.
#
# WHY: Provide a reusable guard module that prevents entries near expiry when
#      spot is hovering at the strike and pin risk / chop is highest.

"""Expiry pin-risk guard for Polymarket BTC 5m up/down markets."""

# Module-level state required by the interface.
_STATE = {}


def _neutral(spot_price, reason):
    return {
        "triggered": False,
        "direction": None,
        "confidence": 0.0,
        "signal_price": spot_price,
        "entry_price": 0.0,
        "source": "EXPIRY_PIN_GUARD",
        "reason": reason,
    }


def expiry_pin_guard_signal(**kwargs):
    """Return a neutral/negative signal unless expiry pin-risk conditions block entry.

    This signal never triggers on its own; it is designed to be used as a gate
    inside a composite signal.  It returns ``triggered=False`` in all branches,
    with a ``reason`` explaining whether the guard is blocking entry.
    """
    spot_price = float(kwargs.get("spot_price", 0.0))
    strike = float(kwargs.get("strike", 0.0))
    rem_sec = float(kwargs.get("rem_sec", 0.0))
    elapsed_sec = float(kwargs.get("elapsed_sec", 0.0))

    # Global time guards from INTERFACE.md.
    if rem_sec <= 5 or elapsed_sec <= 5:
        return _neutral(spot_price, "time guard blocks first/last 5 seconds")

    # Strategy only runs inside the final 90 seconds.
    if rem_sec > 90:
        return _neutral(spot_price, "outside expiry window (rem_sec > 90)")

    # Avoid divide-by-zero on a zero strike (should not happen for BTC, but be safe).
    if strike == 0:
        return _neutral(spot_price, "invalid zero strike")

    dist = abs(spot_price - strike) / strike

    # Pin-risk guard: if spot is too close to the strike, block entry.
    if dist < 0.0003:
        return _neutral(spot_price, f"pin-risk guard active (dist={dist:.6f} < 0.0003)")

    # Guard would allow entry, but as a standalone signal we still do not trigger.
    return _neutral(spot_price, f"guard allows entry (dist={dist:.6f}) but standalone never triggers")
