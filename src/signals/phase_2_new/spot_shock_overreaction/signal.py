# CHANGE_SUMMARY
# 2026-07-16  kilo
#   - Created spot_shock_overreaction signal module.
#   - Implemented 5-tick shock detection fading sharp BTC spot moves once
#     the last two ticks show deceleration, per STRATEGY_SPECS.md.
# 2026-07-16  kilo
#   - Switched entry_price to the ask side for taker fills:
#     YES direction uses yes_ask (fallback to yp), NO direction uses no_ask
#     (fallback to np_val) when ask is missing or invalid.
#   - The [0.05, 0.85] entry-price guard is unchanged.
#
# WHY: Adds a new Polymarket BTC 5m up/down mean-reversion signal.

"""Spot-shock overreaction signal for Polymarket BTC 5m up/down markets.

Fades a sharp spot move after it begins to exhaust.  Requires a 5-tick
return more than 2.5x the recent 20-tick realized volatility and a
latest-tick acceleration opposite to the 5-tick move.
"""

import statistics
from typing import Any, Dict, List

# Module-level state, available for future per-market bookkeeping.
_STATE = {}

NEUTRAL = {
    "triggered": False,
    "direction": None,
    "confidence": 0.0,
    "signal_price": 0.0,
    "entry_price": 0.0,
    "source": "SPOT_SHOCK_OVERREACTION",
    "reason": "neutral",
}


def _compute_returns(prices: List[float]) -> List[float]:
    """Return a list of 1-tick percentage returns from a price series."""
    returns = []
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        if prev == 0:
            returns.append(0.0)
        else:
            returns.append((prices[i] - prev) / prev)
    return returns


def spot_shock_overreaction_signal(**kwargs) -> Dict[str, Any]:
    """Return a signal dict following the new signal interface.

    Accepts the full interface kwargs and ignores any that are not needed.
    """
    spot_price = float(kwargs.get("spot_price", 0.0))
    rem_sec = float(kwargs.get("rem_sec", 0.0))
    elapsed_sec = float(kwargs.get("elapsed_sec", 0.0))
    yp = float(kwargs.get("yp", 0.0))
    np_val = float(kwargs.get("np_val", 0.0))
    spot_history = kwargs.get("spot_history", []) or []

    # Time guards: do not trade in the first or last 5 seconds.
    if rem_sec <= 5 or elapsed_sec <= 5:
        return {**NEUTRAL, "reason": "time_guard"}

    # Need at least 20 historical prices so we can build 20 1-tick returns
    # plus the 5-tick lookback and the 2-tick deceleration check.
    if len(spot_history) < 20 or spot_price <= 0:
        return {**NEUTRAL, "reason": "insufficient_history"}

    price_5_ago = spot_history[-5]
    if price_5_ago <= 0:
        return {**NEUTRAL, "reason": "invalid_history"}

    # 5-tick spot return.
    r5 = (spot_price - price_5_ago) / price_5_ago

    # 20-tick realized volatility (population std of 1-tick returns).
    recent_prices = spot_history[-20:] + [spot_price]
    returns = _compute_returns(recent_prices)
    if len(returns) < 20:
        return {**NEUTRAL, "reason": "insufficient_returns"}

    vol = statistics.pstdev(returns)
    if vol <= 0:
        return {**NEUTRAL, "reason": "zero_volatility"}

    # Shock filter: only trade moves that are large vs recent vol.
    if abs(r5) <= 2.5 * vol:
        return {**NEUTRAL, "reason": "no_shock"}

    # Deceleration check: acceleration of the last two ticks must be
    # opposite in sign to the 5-tick move.
    r1 = (spot_price - spot_history[-1]) / spot_history[-1]
    r1_prev = (spot_history[-1] - spot_history[-2]) / spot_history[-2]
    acceleration = r1 - r1_prev
    if r5 * acceleration >= 0:
        return {**NEUTRAL, "reason": "no_deceleration"}

    # Fade the shock: positive r5 -> NO, negative r5 -> YES.
    if r5 > 0:
        direction = "NO"
        entry_price = float(kwargs.get("no_ask", np_val) or np_val)  # taker fill at ask
    else:
        direction = "YES"
        entry_price = float(kwargs.get("yes_ask", yp) or yp)  # taker fill at ask

    # Entry price cap.
    if entry_price < 0.05 or entry_price > 0.85:
        return {**NEUTRAL, "reason": "entry_price_outside_cap"}

    # Confidence scales with shock magnitude, capped at 1.0.
    confidence = min(abs(r5) / (5.0 * vol + 1e-12), 1.0)

    return {
        "triggered": True,
        "direction": direction,
        "confidence": confidence,
        "signal_price": spot_price,
        "entry_price": entry_price,
        "source": "SPOT_SHOCK_OVERREACTION",
        "reason": (
            f"spot_shock_fade r5={r5:.6f} vol={vol:.6f} "
            f"accel={acceleration:.6f}"
        ),
    }
