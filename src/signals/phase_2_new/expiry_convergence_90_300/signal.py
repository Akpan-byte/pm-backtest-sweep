# CHANGE_SUMMARY
# 2026-07-16  kilo
#   - Created expiry_convergence_90_300 signal module.
#   - Implements the Wave 1 strategy: between 90s and 300s before expiry,
#     buy YES when spot is clearly above strike and YES is cheap,
#     buy NO when spot is clearly below strike and NO is cheap.
#   - Applies INTERFACE time guards (rem_sec > 5, elapsed_sec > 5) and
#     entry price cap (0.05 <= entry_price <= 0.85).
# WHY: Deliver a standalone Polymarket BTC 5m up/down signal per STRATEGY_SPECS.md.

"""
Expiry convergence signal for Polymarket BTC 5m UP/DOWN markets.

Between 90 and 300 seconds before expiry, if spot has moved clearly away
from the strike, the binary should converge toward 1 or 0. This signal
buys the likely winner when it is still cheap.
"""

from typing import Any, Dict

# Module-level state, keyed by market_id, for any cross-snapshot data.
_STATE: Dict[str, Any] = {}


def _neutral(spot_price: float = 0.0) -> Dict[str, Any]:
    """Return a neutral (no-trade) result dict."""
    return {
        "triggered": False,
        "direction": None,
        "confidence": 0.0,
        "signal_price": spot_price,
        "entry_price": 0.0,
        "source": "EXPIRY_CONVERGENCE_90_300",
        "reason": "neutral",
    }


def expiry_convergence_90_300_signal(
    spot_price: float = None,
    strike: float = None,
    rem_sec: float = None,
    elapsed_sec: float = None,
    duration_sec: float = None,
    yp: float = None,
    np_val: float = None,
    yes_ask: float = None,
    no_ask: float = None,
    spot_history: list = None,
    yp_history: list = None,
    np_history: list = None,
    tf_hint: str = None,
    market_id: str = None,
    start_date_iso: str = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate an expiry-convergence signal for a BTC 5m binary market.

    See /config/new_signals/INTERFACE.md for the full keyword interface.
    """
    result = _neutral(spot_price if spot_price is not None else 0.0)

    # Required fields for this strategy.
    if (
        spot_price is None
        or strike is None
        or rem_sec is None
        or elapsed_sec is None
        or yp is None
        or np_val is None
    ):
        result["reason"] = "missing required inputs"
        return result

    if strike <= 0:
        result["reason"] = "invalid strike"
        return result

    # Time guard: do not trade in the first or last 5 seconds.
    if rem_sec <= 5 or elapsed_sec <= 5:
        result["reason"] = "time guard"
        return result

    # Strategy window: only trade between 90s and 300s before expiry.
    if not (90 <= rem_sec <= 300):
        result["reason"] = "outside expiry window"
        return result

    dist = (spot_price - strike) / strike
    abs_dist = abs(dist)
    threshold = 0.0005

    triggered = False
    direction = None
    entry_price = 0.0
    confidence = 0.0
    reason = "neutral"

    if dist > threshold:
        direction = "YES"
        entry_price = kwargs.get("yes_ask", yp)
        if 0.05 <= entry_price <= 0.85:
            triggered = True
            # Confidence scales linearly with distance, capped at 1.0.
            confidence = min(abs_dist / threshold, 1.0)
            reason = (
                f"spot dist {dist:.6f} above strike, YES cheap at {entry_price:.4f}"
            )
        else:
            reason = (
                f"spot dist {dist:.6f} above strike but YES price {entry_price:.4f} "
                "outside entry cap"
            )
    elif dist < -threshold:
        direction = "NO"
        entry_price = kwargs.get("no_ask", np_val)
        if 0.05 <= entry_price <= 0.85:
            triggered = True
            confidence = min(abs_dist / threshold, 1.0)
            reason = (
                f"spot dist {dist:.6f} below strike, NO cheap at {entry_price:.4f}"
            )
        else:
            reason = (
                f"spot dist {dist:.6f} below strike but NO price {entry_price:.4f} "
                "outside entry cap"
            )
    else:
        reason = f"spot dist {dist:.6f} inside neutral band (+/-{threshold})"

    return {
        "triggered": triggered,
        "direction": direction,
        "confidence": confidence,
        "signal_price": spot_price,
        "entry_price": entry_price,
        "source": "EXPIRY_CONVERGENCE_90_300",
        "reason": reason,
    }
