# CHANGE_SUMMARY
# 2026-07-16  assistant
#   - Created kalman_signal_filter signal module per INTERFACE.md and STRATEGY_SPECS.md.
#   - Implements a per-market exponential-Kalman trend estimate (alpha=0.15).
#   - Triggers YES when spot is clearly above the filtered trend and NO when below,
#     subject to time guard, entry-price cap, and strategy-specific <=0.80 threshold.
# WHY: Adds a standalone 5m BTC up/down signal that trades trend deviations from a
#      smoothed spot estimate while keeping all state local and avoiding network calls.

from typing import Any, Dict

# Per-market persistent state for the Kalman estimate, keyed by market_id.
_STATE: Dict[str, Dict[str, float]] = {}


def _update_state(market_id: str, spot_price: float, alpha: float = 0.15) -> float:
    """Return updated exponential-Kalman estimate for this market."""
    if market_id not in _STATE:
        # First observation: initialise estimate to current spot so we do not
        # immediately fire on a single tick.
        _STATE[market_id] = {"estimate": spot_price}
    prev = _STATE[market_id]["estimate"]
    estimate = alpha * spot_price + (1.0 - alpha) * prev
    _STATE[market_id]["estimate"] = estimate
    return estimate


def kalman_signal_filter_signal(**kwargs: Any) -> Dict[str, Any]:
    """Kalman signal filter for Polymarket BTC 5m up/down markets.

    Uses a simple exponential-Kalman estimate of spot price and triggers only
    when spot deviates meaningfully from the smoothed trend.
    """
    spot_price = float(kwargs.get("spot_price", 0.0))
    yp = float(kwargs.get("yp", 0.0))
    np_val = float(kwargs.get("np_val", 0.0))
    rem_sec = float(kwargs.get("rem_sec", 0.0))
    elapsed_sec = float(kwargs.get("elapsed_sec", 0.0))
    market_id = str(kwargs.get("market_id", ""))

    neutral = {
        "triggered": False,
        "direction": None,
        "confidence": 0.0,
        "signal_price": spot_price,
        "entry_price": 0.0,
        "source": "KALMAN_SIGNAL_FILTER",
        "reason": "no signal",
    }

    # Time guard: do not trade in the first or last 5 seconds.
    if rem_sec <= 5.0 or elapsed_sec <= 5.0:
        neutral["reason"] = "time guard"
        return neutral

    # Need a valid spot price to compute the filter.
    if spot_price <= 0.0:
        neutral["reason"] = "invalid spot price"
        return neutral

    estimate = _update_state(market_id, spot_price)

    # Avoid division-by-zero / degenerate estimate.
    if estimate <= 0.0:
        neutral["reason"] = "invalid estimate"
        return neutral

    # YES: spot clearly above filtered trend.
    if spot_price > estimate * 1.0002:
        entry_price = kwargs.get("yes_ask", yp)
        if 0.05 <= entry_price <= 0.80:
            deviation = (spot_price / estimate) - 1.0
            confidence = min(1.0, max(0.0, deviation / 0.002))
            return {
                "triggered": True,
                "direction": "YES",
                "confidence": confidence,
                "signal_price": spot_price,
                "entry_price": entry_price,
                "source": "KALMAN_SIGNAL_FILTER",
                "reason": f"spot {spot_price:.2f} above kalman estimate {estimate:.2f} "
                          f"(deviation {deviation:.6f})",
            }
        else:
            neutral["reason"] = f"YES entry price {entry_price} outside cap"
            neutral["entry_price"] = entry_price
            return neutral

    # NO: spot clearly below filtered trend.
    if spot_price < estimate * 0.9998:
        entry_price = kwargs.get("no_ask", np_val)
        if 0.05 <= entry_price <= 0.80:
            deviation = 1.0 - (spot_price / estimate)
            confidence = min(1.0, max(0.0, deviation / 0.002))
            return {
                "triggered": True,
                "direction": "NO",
                "confidence": confidence,
                "signal_price": spot_price,
                "entry_price": entry_price,
                "source": "KALMAN_SIGNAL_FILTER",
                "reason": f"spot {spot_price:.2f} below kalman estimate {estimate:.2f} "
                          f"(deviation {deviation:.6f})",
            }
        else:
            neutral["reason"] = f"NO entry price {entry_price} outside cap"
            neutral["entry_price"] = entry_price
            return neutral

    neutral["reason"] = "spot within kalman band"
    return neutral
