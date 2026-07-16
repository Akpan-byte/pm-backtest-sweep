# CHANGE_SUMMARY
# 2026-07-16  kilo
#   - Created noise_signal_decomp signal module per INTERFACE.md and STRATEGY_SPECS.md.
#   - Implements fast/slow EMA pullback crossover logic with time and entry-price guards.
# WHY: Provide a standalone Polymarket BTC 5m signal that trades trend pullbacks.

"""Noise vs signal decomposition signal for Polymarket BTC 5m up/down markets.

Logic:
- Maintain a fast EMA (alpha=0.4) and a slow EMA (alpha=0.1) of spot price per market.
- A YES trigger requires the slow EMA to be rising and the fast EMA to have just
  crossed back above the slow EMA (pullback to trend).
- A NO trigger requires the slow EMA to be falling and the fast EMA to have just
  crossed back below the slow EMA.
- Time guard and entry-price cap are enforced exactly per INTERFACE.md.
"""

from typing import Any, Dict, Optional

# Module-level state keyed by market_id.
_STATE: Dict[str, Dict[str, float]] = {}

FAST_ALPHA = 0.4
SLOW_ALPHA = 0.1

SOURCE = "NOISE_SIGNAL_DECOMP"


def _neutral(signal_price: float, reason: str) -> Dict[str, Any]:
    """Return a neutral signal dict."""
    return {
        "triggered": False,
        "direction": None,
        "confidence": 0.0,
        "signal_price": float(signal_price),
        "entry_price": 0.0,
        "source": SOURCE,
        "reason": reason,
    }


def noise_signal_decomp_signal(**kwargs) -> Dict[str, Any]:
    """Generate a pullback-in-trend signal from fast/slow EMA decomposition.

    See /config/new_signals/INTERFACE.md for the accepted keyword arguments.
    """
    spot_price = float(kwargs.get("spot_price", 0.0))
    rem_sec = float(kwargs.get("rem_sec", 0.0))
    elapsed_sec = float(kwargs.get("elapsed_sec", 0.0))
    yp = float(kwargs.get("yp", 0.0))
    np_val = float(kwargs.get("np_val", 0.0))
    market_id = kwargs.get("market_id")

    if market_id is None:
        return _neutral(spot_price, "missing market_id")
    market_id = str(market_id)

    # Rule 1: do not trigger in first/last 5 seconds.
    if rem_sec <= 5 or elapsed_sec <= 5:
        return _neutral(spot_price, "time guard")

    prev = _STATE.get(market_id)

    if prev is None:
        # First observation: seed both EMAs to current spot.
        _STATE[market_id] = {
            "fast": spot_price,
            "slow": spot_price,
            "prev_fast": spot_price,
            "prev_slow": spot_price,
        }
        return _neutral(spot_price, "initializing EMA state")

    prev_fast = prev["fast"]
    prev_slow = prev["slow"]

    # Update EMAs.
    fast = FAST_ALPHA * spot_price + (1.0 - FAST_ALPHA) * prev_fast
    slow = SLOW_ALPHA * spot_price + (1.0 - SLOW_ALPHA) * prev_slow

    # Slow trend direction.
    slow_rising = slow > prev_slow
    slow_falling = slow < prev_slow

    # Fast "noise" crossed back toward the slow "signal" this tick.
    crossed_above = prev_fast <= prev_slow and fast > slow
    crossed_below = prev_fast >= prev_slow and fast < slow

    # Persist updated state.
    _STATE[market_id] = {
        "fast": fast,
        "slow": slow,
        "prev_fast": prev_fast,
        "prev_slow": prev_slow,
    }

    # Determine direction and entry price.
    direction: Optional[str] = None
    entry_price = 0.0
    reason = "no pullback crossover"

    if slow_rising and crossed_above:
        direction = "YES"
        entry_price = yp
        reason = "slow EMA rising and fast EMA crossed back above slow EMA"
    elif slow_falling and crossed_below:
        direction = "NO"
        entry_price = np_val
        reason = "slow EMA falling and fast EMA crossed back below slow EMA"
    else:
        return _neutral(spot_price, reason)

    # Rule 2: entry price cap.
    if entry_price < 0.05 or entry_price > 0.85:
        return _neutral(spot_price, f"entry price {entry_price:.4f} outside [0.05, 0.85]")

    # Confidence scales with the fast/slow deviation, floored at 0.5.
    denom = slow if slow != 0.0 else 1.0
    deviation = abs(fast - slow) / denom
    confidence = min(0.95, max(0.5, 0.5 + deviation * 500.0))

    return {
        "triggered": True,
        "direction": direction,
        "confidence": confidence,
        "signal_price": spot_price,
        "entry_price": entry_price,
        "source": SOURCE,
        "reason": reason,
    }
