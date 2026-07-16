# CHANGE_SUMMARY
# 2026-07-16  kilo
#   - Created vol_compression_breakout signal module.
#   - Implements ATR compression/expansion breakout logic per STRATEGY_SPECS.md.
#   - Tracks per-market compression tick count in _STATE.
# WHY: Adds a standalone Polymarket BTC 5m signal that fires after volatility
#      compression followed by directional expansion.

"""Volatility compression -> breakout signal for Polymarket BTC 5m markets."""

from typing import Any, Dict, List, Optional

_STATE: Dict[str, Dict[str, Any]] = {}


def _neutral(spot: float, reason: str) -> Dict[str, Any]:
    """Return a non-triggered signal dict."""
    return {
        "triggered": False,
        "direction": None,
        "confidence": 0.0,
        "signal_price": spot,
        "entry_price": 0.0,
        "source": "VOL_COMPRESSION_BREAKOUT",
        "reason": reason,
    }


def _atr(prices: List[float], lookback: int) -> Optional[float]:
    """Mean absolute price change over the last `lookback` intervals."""
    if len(prices) < lookback + 1:
        return None
    diffs = [abs(prices[i] - prices[i - 1]) for i in range(-lookback, 0)]
    return sum(diffs) / lookback


def _recent_range(prices: List[float], lookback: int) -> Optional[tuple]:
    """Return (high, low) over the last `lookback` closes."""
    if len(prices) < lookback:
        return None
    window = prices[-lookback:]
    return max(window), min(window)


def vol_compression_breakout_signal(**kwargs) -> Dict[str, Any]:
    """
    Fire when short-term ATR is compressed vs medium-term ATR for >=5 ticks,
    then a sudden directional expansion breaks out.

    Follows the new-signal INTERFACE.md exactly.
    """
    spot_price: float = kwargs.get("spot_price", 0.0)
    rem_sec: float = kwargs.get("rem_sec", 0.0)
    elapsed_sec: float = kwargs.get("elapsed_sec", 0.0)
    yp: float = kwargs.get("yp", 0.0)
    np_val: float = kwargs.get("np_val", 0.0)
    spot_history: List[float] = kwargs.get("spot_history") or []
    market_id: str = kwargs.get("market_id", "")

    # Basic input guard.
    if not market_id or spot_price <= 0:
        return _neutral(spot_price, "missing required inputs")

    # Time guard: no trades in the first/last 5 seconds.
    if rem_sec <= 5 or elapsed_sec <= 5:
        return _neutral(spot_price, "time guard")

    # Build a price series that definitely ends with the current spot price.
    prices = list(spot_history)
    if not prices or prices[-1] != spot_price:
        prices.append(spot_price)

    atr10 = _atr(prices, 10)
    atr30 = _atr(prices, 30)
    if atr10 is None or atr30 is None or atr30 <= 0:
        return _neutral(spot_price, "insufficient history")

    # Track sustained compression.
    state = _STATE.setdefault(market_id, {"compression_count": 0})
    compressed = atr10 < 0.6 * atr30
    if compressed:
        state["compression_count"] += 1
    else:
        state["compression_count"] = 0

    if state["compression_count"] < 5:
        return _neutral(spot_price, "compression not sustained")

    # Expansion: current 3-tick range must exceed 1.5x the 10-tick ATR.
    range_3 = _recent_range(prices, 3)
    if range_3 is None:
        return _neutral(spot_price, "insufficient range window")
    high3, low3 = range_3
    expansion = high3 - low3
    if expansion <= 0:
        return _neutral(spot_price, "no range")

    threshold = 1.5 * atr10
    if expansion <= threshold:
        return _neutral(spot_price, "no expansion")

    # Direction: close near the top of the 3-tick range -> YES, bottom -> NO.
    position = (spot_price - low3) / expansion
    if position >= 0.7:
        direction = "YES"
        entry_price = kwargs.get("yes_ask", yp)
    elif position <= 0.3:
        direction = "NO"
        entry_price = kwargs.get("no_ask", np_val)
    else:
        return _neutral(spot_price, "expansion direction unclear")

    if entry_price is None or entry_price < 0.05 or entry_price > 0.85:
        return _neutral(spot_price, "entry price outside cap")

    # Confidence scales with how far expansion exceeds the threshold.
    confidence = min(0.95, 0.4 + 0.6 * (expansion / threshold - 1.0))

    return {
        "triggered": True,
        "direction": direction,
        "confidence": confidence,
        "signal_price": spot_price,
        "entry_price": entry_price,
        "source": "VOL_COMPRESSION_BREAKOUT",
        "reason": (
            f"{direction}: ATR10={atr10:.4g} compressed vs ATR30={atr30:.4g}; "
            f"3-tick range={expansion:.4g} > 1.5*ATR10={threshold:.4g}"
        ),
    }
