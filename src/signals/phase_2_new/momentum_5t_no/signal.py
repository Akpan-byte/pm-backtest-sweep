# CHANGE_SUMMARY
# 2026-07-16  assistant
#   - Created momentum_5t_no signal module per INTERFACE.md and the active wave-2 strategy plan.
#   - spot lower than 5 ticks ago and np <= 0.80 -> NO
#   - Respects the standard time guard (no trade in first/last 5s) and entry-price
#     cap [0.05, 0.85].
# WHY: Wave-1 strategies were mostly too restrictive and produced zero trades. This
#      module is intentionally simple so it actually fires on realistic 5m BTC data.
from typing import Any, Dict, List

# Per-market persistent state keyed by market_id, as required by INTERFACE.md.
_STATE: Dict[str, Any] = {}


def _neutral(spot_price: float, reason: str, source: str) -> Dict[str, Any]:
    return {
        "triggered": False,
        "direction": None,
        "confidence": 0.0,
        "signal_price": spot_price,
        "entry_price": 0.0,
        "source": source,
        "reason": reason,
    }



def momentum_5t_no_signal(**kwargs: Any) -> Dict[str, Any]:
    """Simple 5m BTC up/down signal: spot lower than 5 ticks ago and np <= 0.80 -> NO"""
    spot_price = float(kwargs.get("spot_price", 0.0))
    strike = float(kwargs.get("strike", 0.0))
    rem_sec = float(kwargs.get("rem_sec", 0.0))
    elapsed_sec = float(kwargs.get("elapsed_sec", 0.0))
    yp = float(kwargs.get("yp", 0.0))
    np_val = float(kwargs.get("np_val", 0.0))
    spot_history: List[float] = kwargs.get("spot_history", [])
    yp_history: List[float] = kwargs.get("yp_history", [])
    np_history: List[float] = kwargs.get("np_history", [])
    source = "MOMENTUM_5T_NO"

    # Time guard: do not trade in the first or last 5 seconds.
    if rem_sec <= 5.0 or elapsed_sec <= 5.0:
        return _neutral(spot_price, "time guard", source)

    if len(spot_history) < 5:
        return _neutral(spot_price, "insufficient spot history", source)
    if spot_price < spot_history[-5] and np_val <= 0.80:
        entry = np_val
        if 0.05 <= entry <= 0.85:
            ret = (spot_price - spot_history[-5]) / spot_history[-5] if spot_history[-5] != 0 else 0.0
            return {
                "triggered": True, "direction": "NO",
                "confidence": min(1.0, max(0.0, abs(ret) / 0.001)),
                "signal_price": spot_price, "entry_price": entry,
                "source": source,
                "reason": f"spot {spot_price:.2f} below 5-tick ago {spot_history[-5]:.2f}",
            }
    return _neutral(spot_price, "no signal", source)

