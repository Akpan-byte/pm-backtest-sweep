# CHANGE_SUMMARY
# 2026-07-16  assistant
#   - Created strike_distance_yes signal module per INTERFACE.md and the active wave-2 strategy plan.
#   - spot > strike +10bps and yp <= 0.75 -> YES
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



def strike_distance_yes_signal(**kwargs: Any) -> Dict[str, Any]:
    """Simple 5m BTC up/down signal: spot > strike +10bps and yp <= 0.75 -> YES"""
    spot_price = float(kwargs.get("spot_price", 0.0))
    strike = float(kwargs.get("strike", 0.0))
    rem_sec = float(kwargs.get("rem_sec", 0.0))
    elapsed_sec = float(kwargs.get("elapsed_sec", 0.0))
    yp = float(kwargs.get("yp", 0.0))
    np_val = float(kwargs.get("np_val", 0.0))
    spot_history: List[float] = kwargs.get("spot_history", [])
    yp_history: List[float] = kwargs.get("yp_history", [])
    np_history: List[float] = kwargs.get("np_history", [])
    source = "STRIKE_DISTANCE_YES"

    # Time guard: do not trade in the first or last 5 seconds.
    if rem_sec <= 5.0 or elapsed_sec <= 5.0:
        return _neutral(spot_price, "time guard", source)

    if strike <= 0.0:
        return _neutral(spot_price, "invalid strike", source)
    dist = (spot_price - strike) / strike
    if dist > 0.001 and yp <= 0.75:
        entry = yp
        if 0.05 <= entry <= 0.85:
            return {
                "triggered": True, "direction": "YES",
                "confidence": min(1.0, max(0.0, dist / 0.003)),
                "signal_price": spot_price, "entry_price": entry,
                "source": source,
                "reason": f"spot {spot_price:.2f} {dist:.5f} above strike {strike:.2f}",
            }
    return _neutral(spot_price, "no signal", source)

